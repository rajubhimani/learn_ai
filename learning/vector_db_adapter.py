from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    ELASTICSEARCH = "elasticsearch"
    PGVECTOR = "pgvector"

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: Optional[float] = None

@dataclass
class CollectionInfo:
    name: str
    document_count: int
    dimension: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class VectorDatabaseAdapter(ABC):
    """Abstract base class for vector database adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self._setup_embedding_model()
    
    def _setup_embedding_model(self):
        """Setup embedding model if needed"""
        if self.config.get('auto_embed', True):
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Manual embeddings required.")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.embedding_model:
            return self.embedding_model.encode([text])[0].tolist()
        else:
            raise ValueError("No embedding model available. Provide embedding manually or install sentence-transformers.")
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection"""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections"""
        pass
    
    @abstractmethod
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to collection"""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query: Union[str, List[float]], 
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID"""
        pass
    
    @abstractmethod
    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """Get collection information"""
        pass
    
    @abstractmethod
    def update_document(self, collection_name: str, document: Document) -> bool:
        """Update an existing document"""
        pass

class ChromaDBAdapter(VectorDatabaseAdapter):
    """ChromaDB adapter implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import chromadb
            from chromadb.config import Settings
            
            if config.get('persistent', True):
                self.client = chromadb.PersistentClient(
                    path=config.get('path', './chroma_db')
                )
            else:
                self.client = chromadb.Client()
            
            logger.info("ChromaDB client initialized")
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
    
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": kwargs.get("metric", "cosine")}
            )
            return True
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        try:
            self.client.delete_collection(name=name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
            
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings if not provided
            embeddings = []
            for doc in documents:
                if doc.embedding:
                    embeddings.append(doc.embedding)
                else:
                    embeddings.append(self._generate_embedding(doc.content))
            
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            return False
    
    def search(self, collection_name: str, query: Union[str, List[float]], 
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        try:
            collection = self.client.get_collection(name=collection_name)
            
            if isinstance(query, str):
                query_embedding = [self._generate_embedding(query)]
            else:
                query_embedding = [query]
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=filters
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    distance=results['distances'][0][i]
                )
                search_results.append(result)
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {e}")
            return []
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Document]:
        try:
            collection = self.client.get_collection(name=collection_name)
            result = collection.get(ids=[doc_id], include=['documents', 'metadatas', 'embeddings'])
            
            if result['ids']:
                return Document(
                    id=result['ids'][0],
                    content=result['documents'][0],
                    metadata=result['metadatas'][0],
                    embedding=result['embeddings'][0] if result['embeddings'] else None
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=doc_ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return CollectionInfo(
                name=collection_name,
                document_count=count,
                metadata=collection.metadata
            )
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return CollectionInfo(name=collection_name, document_count=0)
    
    def update_document(self, collection_name: str, document: Document) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
            
            embedding = document.embedding
            if not embedding:
                embedding = self._generate_embedding(document.content)
            
            collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata],
                embeddings=[embedding]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

class PineconeAdapter(VectorDatabaseAdapter):
    """Pinecone adapter implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import pinecone
            from pinecone import Pinecone
            
            api_key = config.get('api_key') or os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key required")
            
            self.pc = Pinecone(api_key=api_key)
            self.environment = config.get('environment', 'us-east-1-aws')
            
            logger.info("Pinecone client initialized")
        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
    
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=kwargs.get('metric', 'cosine'),
                spec=kwargs.get('spec', {'serverless': {'cloud': 'aws', 'region': 'us-east-1'}})
            )
            return True
        except Exception as e:
            logger.error(f"Error creating index {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        try:
            self.pc.delete_index(name)
            return True
        except Exception as e:
            logger.error(f"Error deleting index {name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        try:
            indexes = self.pc.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return []
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        try:
            index = self.pc.Index(collection_name)
            
            vectors = []
            for doc in documents:
                embedding = doc.embedding
                if not embedding:
                    embedding = self._generate_embedding(doc.content)
                
                vectors.append({
                    'id': doc.id,
                    'values': embedding,
                    'metadata': {**doc.metadata, 'content': doc.content}
                })
            
            index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            return False
    
    def search(self, collection_name: str, query: Union[str, List[float]], 
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        try:
            index = self.pc.Index(collection_name)
            
            if isinstance(query, str):
                query_vector = self._generate_embedding(query)
            else:
                query_vector = query
            
            results = index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True,
                filter=filters
            )
            
            search_results = []
            for match in results['matches']:
                metadata = match['metadata'].copy()
                content = metadata.pop('content', '')
                
                result = SearchResult(
                    id=match['id'],
                    content=content,
                    metadata=metadata,
                    score=match['score']
                )
                search_results.append(result)
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {e}")
            return []
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Document]:
        try:
            index = self.pc.Index(collection_name)
            result = index.fetch(ids=[doc_id])
            
            if doc_id in result['vectors']:
                vector_data = result['vectors'][doc_id]
                metadata = vector_data['metadata'].copy()
                content = metadata.pop('content', '')
                
                return Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=vector_data['values']
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        try:
            index = self.pc.Index(collection_name)
            index.delete(ids=doc_ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        try:
            index = self.pc.Index(collection_name)
            stats = index.describe_index_stats()
            
            return CollectionInfo(
                name=collection_name,
                document_count=stats['total_vector_count'],
                dimension=stats.get('dimension'),
                metadata=stats
            )
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return CollectionInfo(name=collection_name, document_count=0)
    
    def update_document(self, collection_name: str, document: Document) -> bool:
        # Pinecone uses upsert for updates
        return self.add_documents(collection_name, [document])

class WeaviateAdapter(VectorDatabaseAdapter):
    """Weaviate adapter implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import weaviate
            
            url = config.get('url', 'http://localhost:8080')
            api_key = config.get('api_key')
            
            if api_key:
                self.client = weaviate.Client(
                    url=url,
                    auth_client_secret=weaviate.AuthApiKey(api_key=api_key)
                )
            else:
                self.client = weaviate.Client(url=url)
            
            logger.info("Weaviate client initialized")
        except ImportError:
            raise ImportError("weaviate-client not installed. Run: pip install weaviate-client")
    
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        try:
            class_obj = {
                "class": name,
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"]
                    }
                ]
            }
            
            self.client.schema.create_class(class_obj)
            return True
        except Exception as e:
            logger.error(f"Error creating class {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        try:
            self.client.schema.delete_class(name)
            return True
        except Exception as e:
            logger.error(f"Error deleting class {name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        try:
            schema = self.client.schema.get()
            return [cls['class'] for cls in schema.get('classes', [])]
        except Exception as e:
            logger.error(f"Error listing classes: {e}")
            return []
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        try:
            with self.client.batch as batch:
                for doc in documents:
                    embedding = doc.embedding
                    if not embedding:
                        embedding = self._generate_embedding(doc.content)
                    
                    batch.add_data_object(
                        data_object={
                            "content": doc.content,
                            "metadata": doc.metadata
                        },
                        class_name=collection_name,
                        uuid=doc.id,
                        vector=embedding
                    )
            return True
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            return False
    
    def search(self, collection_name: str, query: Union[str, List[float]], 
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        try:
            if isinstance(query, str):
                query_vector = self._generate_embedding(query)
            else:
                query_vector = query
            
            query_builder = self.client.query.get(collection_name, ["content", "metadata"]) \
                               .with_near_vector({"vector": query_vector}) \
                               .with_limit(limit) \
                               .with_additional(["id", "certainty"])
            
            if filters:
                query_builder = query_builder.with_where(filters)
            
            results = query_builder.do()
            
            search_results = []
            for item in results['data']['Get'][collection_name]:
                result = SearchResult(
                    id=item['_additional']['id'],
                    content=item['content'],
                    metadata=item['metadata'],
                    score=item['_additional']['certainty']
                )
                search_results.append(result)
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {e}")
            return []
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Document]:
        try:
            result = self.client.data_object.get_by_id(doc_id, class_name=collection_name)
            
            if result:
                return Document(
                    id=doc_id,
                    content=result['properties']['content'],
                    metadata=result['properties']['metadata']
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        try:
            for doc_id in doc_ids:
                self.client.data_object.delete(uuid=doc_id, class_name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        try:
            result = self.client.query.aggregate(collection_name).with_meta_count().do()
            count = result['data']['Aggregate'][collection_name][0]['meta']['count']
            
            return CollectionInfo(
                name=collection_name,
                document_count=count
            )
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return CollectionInfo(name=collection_name, document_count=0)
    
    def update_document(self, collection_name: str, document: Document) -> bool:
        try:
            embedding = document.embedding
            if not embedding:
                embedding = self._generate_embedding(document.content)
            
            self.client.data_object.replace(
                uuid=document.id,
                class_name=collection_name,
                data_object={
                    "content": document.content,
                    "metadata": document.metadata
                },
                vector=embedding
            )
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

class VectorDatabaseFactory:
    """Factory class for creating vector database adapters"""
    
    _adapters = {
        VectorDBType.CHROMADB: ChromaDBAdapter,
        VectorDBType.PINECONE: PineconeAdapter,
        VectorDBType.WEAVIATE: WeaviateAdapter,
        # Add more adapters as implemented
    }
    
    @classmethod
    def create_adapter(cls, db_type: VectorDBType, config: Dict[str, Any]) -> VectorDatabaseAdapter:
        """Create a vector database adapter"""
        if db_type not in cls._adapters:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        adapter_class = cls._adapters[db_type]
        return adapter_class(config)
    
    @classmethod
    def register_adapter(cls, db_type: VectorDBType, adapter_class):
        """Register a custom adapter"""
        cls._adapters[db_type] = adapter_class

class MultiVectorDBManager:
    """Manager class for handling multiple vector databases"""
    
    def __init__(self):
        self.adapters: Dict[str, VectorDatabaseAdapter] = {}
        self.default_adapter: Optional[str] = None
    
    def add_adapter(self, name: str, db_type: VectorDBType, config: Dict[str, Any], 
                   set_as_default: bool = False) -> bool:
        """Add a vector database adapter"""
        try:
            adapter = VectorDatabaseFactory.create_adapter(db_type, config)
            self.adapters[name] = adapter
            
            if set_as_default or not self.default_adapter:
                self.default_adapter = name
            
            logger.info(f"Added adapter '{name}' of type {db_type.value}")
            return True
        except Exception as e:
            logger.error(f"Error adding adapter '{name}': {e}")
            return False
    
    def get_adapter(self, name: Optional[str] = None) -> Optional[VectorDatabaseAdapter]:
        """Get an adapter by name, or the default adapter"""
        if name is None:
            name = self.default_adapter
        
        return self.adapters.get(name)
    
    def list_adapters(self) -> List[str]:
        """List all available adapters"""
        return list(self.adapters.keys())
    
    def remove_adapter(self, name: str) -> bool:
        """Remove an adapter"""
        if name in self.adapters:
            del self.adapters[name]
            if self.default_adapter == name:
                self.default_adapter = next(iter(self.adapters), None)
            return True
        return False
    
    def search_all(self, collection_name: str, query: Union[str, List[float]], 
                   limit: int = 10, filters: Optional[Dict] = None) -> Dict[str, List[SearchResult]]:
        """Search across all adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = adapter.search(collection_name, query, limit, filters)
            except Exception as e:
                logger.error(f"Error searching in adapter '{name}': {e}")
                results[name] = []
        
        return results

# Example usage and configuration
if __name__ == "__main__":
    # Initialize the multi-database manager
    manager = MultiVectorDBManager()
    
    # Add ChromaDB adapter
    chroma_config = {
        'persistent': True,
        'path': './chroma_db',
        'auto_embed': True,
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    manager.add_adapter('chroma', VectorDBType.CHROMADB, chroma_config, set_as_default=True)
    
    # Add Pinecone adapter (if API key available)
    if os.getenv('PINECONE_API_KEY'):
        pinecone_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': 'us-east-1-aws',
            'auto_embed': True
        }
        manager.add_adapter('pinecone', VectorDBType.PINECONE, pinecone_config)
    
    # Example usage
    adapter = manager.get_adapter()  # Get default adapter
    
    # Create collection
    adapter.create_collection('test_collection', dimension=384)
    
    # Add documents
    documents = [
        Document(
            id='doc1',
            content='This is a test document about machine learning.',
            metadata={'category': 'ML', 'source': 'test'}
        ),
        Document(
            id='doc2', 
            content='This document discusses natural language processing.',
            metadata={'category': 'NLP', 'source': 'test'}
        )
    ]
    
    adapter.add_documents('test_collection', documents)
    
    # Search
    results = adapter.search('test_collection', 'machine learning', limit=5)
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"- {result.id}: {result.content[:100]}... (score: {result.score:.3f})")
    
    # Get collection info
    info = adapter.get_collection_info('test_collection')
    print(f"Collection info: {info}")
