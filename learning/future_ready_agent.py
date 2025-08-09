from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Set
import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
import logging
from contextlib import asynccontextmanager

# LangChain imports
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone, Weaviate, Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import GooglePalm, Cohere
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Vector DB specific imports
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
import boto3
from opensearchpy import OpenSearch
import weaviate

import redis
import structlog

logger = structlog.get_logger()

# Enums
class ModelProvider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"

class VectorDBProvider(str, Enum):
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    AWS_OPENSEARCH = "aws_opensearch"
    AWS_KENDRA = "aws_kendra"
    AZURE_SEARCH = "azure_search"
    FAISS = "faiss"

class CollectionSelectionMode(str, Enum):
    SINGLE = "single"
    MULTIPLE = "multiple" 
    ALL = "all"

# Pydantic Models
class Collection(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    provider: VectorDBProvider
    connection_config: Dict[str, Any]
    embedding_model: str
    vector_dimension: int
    document_count: int = 0
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True

class CollectionSearchResult(BaseModel):
    collection_id: str
    collection_name: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    model_provider: ModelProvider = ModelProvider.CLAUDE
    
    # Collection selection options
    collection_selection_mode: CollectionSelectionMode = CollectionSelectionMode.ALL
    selected_collections: Optional[List[str]] = None  # Collection IDs
    collection_filters: Optional[Dict[str, Any]] = None  # Filter by tags, provider, etc.
    
    # Search parameters
    top_k_per_collection: int = Field(default=3, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_results: bool = True
    
    # Model parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0, le=8192)
    stream: bool = False

class ChatResponse(BaseModel):
    message: str
    model_used: str
    tokens_used: int
    response_time: float
    collections_searched: List[str]
    search_results: List[CollectionSearchResult]
    total_documents_found: int

class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    provider: VectorDBProvider
    connection_config: Dict[str, Any]
    embedding_model: str = "openai"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Vector Database Adapters
class VectorDBAdapter:
    """Base adapter for vector databases"""
    
    async def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        raise NotImplementedError
    
    async def get_collection_info(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        raise NotImplementedError

class QdrantAdapter(VectorDBAdapter):
    def __init__(self, config: Dict[str, Any], embedding_model):
        self.client = qdrant_client.QdrantClient(
            url=config.get("url", "localhost"),
            port=config.get("port", 6333),
            api_key=config.get("api_key"),
            https=config.get("https", False)
        )
        self.collection_name = config["collection_name"]
        self.embedding_model = embedding_model
    
    async def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        try:
            # Get query embedding
            query_embedding = await self.embedding_model.aembed_query(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold
            )
            
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.payload.get("text", ""),
                    metadata={
                        "source": result.payload.get("source", ""),
                        "score": result.score,
                        "id": result.id,
                        **result.payload
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    async def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vector_count": info.points_count,
                "vector_dimension": info.config.params.vectors.size,
                "status": "healthy"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except:
            return False

class AWSOpenSearchAdapter(VectorDBAdapter):
    def __init__(self, config: Dict[str, Any], embedding_model):
        self.client = OpenSearch([{
            'host': config['host'],
            'port': config.get('port', 443)
        }], 
        http_auth=(config['username'], config['password']),
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False)
        
        self.index_name = config["index_name"]
        self.embedding_model = embedding_model
    
    async def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        try:
            query_embedding = await self.embedding_model.aembed_query(query)
            
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "content_vector": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                },
                "min_score": threshold
            }
            
            response = self.client.search(
                body=search_body,
                index=self.index_name
            )
            
            documents = []
            for hit in response['hits']['hits']:
                doc = Document(
                    page_content=hit['_source'].get('content', ''),
                    metadata={
                        "source": hit['_source'].get('source', ''),
                        "score": hit['_score'],
                        "id": hit['_id'],
                        **hit['_source']
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"AWS OpenSearch search failed: {e}")
            return []
    
    async def get_collection_info(self) -> Dict[str, Any]:
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return {
                "document_count": stats['indices'][self.index_name]['total']['docs']['count'],
                "status": "healthy"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False

class PineconeAdapter(VectorDBAdapter):
    def __init__(self, config: Dict[str, Any], embedding_model):
        import pinecone
        pinecone.init(
            api_key=config["api_key"],
            environment=config["environment"]
        )
        self.index = pinecone.Index(config["index_name"])
        self.embedding_model = embedding_model
    
    async def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        try:
            query_embedding = await self.embedding_model.aembed_query(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            documents = []
            for match in results['matches']:
                if match['score'] >= threshold:
                    doc = Document(
                        page_content=match['metadata'].get('text', ''),
                        metadata={
                            "score": match['score'],
                            "id": match['id'],
                            **match['metadata']
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    async def get_collection_info(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                "vector_count": stats['total_vector_count'],
                "vector_dimension": stats['dimension'],
                "status": "healthy"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> bool:
        try:
            self.index.describe_index_stats()
            return True
        except:
            return False

class CollectionManager:
    """Manages collections across different vector database providers"""
    
    def __init__(self, embedding_configs: Dict[str, Any]):
        self.collections: Dict[str, Collection] = {}
        self.adapters: Dict[str, VectorDBAdapter] = {}
        self.embedding_models = self._initialize_embeddings(embedding_configs)
    
    def _initialize_embeddings(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize different embedding models"""
        models = {}
        for name, config in configs.items():
            if config["type"] == "openai":
                models[name] = OpenAIEmbeddings(openai_api_key=config["api_key"])
            elif config["type"] == "huggingface":
                models[name] = HuggingFaceEmbeddings(model_name=config["model_name"])
        return models
    
    def _create_adapter(self, collection: Collection) -> VectorDBAdapter:
        """Factory method to create vector DB adapters"""
        embedding_model = self.embedding_models[collection.embedding_model]
        
        if collection.provider == VectorDBProvider.QDRANT:
            return QdrantAdapter(collection.connection_config, embedding_model)
        elif collection.provider == VectorDBProvider.AWS_OPENSEARCH:
            return AWSOpenSearchAdapter(collection.connection_config, embedding_model)
        elif collection.provider == VectorDBProvider.PINECONE:
            return PineconeAdapter(collection.connection_config, embedding_model)
        else:
            raise ValueError(f"Unsupported provider: {collection.provider}")
    
    async def register_collection(self, collection_data: CollectionCreateRequest) -> Collection:
        """Register a new collection"""
        collection_id = str(uuid.uuid4())
        
        collection = Collection(
            id=collection_id,
            name=collection_data.name,
            description=collection_data.description,
            provider=collection_data.provider,
            connection_config=collection_data.connection_config,
            embedding_model=collection_data.embedding_model,
            vector_dimension=768,  # Would be detected from actual collection
            created_at=datetime.now(),
            tags=collection_data.tags,
            metadata=collection_data.metadata
        )
        
        # Create and test adapter
        try:
            adapter = self._create_adapter(collection)
            if await adapter.health_check():
                info = await adapter.get_collection_info()
                collection.document_count = info.get("document_count", 0)
                collection.vector_dimension = info.get("vector_dimension", 768)
                
                self.collections[collection_id] = collection
                self.adapters[collection_id] = adapter
                
                logger.info(f"Registered collection: {collection.name} ({collection.provider})")
                return collection
            else:
                raise ValueError("Health check failed for collection")
                
        except Exception as e:
            logger.error(f"Failed to register collection {collection.name}: {e}")
            raise HTTPException(status_code=400, detail=f"Collection registration failed: {e}")
    
    def get_collections_by_filters(self, filters: Optional[Dict[str, Any]] = None) -> List[Collection]:
        """Get collections based on filters"""
        collections = list(self.collections.values())
        
        if not filters:
            return [c for c in collections if c.is_active]
        
        filtered = []
        for collection in collections:
            if not collection.is_active:
                continue
                
            # Filter by provider
            if "provider" in filters and collection.provider not in filters["provider"]:
                continue
            
            # Filter by tags
            if "tags" in filters:
                required_tags = set(filters["tags"])
                collection_tags = set(collection.tags)
                if not required_tags.intersection(collection_tags):
                    continue
            
            # Filter by metadata
            if "metadata" in filters:
                for key, value in filters["metadata"].items():
                    if collection.metadata.get(key) != value:
                        continue
            
            filtered.append(collection)
        
        return filtered
    
    def resolve_collection_selection(self, 
                                   mode: CollectionSelectionMode,
                                   selected_collections: Optional[List[str]] = None,
                                   filters: Optional[Dict[str, Any]] = None) -> List[Collection]:
        """Resolve which collections to search based on selection mode"""
        
        if mode == CollectionSelectionMode.ALL:
            return self.get_collections_by_filters(filters)
        
        elif mode == CollectionSelectionMode.MULTIPLE:
            if not selected_collections:
                raise ValueError("selected_collections required for MULTIPLE mode")
            
            collections = []
            for collection_id in selected_collections:
                if collection_id in self.collections and self.collections[collection_id].is_active:
                    collection = self.collections[collection_id]
                    # Apply filters if provided
                    if not filters or self._passes_filters(collection, filters):
                        collections.append(collection)
            return collections
        
        elif mode == CollectionSelectionMode.SINGLE:
            if not selected_collections or len(selected_collections) != 1:
                raise ValueError("Exactly one collection required for SINGLE mode")
            
            collection_id = selected_collections[0]
            if collection_id not in self.collections:
                raise ValueError(f"Collection {collection_id} not found")
            
            collection = self.collections[collection_id]
            if not collection.is_active:
                raise ValueError(f"Collection {collection_id} is not active")
            
            # Apply filters if provided
            if filters and not self._passes_filters(collection, filters):
                return []
            
            return [collection]
        
        return []
    
    def _passes_filters(self, collection: Collection, filters: Dict[str, Any]) -> bool:
        """Check if collection passes the given filters"""
        if "provider" in filters and collection.provider not in filters["provider"]:
            return False
        
        if "tags" in filters:
            required_tags = set(filters["tags"])
            collection_tags = set(collection.tags)
            if not required_tags.intersection(collection_tags):
                return False
        
        return True
    
    async def search_across_collections(self, 
                                      query: str,
                                      collections: List[Collection],
                                      top_k_per_collection: int = 3,
                                      threshold: float = 0.7,
                                      rerank: bool = True) -> List[CollectionSearchResult]:
        """Search across multiple collections"""
        
        search_tasks = []
        for collection in collections:
            if collection.id in self.adapters:
                adapter = self.adapters[collection.id]
                task = self._search_collection(adapter, collection, query, top_k_per_collection, threshold)
                search_tasks.append(task)
        
        # Execute searches concurrently
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        search_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for collection {collections[i].name}: {result}")
                continue
            
            if result.documents:  # Only include collections with results
                search_results.append(result)
        
        # Rerank results across collections if requested
        if rerank and len(search_results) > 1:
            search_results = self._rerank_results(search_results, query)
        
        return search_results
    
    async def _search_collection(self, 
                               adapter: VectorDBAdapter,
                               collection: Collection,
                               query: str,
                               top_k: int,
                               threshold: float) -> CollectionSearchResult:
        """Search a single collection"""
        try:
            documents = await adapter.search(query, top_k, threshold)
            
            doc_dicts = []
            scores = []
            for doc in documents:
                doc_dicts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                scores.append(doc.metadata.get("score", 0.0))
            
            return CollectionSearchResult(
                collection_id=collection.id,
                collection_name=collection.name,
                documents=doc_dicts,
                scores=scores,
                metadata={
                    "provider": collection.provider,
                    "embedding_model": collection.embedding_model
                }
            )
            
        except Exception as e:
            logger.error(f"Search failed for collection {collection.name}: {e}")
            return CollectionSearchResult(
                collection_id=collection.id,
                collection_name=collection.name,
                documents=[],
                scores=[],
                metadata={"error": str(e)}
            )
    
    def _rerank_results(self, results: List[CollectionSearchResult], query: str) -> List[CollectionSearchResult]:
        """Rerank results across collections (placeholder for more sophisticated reranking)"""
        # Simple reranking by average score per collection
        for result in results:
            if result.scores:
                result.metadata["avg_score"] = sum(result.scores) / len(result.scores)
            else:
                result.metadata["avg_score"] = 0.0
        
        # Sort by average score
        results.sort(key=lambda x: x.metadata.get("avg_score", 0.0), reverse=True)
        return results

class FutureReadyAgent:
    """Main agent with advanced collection management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_providers = self._initialize_models()
        self.collection_manager = CollectionManager(config.get("embedding_models", {}))
        self.session_memories = {}
        
    def _initialize_models(self):
        """Initialize model providers"""
        providers = {}
        for provider in ModelProvider:
            try:
                api_key = self.config.get(f"{provider.value}_api_key")
                if api_key:
                    if provider == ModelProvider.CLAUDE:
                        providers[provider] = ChatAnthropic(
                            model="claude-3-sonnet-20240229",
                            anthropic_api_key=api_key
                        )
                    elif provider == ModelProvider.OPENAI:
                        providers[provider] = ChatOpenAI(
                            model="gpt-4-turbo-preview",
                            openai_api_key=api_key
                        )
                    # Add other providers...
                    logger.info(f"Initialized {provider.value} model provider")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider.value}: {e}")
        return providers
    
    def _get_session_memory(self, session_id: str):
        """Get conversation memory for session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=10, memory_key="chat_history", return_messages=True
            )
        return self.session_memories[session_id]
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process chat with flexible collection selection"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get model
            model = self.model_providers.get(request.model_provider)
            if not model:
                raise ValueError(f"Model provider {request.model_provider} not available")
            
            # Resolve collections to search
            collections_to_search = self.collection_manager.resolve_collection_selection(
                mode=request.collection_selection_mode,
                selected_collections=request.selected_collections,
                filters=request.collection_filters
            )
            
            logger.info(f"Searching {len(collections_to_search)} collections: {[c.name for c in collections_to_search]}")
            
            # Search across selected collections
            search_results = []
            total_documents = 0
            
            if collections_to_search:
                search_results = await self.collection_manager.search_across_collections(
                    query=request.message,
                    collections=collections_to_search,
                    top_k_per_collection=request.top_k_per_collection,
                    threshold=request.similarity_threshold,
                    rerank=request.rerank_results
                )
                total_documents = sum(len(result.documents) for result in search_results)
            
            # Build context from search results
            context_parts = []
            for result in search_results:
                if result.documents:
                    context_parts.append(f"\n--- {result.collection_name} ---")
                    for doc in result.documents[:3]:  # Top 3 per collection
                        context_parts.append(f"Content: {doc['content'][:500]}...")
            
            context = "\n".join(context_parts) if context_parts else ""
            
            # Build prompt
            system_prompt = """You are a helpful AI assistant with access to multiple knowledge bases.
            Use the provided context from the knowledge bases to answer questions accurately.
            If you don't find relevant information, say so clearly."""
            
            if context:
                system_prompt += f"\n\nRelevant context from {len(search_results)} knowledge bases:\n{context}"
            
            # Get conversation memory
            memory = self._get_session_memory(request.session_id)
            
            # Create messages
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(memory.chat_memory.messages)
            messages.append(HumanMessage(content=request.message))
            
            # Generate response
            result = await model.ainvoke(messages)
            response_content = result.content
            
            # Update memory
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(response_content)
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return ChatResponse(
                message=response_content,
                model_used=request.model_provider.value,
                tokens_used=len(response_content.split()),  # Simplified
                response_time=response_time,
                collections_searched=[c.name for c in collections_to_search],
                search_results=search_results,
                total_documents_found=total_documents
            )
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global instances
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    
    # Startup configuration
    config = {
        "claude_api_key": "your-claude-key",
        "openai_api_key": "your-openai-key",
        "embedding_models": {
            "openai": {"type": "openai", "api_key": "your-openai-key"},
            "huggingface": {"type": "huggingface", "model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        }
    }
    
    agent_instance = FutureReadyAgent(config)
    logger.info("Agent with collection management initialized")
    
    yield
    logger.info("Shutting down agent")

# FastAPI app
app = FastAPI(
    title="Multi-Collection Vector Agent",
    description="Agent with flexible collection selection across multiple vector databases",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Collection Management Endpoints
@app.post("/collections", response_model=Collection)
async def create_collection(collection_data: CollectionCreateRequest):
    """Create a new collection"""
    return await agent_instance.collection_manager.register_collection(collection_data)

@app.get("/collections", response_model=List[Collection])
async def list_collections(
    provider: Optional[List[VectorDBProvider]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    active_only: bool = True
):
    """List collections with optional filters"""
    filters = {}
    if provider:
        filters["provider"] = provider
    if tags:
        filters["tags"] = tags
    
    collections = agent_instance.collection_manager.get_collections_by_filters(filters)
    
    if active_only:
        collections = [c for c in collections if c.is_active]
    
    return collections

@app.get("/collections/{collection_id}", response_model=Collection)
async def get_collection(collection_id: str):
    """Get specific collection details"""
    if collection_id not in agent_instance.collection_manager.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    return agent_instance.collection_manager.collections[collection_id]

@app.put("/collections/{collection_id}/status")
async def toggle_collection_status(collection_id: str, active: bool):
    """Enable/disable a collection"""
    if collection_id not in agent_instance.collection_manager.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    agent_instance.collection_manager.collections[collection_id].is_active = active
    return {"message": f"Collection {'activated' if active else 'deactivated'}"}

@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a collection"""
    if collection_id not in agent_instance.collection_manager.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    del agent_instance.collection_manager.collections[collection_id]
    if collection_id in agent_instance.collection_manager.adapters:
        del agent_instance.collection_manager.adapters[collection_id]
    
    return {"message": "Collection deleted"}

# Search and Test Endpoints
@app.post("/collections/search")
async def search_collections(
    query: str,
    collection_selection_mode: CollectionSelectionMode = CollectionSelectionMode.ALL,
    selected_collections: Optional[List[str]] = None,
    collection_filters: Optional[Dict[str, Any]] = None,
    top_k_per_collection: int = 5,
    similarity_threshold: float = 0.7
):
    """Search across selected collections"""
    collections = agent_instance.collection_manager.resolve_collection_selection(
        mode=collection_selection_mode,
        selected_collections=selected_collections,
        filters=collection_filters
    )
    
    results = await agent_instance.collection_manager.search_across_collections(
        query=query,
        collections=collections,
        top_k_per_collection=top_k_per_collection,
        threshold=similarity_threshold
    )
    
    return {
        "query": query,
        "collections_searched": [c.name for c in collections],
        "results": results,
        "total_documents": sum(len(r.documents) for r in results)
    }

@app.post("/collections/{collection_id}/test")
async def test_collection(collection_id: str):
    """Test collection connectivity and health"""
    if collection_id not in agent_instance.collection_manager.collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = agent_instance.collection_manager.collections[collection_id]
    adapter = agent_instance.collection_manager.adapters.get(collection_id)
    
    if not adapter:
        return {"status": "error", "message": "Adapter not initialized"}
    
    health = await adapter.health_check()
    info = await adapter.get_collection_info()
    
    return {
        "collection_name": collection.name,
        "provider": collection.provider,
        "health_check": "passed" if health else "failed",
        "info": info
    }

# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with flexible collection selection"""
    return await agent_instance.process_chat(request)

# Utility Endpoints
@app.get("/providers")
async def list_providers():
    """List all supported providers and their capabilities"""
    return {
        "vector_db_providers": {
            "qdrant": {
                "name": "Qdrant",
                "supports": ["hybrid_search", "filtering", "geo_search"],
                "deployment": ["cloud", "self_hosted"]
            },
            "pinecone": {
                "name": "Pinecone",
                "supports": ["metadata_filtering", "namespaces"],
                "deployment": ["cloud"]
            },
            "aws_opensearch": {
                "name": "AWS OpenSearch",
                "supports": ["knn_search", "lexical_search", "hybrid"],
                "deployment": ["aws_managed"]
            },
            "weaviate": {
                "name": "Weaviate",
                "supports": ["multi_modal", "generative_search", "classification"],
                "deployment": ["cloud", "self_hosted"]
            },
            "chroma": {
                "name": "ChromaDB",
                "supports": ["local_storage", "metadata_filtering"],
                "deployment": ["self_hosted", "embedded"]
            }
        },
        "model_providers": {
            provider.value: {
                "available": provider in agent_instance.model_providers,
                "features": ["chat", "streaming"] if provider in agent_instance.model_providers else []
            }
            for provider in ModelProvider
        }
    }

@app.get("/collections/stats")
async def get_collection_stats():
    """Get statistics about all collections"""
    collections = list(agent_instance.collection_manager.collections.values())
    
    stats = {
        "total_collections": len(collections),
        "active_collections": len([c for c in collections if c.is_active]),
        "by_provider": {},
        "total_documents": 0,
        "by_embedding_model": {}
    }
    
    for collection in collections:
        # Count by provider
        provider = collection.provider.value
        if provider not in stats["by_provider"]:
            stats["by_provider"][provider] = {"count": 0, "active": 0, "documents": 0}
        
        stats["by_provider"][provider]["count"] += 1
        if collection.is_active:
            stats["by_provider"][provider]["active"] += 1
        stats["by_provider"][provider]["documents"] += collection.document_count
        
        # Count by embedding model
        embedding = collection.embedding_model
        if embedding not in stats["by_embedding_model"]:
            stats["by_embedding_model"][embedding] = {"count": 0, "documents": 0}
        
        stats["by_embedding_model"][embedding]["count"] += 1
        stats["by_embedding_model"][embedding]["documents"] += collection.document_count
        
        stats["total_documents"] += collection.document_count
    
    return stats

@app.post("/collections/bulk-test")
async def bulk_test_collections(collection_ids: Optional[List[str]] = None):
    """Test multiple collections at once"""
    if collection_ids is None:
        collection_ids = list(agent_instance.collection_manager.collections.keys())
    
    test_results = {}
    
    for collection_id in collection_ids:
        if collection_id not in agent_instance.collection_manager.collections:
            test_results[collection_id] = {"status": "not_found"}
            continue
        
        collection = agent_instance.collection_manager.collections[collection_id]
        adapter = agent_instance.collection_manager.adapters.get(collection_id)
        
        if not adapter:
            test_results[collection_id] = {
                "status": "error",
                "message": "Adapter not initialized",
                "collection_name": collection.name
            }
            continue
        
        try:
            health = await adapter.health_check()
            info = await adapter.get_collection_info()
            
            test_results[collection_id] = {
                "status": "success" if health else "failed",
                "collection_name": collection.name,
                "provider": collection.provider.value,
                "health_check": health,
                "info": info
            }
        except Exception as e:
            test_results[collection_id] = {
                "status": "error",
                "collection_name": collection.name,
                "error": str(e)
            }
    
    return test_results

@app.get("/health")
async def health_check():
    """Overall system health check"""
    collections = agent_instance.collection_manager.collections
    active_collections = [c for c in collections.values() if c.is_active]
    
    return {
        "status": "healthy",
        "models_available": len(agent_instance.model_providers),
        "total_collections": len(collections),
        "active_collections": len(active_collections),
        "providers_supported": len(VectorDBProvider),
        "embedding_models": len(agent_instance.collection_manager.embedding_models)
    }

# WebSocket for real-time chat with collection selection
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Create chat request
            request = ChatRequest(
                message=message_data["message"],
                session_id=session_id,
                model_provider=ModelProvider(message_data.get("model_provider", "claude")),
                collection_selection_mode=CollectionSelectionMode(
                    message_data.get("collection_selection_mode", "all")
                ),
                selected_collections=message_data.get("selected_collections"),
                collection_filters=message_data.get("collection_filters"),
                top_k_per_collection=message_data.get("top_k_per_collection", 3),
                similarity_threshold=message_data.get("similarity_threshold", 0.7),
                stream=True
            )
            
            # Send typing indicator
            await manager.send_message(session_id, {
                "type": "typing",
                "message": "Searching collections and generating response..."
            })
            
            # Process chat
            try:
                response = await agent_instance.process_chat(request)
                
                # Send complete response
                await manager.send_message(session_id, {
                    "type": "response",
                    "message": response.message,
                    "model_used": response.model_used,
                    "collections_searched": response.collections_searched,
                    "total_documents_found": response.total_documents_found,
                    "response_time": response.response_time,
                    "search_results": [
                        {
                            "collection_name": result.collection_name,
                            "document_count": len(result.documents),
                            "avg_score": sum(result.scores) / len(result.scores) if result.scores else 0
                        }
                        for result in response.search_results
                    ]
                })
                
            except Exception as e:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)