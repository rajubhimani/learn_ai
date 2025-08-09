# Vector Database Adapter System Usage Guide

## Overview

This adapter system provides a unified interface for working with multiple vector databases, making it easy to switch between different providers or use multiple databases simultaneously.

## Supported Databases

- **ChromaDB**: Local/persistent vector database
- **Pinecone**: Managed cloud vector database
- **Weaviate**: Open-source vector database
- **Qdrant**: High-performance vector database (adapter structure ready)
- **Elasticsearch**: Search engine with vector capabilities (adapter structure ready)
- **PostgreSQL + pgvector**: Traditional database with vector extension (adapter structure ready)

## Installation Requirements

```bash
# Core requirements
pip install sentence-transformers numpy

# Database-specific requirements
pip install chromadb              # For ChromaDB
pip install pinecone-client       # For Pinecone
pip install weaviate-client       # For Weaviate
pip install qdrant-client         # For Qdrant
pip install elasticsearch         # For Elasticsearch
pip install psycopg2-binary pgvector  # For PostgreSQL
```

## Configuration Examples

### ChromaDB Configuration
```python
chroma_config = {
    'persistent': True,           # Use persistent storage
    'path': './chroma_db',       # Storage path
    'auto_embed': True,          # Auto-generate embeddings
    'embedding_model': 'all-MiniLM-L6-v2'  # Embedding model
}
```

### Pinecone Configuration
```python
pinecone_config = {
    'api_key': 'your-pinecone-api-key',    # Or set PINECONE_API_KEY env var
    'environment': 'us-east-1-aws',        # Pinecone environment
    'auto_embed': True,
    'embedding_model': 'all-MiniLM-L6-v2'
}
```

### Weaviate Configuration
```python
weaviate_config = {
    'url': 'http://localhost:8080',        # Weaviate URL
    'api_key': 'optional-api-key',         # For Weaviate Cloud
    'auto_embed': True,
    'embedding_model': 'all-MiniLM-L6-v2'
}
```

## Basic Usage

### Single Database Setup

```python
from vector_db_adapter import VectorDatabaseFactory, VectorDBType, Document

# Create adapter
config = {
    'persistent': True,
    'path': './my_db',
    'auto_embed': True
}
adapter = VectorDatabaseFactory.create_adapter(VectorDBType.CHROMADB, config)

# Create collection
adapter.create_collection('my_collection', dimension=384)

# Add documents
documents = [
    Document(
        id='doc1',
        content='Your document content here',
        metadata={'category': 'example', 'timestamp': '2024-01-01'}
    )
]
adapter.add_documents('my_collection', documents)

# Search
results = adapter.search('my_collection', 'your search query', limit=5)
```

### Multi-Database Setup

```python
from vector_db_adapter import MultiVectorDBManager, VectorDBType

# Initialize manager
manager = MultiVectorDBManager()

# Add multiple adapters
manager.add_adapter('local', VectorDBType.CHROMADB, {
    'persistent': True,
    'path': './local_db'
}, set_as_default=True)

manager.add_adapter('cloud', VectorDBType.PINECONE, {
    'api_key': 'your-api-key',
    'environment': 'us-east-1-aws'
})

# Use specific adapter
local_adapter = manager.get_adapter('local')
cloud_adapter = manager.get_adapter('cloud')

# Or use default adapter
default_adapter = manager.get_adapter()

# Search across all adapters
all_results = manager.search_all('collection_name', 'query', limit=10)
```

## Integration with Document Processing Pipeline

```python
from document_collection_pipeline import DocumentCollectionPipeline
from vector_db_adapter import MultiVectorDBManager, VectorDBType

class EnhancedDocumentPipeline(DocumentCollectionPipeline):
    def __init__(self, db_manager: MultiVectorDBManager, collection_name: str = "documents"):
        self.processor = FileProcessor()
        self.chunker = DocumentChunker()
        self.db_manager = db_manager
        self.collection_name = collection_name
        
        # Create collection in all adapters
        for adapter_name in db_manager.list_adapters():
            adapter = db_manager.get_adapter(adapter_name)
            adapter.create_collection(collection_name, dimension=384)
    
    def process_files_multi_db(self, file_paths: List[str], 
                              target_adapters: Optional[List[str]] = None) -> Dict:
        """Process files and store in multiple databases"""
        if target_adapters is None:
            target_adapters = self.db_manager.list_adapters()
        
        total_docs = 0
        results = {}
        
        for file_path in file_paths:
            try:
                # Parse and chunk documents
                documents = self.processor.process_file(file_path)
                all_chunks = []
                
                for doc in documents:
                    chunks = self.chunker.chunk_document(doc)
                    all_chunks.extend(chunks)
                
                # Convert to adapter format
                adapter_docs = []
                for chunk in all_chunks:
                    adapter_doc = Document(
                        id=chunk.chunk_id or chunk.doc_id,
                        content=chunk.content,
                        metadata=chunk.metadata
                    )
                    adapter_docs.append(adapter_doc)
                
                # Store in selected adapters
                for adapter_name in target_adapters:
                    adapter = self.db_manager.get_adapter(adapter_name)
                    success = adapter.add_documents(self.collection_name, adapter_docs)
                    
                    if adapter_name not in results:
                        results[adapter_name] = {'success': 0, 'failed': 0}
                    
                    if success:
                        results[adapter_name]['success'] += len(adapter_docs)
                    else:
                        results[adapter_name]['failed'] += len(adapter_docs)
                
                total_docs += len(adapter_docs)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return {
            'total_documents_processed': total_docs,
            'adapter_results': results
        }
    
    def search_with_fallback(self, query: str, limit: int = 10, 
                           preferred_adapter: Optional[str] = None) -> List[SearchResult]:
        """Search with fallback to other adapters if primary fails"""
        adapters_to_try = []
        
        if preferred_adapter:
            adapters_to_try.append(preferred_adapter)
        
        # Add remaining adapters
        for name in self.db_manager.list_adapters():
            if name not in adapters_to_try:
                adapters_to_try.append(name)
        
        for adapter_name in adapters_to_try:
            try:
                adapter = self.db_manager.get_adapter(adapter_name)
                results = adapter.search(self.collection_name, query, limit)
                if results:
                    logger.info(f"Search successful using {adapter_name}")
                    return results
            except Exception as e:
                logger.warning(f"Search failed on {adapter_name}: {e}")
        
        return []
```

## Advanced Configuration

### Environment-Based Configuration

```python
import os
from typing import Dict, Any

def load_db_config() -> Dict[str, Dict[str, Any]]:
    """Load database configurations from environment variables"""
    configs = {}
    
    # ChromaDB
    if os.getenv('USE_CHROMADB', 'true').lower() == 'true':
        configs['chroma'] = {
            'type': VectorDBType.CHROMADB,
            'config': {
                'persistent': True,
                'path': os.getenv('CHROMADB_PATH', './chroma_db'),
                'auto_embed': True,
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            }
        }
    
    # Pinecone
    if os.getenv('PINECONE_API_KEY'):
        configs['pinecone'] = {
            'type': VectorDBType.PINECONE,
            'config': {
                'api_key': os.getenv('PINECONE_API_KEY'),
                'environment': os.getenv('PINECONE_ENV', 'us-east-1-aws'),
                'auto_embed': True,
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            }
        }
    
    # Weaviate
    if os.getenv('WEAVIATE_URL'):
        configs['weaviate'] = {
            'type': VectorDBType.WEAVIATE,
            'config': {
                'url': os.getenv('WEAVIATE_URL'),
                'api_key': os.getenv('WEAVIATE_API_KEY'),
                'auto_embed': True,
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            }
        }
    
    return configs

# Usage
def setup_multi_db_manager() -> MultiVectorDBManager:
    manager = MultiVectorDBManager()
    configs = load_db_config()
    
    for name, config_data in configs.items():
        manager.add_adapter(
            name=name,
            db_type=config_data['type'],
            config=config_data['config'],
            set_as_default=(name == 'chroma')  # Set ChromaDB as default
        )
    
    return manager
```

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer
import openai

class CustomEmbeddingAdapter(ChromaDBAdapter):
    """Custom adapter with OpenAI embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if config.get('use_openai_embeddings'):
            openai.api_key = config.get('openai_api_key')
            self.use_openai = True
        else:
            self.use_openai = False
    
    def _generate_embedding(self, text: str) -> List[float]:
        if self.use_openai:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        else:
            return super()._generate_embedding(text)

# Register custom adapter
VectorDatabaseFactory.register_adapter(
    VectorDBType.CHROMADB, 
    CustomEmbeddingAdapter
)
```

## Production Deployment Patterns

### Hybrid Setup (Local + Cloud)

```python
def setup_hybrid_deployment():
    """Setup with local for development, cloud for production"""
    manager = MultiVectorDBManager()
    
    # Local ChromaDB for development/caching
    manager.add_adapter('local', VectorDBType.CHROMADB, {
        'persistent': True,
        'path': './cache_db',
        'auto_embed': True
    })
    
    # Pinecone for production
    manager.add_adapter('production', VectorDBType.PINECONE, {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'environment': 'us-east-1-aws',
        'auto_embed': True
    }, set_as_default=True)
    
    return manager
```

### Load Balancing Across Multiple Instances

```python
import random

class LoadBalancedManager(MultiVectorDBManager):
    """Manager with load balancing across similar adapters"""
    
    def __init__(self):
        super().__init__()
        self.adapter_groups = {}
    
    def add_adapter_group(self, group_name: str, adapters: List[str]):
        """Group adapters for load balancing"""
        self.adapter_groups[group_name] = adapters
    
    def search_load_balanced(self, group_name: str, collection_name: str, 
                           query: str, limit: int = 10) -> List[SearchResult]:
        """Search using load balancing within a group"""
        if group_name not in self.adapter_groups:
            raise ValueError(f"Unknown adapter group: {group_name}")
        
        # Select random adapter from group
        adapter_name = random.choice(self.adapter_groups[group_name])
        adapter = self.get_adapter(adapter_name)
        
        return adapter.search(collection_name, query, limit)

# Example usage
manager = LoadBalancedManager()

# Add multiple Pinecone instances
for i in range(3):
    manager.add_adapter(f'pinecone_{i}', VectorDBType.PINECONE, {
        'api_key': os.getenv(f'PINECONE_API_KEY_{i}'),
        'environment': 'us-east-1-aws'
    })

# Group them for load balancing
manager.add_adapter_group('pinecone_cluster', 
                         ['pinecone_0', 'pinecone_1', 'pinecone_2'])
```

## Monitoring and Health Checks

```python
from datetime import datetime
import time

class MonitoredVectorDBManager(MultiVectorDBManager):
    """Enhanced manager with monitoring capabilities"""
    
    def __init__(self):
        super().__init__()
        self.health_status = {}
        self.performance_metrics = {}
    
    def health_check(self, adapter_name: str) -> Dict[str, Any]:
        """Perform health check on an adapter"""
        adapter = self.get_adapter(adapter_name)
        if not adapter:
            return {'status': 'not_found', 'timestamp': datetime.now()}
        
        try:
            start_time = time.time()
            
            # Try to list collections
            collections = adapter.list_collections()
            
            # If collections exist, try a simple operation
            if collections:
                info = adapter.get_collection_info(collections[0])
                response_time = time.time() - start_time
                
                status = {
                    'status': 'healthy',
                    'response_time': response_time,
                    'collections_count': len(collections),
                    'timestamp': datetime.now()
                }
            else:
                status = {
                    'status': 'healthy_no_collections',
                    'response_time': time.time() - start_time,
                    'collections_count': 0,
                    'timestamp': datetime.now()
                }
            
            self.health_status[adapter_name] = status
            return status
            
        except Exception as e:
            status = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now()
            }
            self.health_status[adapter_name] = status
            return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all adapters"""
        summary = {
            'total_adapters': len(self.adapters),
            'healthy': 0,
            'unhealthy': 0,
            'adapters': {}
        }
        
        for adapter_name in self.adapters:
            health = self.health_check(adapter_name)
            summary['adapters'][adapter_name] = health
            
            if health['status'] in ['healthy', 'healthy_no_collections']:
                summary['healthy'] += 1
            else:
                summary['unhealthy'] += 1
        
        return summary
```

## Error Handling and Retry Logic

```python
import time
from functools import wraps

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            
        return wrapper
    return decorator

class RobustVectorDBAdapter:
    """Wrapper for any adapter with retry logic"""
    
    def __init__(self, adapter: VectorDatabaseAdapter, max_retries: int = 3):
        self.adapter = adapter
        self.max_retries = max_retries
    
    @retry_on_failure(max_retries=3)
    def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        return self.adapter.add_documents(collection_name, documents)
    
    @retry_on_failure(max_retries=3)
    def search(self, collection_name: str, query: Union[str, List[float]], 
               limit: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        return self.adapter.search(collection_name, query, limit, filters)
    
    # Delegate other methods...
    def __getattr__(self, name):
        return getattr(self.adapter, name)
```

## Testing and Validation

```python
import unittest
from unittest.mock import Mock, patch

class TestVectorDBAdapter(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'persistent': False,  # Use in-memory for testing
            'auto_embed': True
        }
        self.adapter = VectorDatabaseFactory.create_adapter(
            VectorDBType.CHROMADB, 
            self.config
        )
        self.test_collection = 'test_collection'
        self.adapter.create_collection(self.test_collection, dimension=384)
    
    def test_add_and_search_documents(self):
        # Add test documents
        documents = [
            Document(
                id='test1',
                content='This is a test document about AI',
                metadata={'category': 'AI'}
            ),
            Document(
                id='test2',
                content='This document discusses machine learning',
                metadata={'category': 'ML'}
            )
        ]
        
        success = self.adapter.add_documents(self.test_collection, documents)
        self.assertTrue(success)
        
        # Search
        results = self.adapter.search(self.test_collection, 'artificial intelligence')
        self.assertGreater(len(results), 0)
        self.assertIn('test1', [r.id for r in results])
    
    def test_collection_management(self):
        # Test collection creation and deletion
        test_collection = 'temp_collection'
        
        success = self.adapter.create_collection(test_collection, dimension=384)
        self.assertTrue(success)
        
        collections = self.adapter.list_collections()
        self.assertIn(test_collection, collections)
        
        success = self.adapter.delete_collection(test_collection)
        self.assertTrue(success)
        
        collections = self.adapter.list_collections()
        self.assertNotIn(test_collection, collections)

if __name__ == '__main__':
    unittest.main()
```

## Best Practices

### 1. Configuration Management
- Use environment variables for sensitive data (API keys)
- Keep configuration files for different environments
- Validate configurations before creating adapters

### 2. Performance Optimization
- Use batch operations when possible
- Implement connection pooling for high-throughput scenarios
- Monitor and tune embedding model performance

### 3. Data Consistency
- Implement proper error handling and rollback mechanisms
- Use transactions where supported
- Regular health checks and data validation

### 4. Security
- Never store API keys in code
- Use proper authentication mechanisms
- Implement access controls and audit logging

### 5. Scaling Considerations
- Plan for horizontal scaling from the beginning
- Consider data partitioning strategies
- Implement proper monitoring and alerting