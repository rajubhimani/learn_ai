# Collection Configuration and Usage Examples

import requests
import asyncio
import json

# Base API URL
BASE_URL = "http://localhost:8000"

# ==========================================
# COLLECTION CONFIGURATION EXAMPLES
# ==========================================

# 1. Qdrant Collection (Self-hosted or Cloud)
qdrant_collection = {
    "name": "Company Knowledge Base",
    "description": "Internal company documentation and policies",
    "provider": "qdrant",
    "connection_config": {
        "url": "https://your-cluster.qdrant.io",  # or "localhost" for self-hosted
        "port": 6333,
        "api_key": "your-qdrant-api-key",
        "https": True,
        "collection_name": "company_docs"
    },
    "embedding_model": "openai",
    "tags": ["internal", "documentation", "policies"],
    "metadata": {
        "department": "all",
        "access_level": "internal",
        "last_updated": "2024-01-15"
    }
}

# 2. AWS OpenSearch Collection
aws_opensearch_collection = {
    "name": "Product Catalog",
    "description": "E-commerce product information and specifications",
    "provider": "aws_opensearch",
    "connection_config": {
        "host": "search-products-xxxxx.us-west-2.es.amazonaws.com",
        "port": 443,
        "username": "your-username",
        "password": "your-password",
        "index_name": "product_catalog"
    },
    "embedding_model": "openai",
    "tags": ["products", "ecommerce", "public"],
    "metadata": {
        "region": "us-west-2",
        "environment": "production"
    }
}

# 3. Pinecone Collection
pinecone_collection = {
    "name": "Customer Support KB",
    "description": "Customer support articles and FAQs",
    "provider": "pinecone",
    "connection_config": {
        "api_key": "your-pinecone-api-key",
        "environment": "us-west1-gcp",
        "index_name": "support-kb"
    },
    "embedding_model": "openai",
    "tags": ["support", "faq", "customer"],
    "metadata": {
        "language": "en",
        "support_tier": "general"
    }
}

# 4. Multiple Collections for Different Departments
collections_config = [
    {
        "name": "Engineering Docs",
        "description": "Technical documentation and API references",
        "provider": "qdrant",
        "connection_config": {
            "url": "localhost",
            "port": 6333,
            "collection_name": "engineering_docs"
        },
        "embedding_model": "openai",
        "tags": ["engineering", "technical", "api"],
        "metadata": {"department": "engineering", "access_level": "team"}
    },
    {
        "name": "Marketing Materials",
        "description": "Marketing content, case studies, and presentations",
        "provider": "aws_opensearch",
        "connection_config": {
            "host": "search-marketing.amazonaws.com",
            "port": 443,
            "username": "marketing-user",
            "password": "marketing-pass",
            "index_name": "marketing_content"
        },
        "embedding_model": "huggingface",
        "tags": ["marketing", "content", "public"],
        "metadata": {"department": "marketing", "access_level": "public"}
    },
    {
        "name": "Legal Documents",
        "description": "Contracts, policies, and legal documentation",
        "provider": "pinecone",
        "connection_config": {
            "api_key": "legal-pinecone-key",
            "environment": "us-east1-gcp",
            "index_name": "legal-docs"
        },
        "embedding_model": "openai",
        "tags": ["legal", "contracts", "confidential"],
        "metadata": {"department": "legal", "access_level": "restricted"}
    }
]

# ==========================================
# SETUP FUNCTIONS
# ==========================================

async def setup_collections():
    """Setup all collections"""
    created_collections = []
    
    for config in collections_config:
        response = requests.post(f"{BASE_URL}/collections", json=config)
        if response.status_code == 200:
            collection = response.json()
            created_collections.append(collection)
            print(f"✅ Created collection: {collection['name']} ({collection['id']})")
        else:
            print(f"❌ Failed to create collection {config['name']}: {response.text}")
    
    return created_collections

def test_all_collections():
    """Test all collections"""
    response = requests.post(f"{BASE_URL}/collections/bulk-test")
    if response.status_code == 200:
        results = response.json()
        for collection_id, result in results.items():
            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} {result.get('collection_name', collection_id)}: {result['status']}")
    else:
        print(f"Failed to test collections: {response.text}")

# ==========================================
# USAGE EXAMPLES
# ==========================================

# Example 1: Search ALL collections
def search_all_collections(query):
    """Search across all available collections"""
    chat_request = {
        "message": query,
        "session_id": "user_123",
        "model_provider": "claude",
        "collection_selection_mode": "all",  # Search all collections
        "top_k_per_collection": 3,
        "similarity_threshold": 0.7,
        "rerank_results": True
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['message']}")
        print(f"Collections searched: {result['collections_searched']}")
        print(f"Total documents found: {result['total_documents_found']}")
        return result
    else:
        print(f"Error: {response.text}")

# Example 2: Search SPECIFIC collections
def search_specific_collections(query, collection_ids):
    """Search only specified collections"""
    chat_request = {
        "message": query,
        "session_id": "user_123",
        "model_provider": "openai",
        "collection_selection_mode": "multiple",  # Search specific collections
        "selected_collections": collection_ids,   # List of collection IDs
        "top_k_per_collection": 5,
        "similarity_threshold": 0.6
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    return response.json() if response.status_code == 200 else None

# Example 3: Search with FILTERS
def search_by_department(query, department):
    """Search collections belonging to a specific department"""
    chat_request = {
        "message": query,
        "session_id": "user_123",
        "model_provider": "claude",
        "collection_selection_mode": "all",
        "collection_filters": {
            "metadata": {"department": department}  # Filter by department
        },
        "top_k_per_collection": 4
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    return response.json() if response.status_code == 200 else None

# Example 4: Search by TAGS
def search_by_tags(query, tags):
    """Search collections with specific tags"""
    chat_request = {
        "message": query,
        "session_id": "user_123",
        "model_provider": "claude",
        "collection_selection_mode": "all",
        "collection_filters": {
            "tags": tags  # Must have at least one of these tags
        }
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    return response.json() if response.status_code == 200 else None

# Example 5: Search SINGLE collection
def search_single_collection(query, collection_id):
    """Search only one specific collection"""
    chat_request = {
        "message": query,
        "session_id": "user_123",
        "model_provider": "claude",
        "collection_selection_mode": "single",
        "selected_collections": [collection_id],  # Single collection
        "top_k_per_collection": 10  # Can get more results from single collection
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    return response.json() if response.status_code == 200 else None

# ==========================================
# MANAGEMENT FUNCTIONS
# ==========================================

def list_collections_with_filters():
    """List collections with various filters"""
    
    # Get all collections
    response = requests.get(f"{BASE_URL}/collections")
    all_collections = response.json() if response.status_code == 200 else []
    
    # Get collections by provider
    response = requests.get(f"{BASE_URL}/collections?provider=qdrant&provider=pinecone")
    filtered_collections = response.json() if response.status_code == 200 else []
    
    # Get collections by tags
    response = requests.get(f"{BASE_URL}/collections?tags=engineering&tags=technical")
    tagged_collections = response.json() if response.status_code == 200 else []
    
    print(f"All collections: {len(all_collections)}")
    print(f"Qdrant/Pinecone collections: {len(filtered_collections)}")
    print(f"Engineering collections: {len(tagged_collections)}")
    
    return {
        "all": all_collections,
        "filtered": filtered_collections,
        "tagged": tagged_collections
    }

def get_collection_stats():
    """Get system-wide collection statistics"""
    response = requests.get(f"{BASE_URL}/collections/stats")
    if response.status_code == 200:
        stats = response.json()
        print("=== Collection Statistics ===")
        print(f"Total collections: {stats['total_collections']}")
        print(f"Active collections: {stats['active_collections']}")
        print(f"Total documents: {stats['total_documents']}")
        
        print("\n--- By Provider ---")
        for provider, data in stats['by_provider'].items():
            print(f"{provider}: {data['active']}/{data['count']} active, {data['documents']} docs")
        
        print("\n--- By Embedding Model ---")
        for model, data in stats['by_embedding_model'].items():
            print(f"{model}: {data['count']} collections, {data['documents']} docs")
        
        return stats

def disable_enable_collection(collection_id, active=True):
    """Enable or disable a collection"""
    response = requests.put(f"{BASE_URL}/collections/{collection_id}/status", 
                          params={"active": active})
    if response.status_code == 200:
        status = "enabled" if active else "disabled"
        print(f"Collection {collection_id} {status}")
    else:
        print(f"Failed to update collection: {response.text}")

# ==========================================
# ADVANCED SEARCH EXAMPLES
# ==========================================

def contextual_search_example():
    """Example of contextual search across different collection types"""
    
    # Technical question - search engineering collections
    tech_response = search_by_tags(
        "How do I implement OAuth2 authentication?", 
        ["engineering", "technical", "api"]
    )
    
    # Business question - search marketing/business collections
    business_response = search_by_tags(
        "What are our key competitive advantages?",
        ["marketing", "business", "strategy"]
    )
    
    # Legal question - search legal collections
    legal_response = search_by_department(
        "What is our data retention policy?",
        "legal"
    )
    
    return {
        "technical": tech_response,
        "business": business_response,
        "legal": legal_response
    }

def multi_provider_search(query):
    """Search across multiple vector database providers"""
    
    # Get collections by provider
    providers_to_search = ["qdrant", "pinecone", "aws_opensearch"]
    
    chat_request = {
        "message": query,
        "session_id": "multi_provider_search",
        "model_provider": "claude",
        "collection_selection_mode": "all",
        "collection_filters": {
            "provider": providers_to_search
        },
        "top_k_per_collection": 2,  # Lower per collection since we're searching many
        "rerank_results": True      # Important for multi-provider results
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_request)
    if response.status_code == 200:
        result = response.json()
        
        # Analyze results by provider
        provider_results = {}
        for search_result in result['search_results']:
            provider = search_result['metadata'].get('provider', 'unknown')
            if provider not in provider_results:
                provider_results[provider] = []
            provider_results[provider].append(search_result)
        
        print(f"Query: {query}")
        print(f"Total collections searched: {len(result['collections_searched'])}")
        
        for provider, results in provider_results.items():
            doc_count = sum(len(r['documents']) for r in results)
            print(f"{provider}: {len(results)} collections, {doc_count} documents")
        
        return result

# ==========================================
# WEBHOOK AND REAL-TIME EXAMPLES
# ==========================================

import websocket
import threading

def websocket_chat_example():
    """Example of real-time chat with collection selection"""
    
    def on_message(ws, message):
        data = json.loads(message)
        if data["type"] == "response":
            print(f"\n🤖 {data['message']}")
            print(f"📚 Searched: {', '.join(data['collections_searched'])}")
            print(f"📄 Found {data['total_documents_found']} documents")
            
            if data.get('search_results'):
                print("\n--- Collection Results ---")
                for result in data['search_results']:
                    print(f"• {result['collection_name']}: {result['document_count']} docs")
        
        elif data["type"] == "typing":
            print(f"⏳ {data['message']}")
        
        elif data["type"] == "error":
            print(f"❌ Error: {data['message']}")
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def send_message(ws, message, collections=None, mode="all"):
        data = {
            "message": message,
            "model_provider": "claude",
            "collection_selection_mode": mode,
            "selected_collections": collections,
            "top_k_per_collection": 3
        }
        ws.send(json.dumps(data))
    
    # Connect to WebSocket
    ws_url = "ws://localhost:8000/ws/realtime_user"
    ws = websocket.WebSocketApp(ws_url,
                              on_message=on_message,
                              on_error=on_error)
    
    # Start WebSocket in background
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    # Example interactions
    import time
    time.sleep(1)  # Wait for connection
    
    # Search all collections
    send_message(ws, "What is our company mission?", mode="all")
    time.sleep(3)
    
    # Search specific collections
    send_message(ws, "How do I deploy the API?", 
                collections=["engineering_collection_id"], mode="single")
    time.sleep(3)
    
    ws.close()

# ==========================================
# MAIN EXECUTION EXAMPLES
# ==========================================

async def main():
    """Main function demonstrating the collection system"""
    
    print("🚀 Setting up Multi-Collection Vector Agent System")
    
    # 1. Setup collections
    print("\n1. Setting up collections...")
    collections = await setup_collections()
    
    # 2. Test all collections
    print("\n2. Testing collections...")
    test_all_collections()
    
    # 3. Get system statistics
    print("\n3. System statistics...")
    get_collection_stats()
    
    # 4. Example searches
    print("\n4. Example searches...")
    
    # Search all collections
    print("\n--- Search All Collections ---")
    search_all_collections("What are the latest API changes?")
    
    # Search by department
    print("\n--- Search Engineering Collections ---") 
    search_by_department("How to handle authentication?", "engineering")
    
    # Search by tags
    print("\n--- Search Public Collections ---")
    search_by_tags("Product pricing information", ["public", "products"])
    
    # Multi-provider search
    print("\n--- Multi-Provider Search ---")
    multi_provider_search("Customer support best practices")
    
    # 5. WebSocket example (commented out for demo)
    # print("\n5. WebSocket chat example...")
    # websocket_chat_example()
    
    print("\n✅ Multi-Collection Vector Agent System Demo Complete!")

if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())

# ==========================================
# CONFIGURATION FOR DIFFERENT SCENARIOS
# ==========================================

# Scenario 1: Multi-tenant SaaS
SAAS_CONFIG = {
    "collections_per_tenant": [
        {"tenant": "company_a", "collections": ["docs_a", "support_a", "products_a"]},
        {"tenant": "company_b", "collections": ["docs_b", "support_b", "products_b"]}
    ],
    "shared_collections": ["public_kb", "general_help"]
}

# Scenario 2: Different data sources
DATA_SOURCES_CONFIG = {
    "internal_docs": {"provider": "qdrant", "access": "internal"},
    "public_kb": {"provider": "pinecone", "access": "public"},
    "customer_data": {"provider": "aws_opensearch", "access": "restricted"},
    "real_time_data": {"provider": "weaviate", "access": "dynamic"}
}

# Scenario 3: Geographic distribution
GEO_CONFIG = {
    "us_west": {"provider": "pinecone", "region": "us-west1"},
    "us_east": {"provider": "aws_opensearch", "region": "us-east-1"},
    "europe": {"provider": "qdrant", "region": "eu-central"},
    "asia": {"provider": "weaviate", "region": "asia-southeast1"}
}