from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import logging
from contextlib import asynccontextmanager
import redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import structlog
from prometheus_client import Counter, Histogram, generate_latest
import time

# Structured logging
logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter('agent_requests_total', 'Total agent requests', ['model', 'status'])
REQUEST_DURATION = Histogram('agent_request_duration_seconds', 'Request duration')

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    model_provider: str = Field(default="claude", regex="^(claude|openai|google|cohere)$")
    knowledge_base: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0, le=8000)

class AgentResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: int
    response_time: float
    sources: List[Dict[str, Any]] = []

class ModelProvider:
    """Abstract base for model providers"""
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class ClaudeProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Implement Claude API call
        return {"response": "Claude response", "tokens": 100}

class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Implement OpenAI API call
        return {"response": "OpenAI response", "tokens": 120}

class VectorDatabase:
    """Abstract vector database interface"""
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

class PineconeDB(VectorDatabase):
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Implement Pinecone search
        return [{"content": "relevant content", "score": 0.95}]

class ProductionAgent:
    def __init__(self, 
                 model_providers: Dict[str, ModelProvider],
                 vector_db: VectorDatabase,
                 cache_client: redis.Redis):
        self.model_providers = model_providers
        self.vector_db = vector_db
        self.cache = cache_client
        
    async def process_query(self, request: AgentRequest) -> AgentResponse:
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"query:{hash(request.query)}:{request.model_provider}"
            cached_response = await self._get_from_cache(cache_key)
            
            if cached_response:
                logger.info("Cache hit", query=request.query[:50])
                return cached_response
            
            # Retrieve relevant documents
            sources = []
            if request.knowledge_base:
                sources = await self.vector_db.search(request.query)
            
            # Build context-aware prompt
            prompt = self._build_prompt(request.query, sources)
            
            # Generate response
            provider = self.model_providers[request.model_provider]
            result = await provider.generate(
                prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_time = time.time() - start_time
            
            agent_response = AgentResponse(
                response=result["response"],
                model_used=request.model_provider,
                tokens_used=result["tokens"],
                response_time=response_time,
                sources=[{"content": s["content"], "score": s["score"]} for s in sources]
            )
            
            # Cache successful response
            await self._cache_response(cache_key, agent_response)
            
            # Record metrics
            REQUEST_COUNT.labels(model=request.model_provider, status="success").inc()
            REQUEST_DURATION.observe(response_time)
            
            logger.info("Query processed successfully", 
                       model=request.model_provider, 
                       response_time=response_time,
                       tokens=result["tokens"])
            
            return agent_response
            
        except Exception as e:
            REQUEST_COUNT.labels(model=request.model_provider, status="error").inc()
            logger.error("Query processing failed", error=str(e), query=request.query[:50])
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    def _build_prompt(self, query: str, sources: List[Dict]) -> str:
        if not sources:
            return f"Answer this query: {query}"
        
        context = "\n".join([s["content"] for s in sources[:3]])
        return f"""
        Context information:
        {context}
        
        Query: {query}
        
        Please answer the query using the provided context. If the context doesn't contain relevant information, say so.
        """
    
    async def _get_from_cache(self, key: str) -> Optional[AgentResponse]:
        try:
            cached = await self.cache.get(key)
            if cached:
                return AgentResponse.parse_raw(cached)
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
        return None
    
    async def _cache_response(self, key: str, response: AgentResponse, ttl: int = 3600):
        try:
            await self.cache.setex(key, ttl, response.json())
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting production agent service")
    yield
    # Shutdown
    logger.info("Shutting down production agent service")

# FastAPI app
app = FastAPI(
    title="Production Multi-Model Agent API",
    description="Enterprise-grade multi-model agent with knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["api.yourcompany.com", "localhost"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize components (in production, use dependency injection)
model_providers = {
    "claude": ClaudeProvider(api_key="your-claude-key"),
    "openai": OpenAIProvider(api_key="your-openai-key"),
}

vector_db = PineconeDB(api_key="your-pinecone-key", environment="production")
cache_client = redis.Redis(host="redis-cluster", port=6379, decode_responses=True)
agent = ProductionAgent(model_providers, vector_db, cache_client)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return generate_latest()

# Main agent endpoint
@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    return await agent.process_query(request)

# Batch processing endpoint
@app.post("/agent/batch", response_model=List[AgentResponse])
async def batch_query(requests: List[AgentRequest]):
    tasks = [agent.process_query(req) for req in requests]
    return await asyncio.gather(*tasks, return_exceptions=False)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )