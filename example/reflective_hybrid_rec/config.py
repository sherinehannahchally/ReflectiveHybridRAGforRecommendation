from dataclasses import dataclass, field
import os

@dataclass
class LLMConfig:
    model_name: str = "gemini-2.5-flash"
    api_key: str = "AIzaSyB81pOOJBeCkdaD4dkPHVmbz8ycVhUdSA4"
    temperature: float = 0.2
    max_tokens: int = 512

@dataclass
class QdrantConfig:
    url: str = "https://49c82fba-ae7f-41df-b52c-5c6413219330.us-west-1-0.aws.cloud.qdrant.io"
    api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UBhFW8Du4eTVPUtWxqryuC047D4MYKSBfhGZKG62OTE"
    
    collection_name: str = "goodreads_youngadult_reviews_gemini" 
    reflection_collection: str = "user_insights_sidecar"           
    # collection_name: str = "goodreads_youngadult_reviews_gemini_001" 
    # reflection_collection: str = "user_insights_sidecar_001"           
    
    dense_model_name: str = "models/text-embedding-004"
    dense_dim: int = 768
    
    # dense_model_name: str = "models/text-embedding-001"
    # dense_dim: int = 3072
    
    sparse_model_id: str = "naver/efficient-splade-VI-BT-large-query"
    device: str = "cpu"

@dataclass
class RetrievalConfig:
    top_k_retrieval: int = 20
    top_k_final: int = 10

@dataclass
class ProjectConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)