import json
import uuid
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer # New Import

from .config import ProjectConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestLocal")
cfg = ProjectConfig()

def get_splade_vector(text, tokenizer, model, device):
    # Truncate to 512 tokens
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * tokens['attention_mask'].unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()
    
    indices = vec.nonzero().squeeze().cpu().tolist()
    values = vec[indices].cpu().tolist()
    
    if isinstance(indices, int):
        indices, values = [indices], [values]
        
    return models.SparseVector(indices=indices, values=values)

def main():
    logger.info(f"Loading Dense Model: {cfg.qdrant.dense_model_id}...")
    dense_model = SentenceTransformer(cfg.qdrant.dense_model_id, device=cfg.qdrant.device)

    logger.info(f"Loading Sparse Model: {cfg.qdrant.sparse_model_id}...")
    sparse_tokenizer = AutoTokenizer.from_pretrained(cfg.qdrant.sparse_model_id)
    sparse_model = AutoModelForMaskedLM.from_pretrained(cfg.qdrant.sparse_model_id).to(cfg.qdrant.device)

    logger.info("Connecting to Qdrant...")
    client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)

    # Always recreate collection when switching models to fix dimension mismatch
    if client.collection_exists(cfg.qdrant.collection_name):
        logger.warning(f"Deleting old collection: {cfg.qdrant.collection_name}")
        client.delete_collection(cfg.qdrant.collection_name)
    
    client.create_collection(
        collection_name=cfg.qdrant.collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=cfg.qdrant.dense_dim, # 384 for MiniLM
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    logger.info("Collection created.")

    data_path = Path("dataset/review.json")
    if not data_path.exists():
        logger.error("dataset/review.json not found.")
        return

    logger.info("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        reviews = [json.loads(line) for line in f]

    points = []
    batch_size = 64 # Can go higher with local models
    
    logger.info("Starting ingestion...")
    for i, review in enumerate(tqdm(reviews)):
        text_content = review.get('text') or review.get('content', '')
        if not text_content: continue

        try:
            # A. Local Dense Vector
            dense_vec = dense_model.encode(text_content).tolist()

            # B. Local Sparse Vector
            sparse_vec = get_splade_vector(text_content, sparse_tokenizer, sparse_model, cfg.qdrant.device)

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload={
                    "user_id": review.get('user_id'),
                    "item_id": review.get('item_id'),
                    "stars": review.get('stars'),
                    "text": text_content,
                    "timestamp": review.get('date')
                }
            ))

            if len(points) >= batch_size:
                client.upsert(cfg.qdrant.collection_name, points)
                points = []

        except Exception as e:
            logger.error(f"Error {i}: {e}")
            continue

    if points:
        client.upsert(cfg.qdrant.collection_name, points)

    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()