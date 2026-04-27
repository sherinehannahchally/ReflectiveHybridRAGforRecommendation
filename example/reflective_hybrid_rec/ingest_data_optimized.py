import json
import uuid
import logging
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForMaskedLM
import google.generativeai as genai

from .config import ProjectConfig

# --- CONFIG ---
BATCH_SIZE = 50 # Gemini rate limit friendly
DEVICE = "cpu"  # Local SPLADE on CPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestGemini")
cfg = ProjectConfig()

genai.configure(api_key=cfg.llm.api_key)

def get_splade_batch(texts, tokenizer, model, device):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * tokens['attention_mask'].unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    
    sparse_vectors = []
    for vec in max_val:
        indices = vec.nonzero().squeeze().cpu().tolist()
        values = vec[indices].cpu().tolist()
        if isinstance(indices, int): indices, values = [indices], [values]
        sparse_vectors.append(models.SparseVector(indices=indices, values=values))
    return sparse_vectors

def main():
    logger.info(f"🚀 Gemini Embedding Mode")
    
    logger.info("Loading Sparse Model...")
    sparse_tokenizer = AutoTokenizer.from_pretrained(cfg.qdrant.sparse_model_id)
    sparse_model = AutoModelForMaskedLM.from_pretrained(cfg.qdrant.sparse_model_id).to(DEVICE)
    sparse_model.eval()

    logger.info("Connecting to Qdrant...")
    client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)

    if client.collection_exists(cfg.qdrant.collection_name):
        logger.warning(f"Deleting existing collection: {cfg.qdrant.collection_name}")
        client.delete_collection(cfg.qdrant.collection_name)
    
    client.create_collection(
        collection_name=cfg.qdrant.collection_name,
        vectors_config={
            "gemini-dense": models.VectorParams(size=cfg.qdrant.dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "splade-sparse": models.SparseVectorParams()
        }
    )

    # data_path = Path("dataset/review.json")
    data_path = Path("dataset/review_small.json")  # <--- Update this
    if not data_path.exists():
        logger.error("dataset/review.json not found.")
        return

    logger.info("Reading dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_reviews = [json.loads(line) for line in f]
    
    logger.info(f"Starting ingestion of {len(all_reviews)} reviews...")

    for i in tqdm(range(0, len(all_reviews), BATCH_SIZE), desc="Ingesting"):
        batch_reviews = all_reviews[i : i + BATCH_SIZE]
        texts = [r.get('text') or r.get('content', '') for r in batch_reviews]
        valid_pairs = [(r, t) for r, t in zip(batch_reviews, texts) if t]
        if not valid_pairs: continue
        batch_reviews, texts = zip(*valid_pairs)

        try:
            # A. Gemini Dense
            dense_resp = genai.embed_content(
                model=cfg.qdrant.dense_model_id,
                content=texts,
                task_type="retrieval_document"
            )
            dense_vecs = dense_resp['embedding']

            # B. SPLADE Sparse
            sparse_vecs = get_splade_batch(texts, sparse_tokenizer, sparse_model, DEVICE)
            
            points = []
            for j, review in enumerate(batch_reviews):
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "gemini-dense": dense_vecs[j],
                        "splade-sparse": sparse_vecs[j]
                    },
                    payload={
                        "user_id": review.get('user_id'),
                        "item_id": review.get('item_id'),
                        "stars": review.get('stars'),
                        "text": texts[j],
                        "timestamp": review.get('date')
                    }
                ))

            client.upsert(cfg.qdrant.collection_name, points)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            continue

    logger.info(" Ingestion complete!")

if __name__ == "__main__":
    main()