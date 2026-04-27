import json
import uuid
import logging
import time
from tqdm import tqdm
from pathlib import Path
from qdrant_client import QdrantClient, models
import google.generativeai as genai
from .config import ProjectConfig

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReflectionGenerator")
cfg = ProjectConfig()

genai.configure(api_key=cfg.llm.api_key)
model = genai.GenerativeModel(cfg.llm.model_name)

def get_test_users():
    """Extracts unique User IDs from your generated tasks."""
    users = set()
    paths = [
        Path("dataset/recommendation_tasks/test.json"),
        Path("dataset/recommendation_tasks/test.jsonl")
    ]
    
    for p in paths:
        if p.exists():
            with open(p, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        users.add(data['user_id'])
            break
            
    return list(users)

def get_user_history(user_ids):
    history = {uid: [] for uid in user_ids}
    target_set = set(user_ids)
    
    # Read from the small file if it exists
    data_path = Path("dataset/review.json")
    if Path("dataset/review_small.json").exists():
        data_path = Path("dataset/review_small.json")

    logger.info(f"Reading history from {data_path}...")
    with open(data_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            if r['user_id'] in target_set:
                history[r['user_id']].append(r)
    return history

def generate_insight(user_id, reviews):
    if not reviews: return None

    review_texts = [f"- Book: {r.get('item_id')} | Stars: {r.get('stars')} | Review: {r.get('text','')[:200]}" for r in reviews[:20]]
    context_str = "\n".join(review_texts)

    prompt = f"""
    Analyze the following review history for User {user_id}.
    Identify 3-5 distinct, high-level insights about their literary taste.
    Focus on: Genres, Tropes (e.g. enemies-to-lovers), Writing Style, and Dealbreakers.
    
    Raw History:
    {context_str}
    
    Output ONLY the insights as a bulleted list.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return None

def main():
    client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)
    
    # Recreate Reflection Collection
    if client.collection_exists(cfg.qdrant.reflection_collection):
        client.delete_collection(cfg.qdrant.reflection_collection)
    
    client.create_collection(
        collection_name=cfg.qdrant.reflection_collection,
        vectors_config={
            "dense": models.VectorParams(size=cfg.qdrant.dense_dim, distance=models.Distance.COSINE)
        },
    )
    logger.info(f"Created Sidecar: {cfg.qdrant.reflection_collection}")

    users = get_test_users()
    logger.info(f"Found {len(users)} test users.")
    
    histories = get_user_history(users)
    points = []
    
    logger.info("Generating Reflections...")
    for uid in tqdm(users):
        user_reviews = histories.get(uid, [])
        if len(user_reviews) < 2: continue 

        insight_text = generate_insight(uid, user_reviews)
        if not insight_text: continue

        try:
            dense_resp = genai.embed_content(
                model=cfg.qdrant.dense_model_name,
                content=insight_text,
                task_type="retrieval_document"
            )
            embedding = dense_resp['embedding']
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": embedding},
                payload={
                    "user_id": uid,
                    "text": insight_text,
                    "type": "reflection"
                }
            ))
            # Sleep to avoid rate limits
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Embedding Error: {e}")

    if points:
        client.upsert(cfg.qdrant.reflection_collection, points)
        logger.info(f"Successfully uploaded {len(points)} user profiles.")

if __name__ == "__main__":
    main()