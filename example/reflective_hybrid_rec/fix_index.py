import logging
from qdrant_client import QdrantClient, models
from example.reflective_hybrid_rec.config import ProjectConfig

# Setup
logging.basicConfig(level=logging.INFO)
cfg = ProjectConfig()
client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)

def create_indices():
    print(f"Creating indices for collection: {cfg.qdrant.collection_name}...")
    
    # 1. Create Index for user_id (Required for filtering history)
    try:
        client.create_payload_index(
            collection_name=cfg.qdrant.collection_name,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print(" Successfully created index for 'user_id'")
    except Exception as e:
        print(f" Could not create user_id index (might already exist): {e}")

    # 2. Create Index for item_id (Good for debugging)
    try:
        client.create_payload_index(
            collection_name=cfg.qdrant.collection_name,
            field_name="item_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print(" Successfully created index for 'item_id'")
    except Exception as e:
        print(f" Could not create item_id index: {e}")

if __name__ == "__main__":
    create_indices()