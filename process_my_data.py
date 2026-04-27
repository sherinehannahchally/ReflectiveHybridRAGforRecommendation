import gzip
import json
import os
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = Path("example/reflective_hybrid_rec/data") 
OUTPUT_DIR = Path("dataset")

BOOKS_FILE = INPUT_DIR / "goodreads_books_young_adult.json.gz"
REVIEWS_FILE = INPUT_DIR / "goodreads_reviews_young_adult.json.gz"

def load_gzip_json(file_path):
    print(f"Reading {file_path}...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def save_as_jsonl(data, output_file):
    """Saves a list of dicts as a JSON Lines file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def process_data():
    if not BOOKS_FILE.exists() or not REVIEWS_FILE.exists():
        print(f"Error: Files not found in {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Process Items (Books)
    print("Processing Books...")
    clean_items = []
    valid_book_ids = set()

    for row in load_gzip_json(BOOKS_FILE):
        item = {
            "item_id": row.get("book_id"),
            "title": row.get("title"),
            "description": row.get("description", ""),
            "categories": ["Young Adult"],
            "stars": float(row.get("average_rating", 0.0)),
            "source": "goodreads",
            "text": f"{row.get('title')}. {row.get('description', '')}"
        }
        valid_book_ids.add(item["item_id"])
        clean_items.append(item)

    save_as_jsonl(clean_items, OUTPUT_DIR / "item.json")
    print(f"Saved {len(clean_items)} items to item.json")

    # 2. Process Reviews
    print("Processing Reviews...")
    clean_reviews = []
    
    for row in load_gzip_json(REVIEWS_FILE):
        book_id = row.get("book_id")
        if book_id not in valid_book_ids:
            continue

        review = {
            "review_id": row.get("review_id"),
            "user_id": row.get("user_id"),
            "item_id": book_id,
            "stars": float(row.get("rating", 0)),
            "text": row.get("review_text", ""),
            "timestamp": row.get("date_added", ""),
            "source": "goodreads"
        }
        clean_reviews.append(review)

    save_as_jsonl(clean_reviews, OUTPUT_DIR / "review.json")
    print(f"Saved {len(clean_reviews)} reviews to review.json")

    # 3. Process Users
    print("Generating User map...")
    users = list(set(r['user_id'] for r in clean_reviews))
    clean_users = [{"user_id": u, "source": "goodreads"} for u in users]
    
    save_as_jsonl(clean_users, OUTPUT_DIR / "user.json")
    print(f"Saved {len(clean_users)} users to user.json")

if __name__ == "__main__":
    process_data()