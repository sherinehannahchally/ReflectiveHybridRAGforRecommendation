import json
from pathlib import Path

# Config
INPUT_FILE = Path("dataset/review.json")
OUTPUT_FILE = Path("dataset/review_small.json")
LIMIT = 50000  # Number of reviews to keep

def create_subset():
    print(f"Creating subset of {LIMIT} reviews...")
    count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            fout.write(line)
            count += 1
            if count >= LIMIT:
                break
                
    print(f" Created {OUTPUT_FILE} with {count} records.")

if __name__ == "__main__":
    create_subset()