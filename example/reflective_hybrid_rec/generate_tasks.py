import json
import random
import os
from pathlib import Path

def generate_tasks():
    dataset_dir = Path("dataset")
    task_dir = dataset_dir / "recommendation_tasks"
    gt_dir = dataset_dir / "groundtruth"
    
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    print("Loading data...")
    # Read using JSON Lines method
    # with open(dataset_dir / "review.json", 'r') as f:
    with open(dataset_dir / "review_small.json", 'r') as f: # <--- Update this
        reviews = [json.loads(line) for line in f]
    with open(dataset_dir / "item.json", 'r') as f:
        items = [json.loads(line) for line in f]
    
    item_ids = [i['item_id'] for i in items]
    user_reviews = {}
    for r in reviews:
        user_reviews.setdefault(r['user_id'], []).append(r['item_id'])
    
    print(f"Generating tasks for {len(user_reviews)} users...")
    tasks = []
    groundtruths = {}
    
    for i, (user_id, history) in enumerate(user_reviews.items()):
        if len(history) < 2: continue
        
        target_item = history[-1]
        
        available_negatives = [i for i in item_ids if i not in history]
        if len(available_negatives) < 19:
            continue
            
        negatives = random.sample(available_negatives, 19)
        candidates = negatives + [target_item]
        random.shuffle(candidates)
        
        tasks.append({
            "id": i,
            "user_id": user_id,
            "candidate_items": candidates
        })
        groundtruths[str(i)] = target_item
        
        # Limit to 10 for Quick Testing
        # if i >= 10: break 
        
    with open(task_dir / "test.json", 'w') as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
            
    with open(gt_dir / "test.json", 'w') as f:
        json.dump(groundtruths, f)
        
    print(f"Generated {len(tasks)} tasks in {task_dir}")

if __name__ == "__main__":
    generate_tasks()