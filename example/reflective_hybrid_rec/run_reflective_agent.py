import json
import os
import logging
from websocietysimulator import Simulator
from .gemini_llm import GeminiLLM 
from .reflective_hybrid_rec_agent import ReflectiveHybridRecAgent
from .config import ProjectConfig

class SimpleTask:
    def __init__(self, data):
        self.data = data
    def to_dict(self):
        return self.data
    def get(self, key, default=None):
        return self.data.get(key, default)
    def __getitem__(self, key):
        return self.data[key]

def manual_evaluation(results, groundtruth):
    print("\n" + "="*40)
    print("Manual Evaluation Report")
    print("="*40)
    
    # --- DEBUGGING OUTPUT ---
    print(f"DEBUG: Results Type: {type(results)}")
    if isinstance(results, list) and len(results) > 0:
        print(f"DEBUG: Sample Item: {results[0]}")
    elif isinstance(results, dict) and len(results) > 0:
        sample_key = list(results.keys())[0]
        print(f"DEBUG: Sample Item ({sample_key}): {results[sample_key]}")
    # ------------------------

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0
    
    gt_map = {str(k): v for k, v in groundtruth.items()}
    
    normalized_results = {}
    
    if isinstance(results, dict):
        normalized_results = results
    elif isinstance(results, list):
        for i, item in enumerate(results):
            # Try to extract ID from the object if it exists
            if isinstance(item, dict) and 'task' in item:
                task_id = str(item['task'].get('id', i))
                output = item.get('output', [])
                normalized_results[task_id] = output
            else:
                # Fallback: assume list index matches task ID
                normalized_results[str(i)] = item

    for task_id, item in normalized_results.items():
        sid = str(task_id)
        if sid not in gt_map: continue
            
        target_item = gt_map[sid]
        
        # Handle cases where item is just the list, or a dict with 'output'
        ranked_items = item
        if isinstance(item, dict) and 'output' in item:
            ranked_items = item['output']
            
        if not isinstance(ranked_items, list): continue

        total += 1
        if target_item in ranked_items[:1]: hits_at_1 += 1
        if target_item in ranked_items[:5]: hits_at_5 += 1
        if target_item in ranked_items[:10]: hits_at_10 += 1

    if total == 0:
        print(" No valid tasks evaluated. (Check debug info above)")
        return

    print(f" Evaluated {total} tasks.")
    print(f"🏆 HR@1:  {hits_at_1/total:.4f} ({hits_at_1}/{total})")
    print(f"🏆 HR@5:  {hits_at_5/total:.4f} ({hits_at_5}/{total})")
    print(f"🏆 HR@10: {hits_at_10/total:.4f} ({hits_at_10}/{total})")
    print("="*40 + "\n")

def main():
    data_dir = "dataset/" 
    task_dir = "dataset/recommendation_tasks/"
    groundtruth_dir = "dataset/groundtruth/"
    cfg = ProjectConfig()

    print("--- Initializing Simulator ---")
    simulator = Simulator(data_dir=data_dir, device="cpu", cache=False)
    
    print(" Loading tasks...")
    task_file = os.path.join(task_dir, "test.json")
    manual_tasks = []
    with open(task_file, "r") as f:
        for line in f:
            if line.strip():
                manual_tasks.append(SimpleTask(json.loads(line)))
        
    gt_file = os.path.join(groundtruth_dir, "test.json")
    with open(gt_file, "r") as f:
        manual_gt = json.load(f)

    simulator.tasks = manual_tasks
    simulator.groundtruth = manual_gt
    print(f" Loaded {len(manual_tasks)} tasks.")

    llm = GeminiLLM(api_key=cfg.llm.api_key, model_name=cfg.llm.model_name)
    simulator.set_llm(llm)
    simulator.set_agent(ReflectiveHybridRecAgent)

    print("--- Starting Simulation ---")
    
    # Running sequentially to stop PyTorch crashes
    agent_outputs = simulator.run_simulation(
        number_of_tasks=len(manual_tasks), 
        enable_threading=True, 
        max_workers=2, 
    )
    
    manual_evaluation(agent_outputs, manual_gt)

if __name__ == "__main__":
    main()