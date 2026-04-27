import json
import os
import logging
import time
from websocietysimulator import Simulator
from .gemini_llm import GeminiLLM 
from .baseline_agent import BaselineRecommendationAgent
# from .RecAgent_baseline import MyRecommendationAgent as BaselineRecommendationAgent
from .agent_v1 import HybridAgentV1                 # V1 (No Reflection)
from .reflective_hybrid_rec_agent import ReflectiveHybridRecAgent # V2 (Reflection)
from .config import ProjectConfig

class SimpleTask:
    def __init__(self, data): self.data = data
    def to_dict(self): return self.data
    def get(self, key, default=None): return self.data.get(key, default)
    def __getitem__(self, key): return self.data[key]

def evaluate_run(results, groundtruth, name):
    hits_1, hits_5, hits_10, total = 0, 0, 0, 0
    gt_map = {str(k): v for k, v in groundtruth.items()}
    
    norm_results = {}
    if isinstance(results, list):
        for i, item in enumerate(results):
            tid = str(item['task']['id']) if isinstance(item, dict) and 'task' in item else str(i)
            norm_results[tid] = item.get('output', []) if isinstance(item, dict) else item
    else: norm_results = results

    for tid, ranked in norm_results.items():
        if tid not in gt_map: continue
        # Handle dict GT (new format) or string GT (old format)
        target = gt_map[tid]['item_id'] if isinstance(gt_map[tid], dict) else gt_map[tid]
        
        if not isinstance(ranked, list): continue
        total += 1
        if target in ranked[:1]: hits_1 += 1
        if target in ranked[:5]: hits_5 += 1
        if target in ranked[:10]: hits_10 += 1

    return {"name": name, "HR@1": hits_1/total, "HR@5": hits_5/total, "HR@10": hits_10/total} if total else None
    

def main():
    data_dir = "dataset/"
    task_dir = "dataset/recommendation_tasks/"
    gt_dir = "dataset/groundtruth/"
    cfg = ProjectConfig()

    print("--- THREE-WAY BATTLE ---")
    simulator = Simulator(data_dir=data_dir, device="cpu", cache=False)
    
    with open(os.path.join(task_dir, "test.json"), "r") as f:
        manual_tasks = [SimpleTask(json.loads(line)) for line in f if line.strip()]
    with open(os.path.join(gt_dir, "test.json"), "r") as f:
        manual_gt = json.load(f)

    simulator.tasks = manual_tasks
    simulator.groundtruth = manual_gt
    simulator.set_llm(GeminiLLM(api_key=cfg.llm.api_key, model_name=cfg.llm.model_name))

    # 1. BASELINE
    print("\n [1/3] BASELINE AGENT...")
    simulator.set_agent(BaselineRecommendationAgent)
    t0 = time.time()
    res_base = simulator.run_simulation(number_of_tasks=len(manual_tasks), enable_threading=True, max_workers=8)
    score_base = evaluate_run(res_base, manual_gt, "Baseline")
    t_base = time.time() - t0

    # 2. HYBRID V1 (Retrieval Only)
    print("\n [2/3] HYBRID V1 (Raw Retrieval)...")
    simulator.set_agent(HybridAgentV1)
    t0 = time.time()
    # Use enable_threading=False for V1/V2 because they use internal threading
    res_v1 = simulator.run_simulation(number_of_tasks=len(manual_tasks), enable_threading=False, max_workers=1)
    score_v1 = evaluate_run(res_v1, manual_gt, "Hybrid V1")
    t_v1 = time.time() - t0

    # 3. HYBRID V2 (Reflective)
    print("\n [3/3] HYBRID V2 (Reflective)...")
    simulator.set_agent(ReflectiveHybridRecAgent)
    t0 = time.time()
    res_v2 = simulator.run_simulation(number_of_tasks=len(manual_tasks), enable_threading=False, max_workers=1)
    score_v2 = evaluate_run(res_v2, manual_gt, "Hybrid V2")
    t_v2 = time.time() - t0

    # REPORT
    print("\n" + "="*100)
    print(f"{'METRIC':<10} | {'BASELINE':<15} | {'HYBRID V1 (Raw)':<20} | {'HYBRID V2 (Reflect)':<20}")
    print("-" * 100)
    if score_base and score_v1 and score_v2:
        print(f"{'HR@1':<10} | {score_base['HR@1']:.4f}          | {score_v1['HR@1']:.4f}               | {score_v2['HR@1']:.4f}")
        print(f"{'HR@5':<10} | {score_base['HR@5']:.4f}          | {score_v1['HR@5']:.4f}               | {score_v2['HR@5']:.4f}")
        print(f"{'HR@10':<10} | {score_base['HR@10']:.4f}          | {score_v1['HR@10']:.4f}               | {score_v2['HR@10']:.4f}")
        print("-" * 100)
        print(f"{'Time':<10} | {t_base:.1f}s             | {t_v1:.1f}s                 | {t_v2:.1f}s")
    else:
        print(" Evaluation failed on one or more agents.")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()