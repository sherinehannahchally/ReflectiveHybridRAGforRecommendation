
import json
import os
import logging
from websocietysimulator import Simulator
from .gemini_llm import GeminiLLM 
from .reflective_hybrid_rec_agent import ReflectiveHybridRecAgent
from .config import ProjectConfig

# Wrapper to satisfy simulator's requirements
class SimpleTask:
    def __init__(self, data):
        self.data = data
    def to_dict(self): return self.data
    def get(self, key, default=None): return self.data.get(key, default)
    def __getitem__(self, key): return self.data[key]

def main():
    data_dir = "dataset/" 
    task_dir = "dataset/recommendation_tasks/"
    groundtruth_dir = "dataset/groundtruth/"
    cfg = ProjectConfig()

    print("--- DEBUG MODE INITIALIZED ---")
    simulator = Simulator(data_dir=data_dir, device="cpu", cache=False)
    
    # Load just 3 tasks for testing
    task_file = os.path.join(task_dir, "test.json")
    if not os.path.exists(task_file): task_file = os.path.join(task_dir, "test.jsonl")
    
    manual_tasks = []
    with open(task_file, "r") as f:
        for line in f:
            if line.strip(): manual_tasks.append(SimpleTask(json.loads(line)))
            if len(manual_tasks) >= 3: break # Only need 3 to check format
            
    # Load GT
    gt_file = os.path.join(groundtruth_dir, "test.json")
    with open(gt_file, "r") as f:
        manual_gt = json.load(f)

    simulator.tasks = manual_tasks
    simulator.groundtruth = manual_gt
    
    # Setup Agent
    llm = GeminiLLM(api_key=cfg.llm.api_key, model_name=cfg.llm.model_name)
    simulator.set_llm(llm)
    simulator.set_agent(ReflectiveHybridRecAgent)

    print(f"Running simulation on {len(manual_tasks)} tasks...")
    
    # Run
    agent_outputs = simulator.run_simulation(
        number_of_tasks=len(manual_tasks), 
        enable_threading=True, 
        max_workers=2, 
    )
    
    print("\n" + "="*40)
    print("🛑 DEBUG OUTPUT - PLEASE COPY THIS")
    print("="*40)
    print(f"Type of agent_outputs: {type(agent_outputs)}")
    print(f"First Ground Truth Key: {list(manual_gt.keys())[0]}")
    
    if isinstance(agent_outputs, list):
        print(f"Length of outputs: {len(agent_outputs)}")
        if len(agent_outputs) > 0:
            print(f"First item in outputs: {agent_outputs[0]}")
    elif isinstance(agent_outputs, dict):
        print(f"Keys in outputs: {list(agent_outputs.keys())}")
        first_key = list(agent_outputs.keys())[0]
        print(f"First item: {agent_outputs[first_key]}")
    else:
        print(f"Raw Output: {agent_outputs}")
        
    print("="*40)

if __name__ == "__main__":
    main()