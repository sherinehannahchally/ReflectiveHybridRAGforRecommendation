import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
import ast
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time

logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Finally, I need to recommend items", "reasoning instruction": "None"}}
Task: {task_description}
'''
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type, memory_type, llm):
        # --- FIX: Use correct argument names for the parent class ---
        super().__init__(
            profile_type_prompt=profile_type, 
            memory=memory_type, 
            llm=llm
        )

    def __call__(self, task_description, **kwargs):
        prompt = f"""
        You are a recommendation assistant.
        Task: {task_description}
        
        Please rank the candidate items based on the user's likely preference.
        Output the list of Item IDs strictly in this format:
        ['item_id1', 'item_id2', ...]
        """
        result = self.llm(prompt)
        return result

class MyRecommendationAgent(RecommendationAgent):
    """
    The Official Baseline Agent
    """
    def __init__(self, *args, **kwargs):
        # Set defaults required by ReasoningBase
        self.profile_type = "text" 
        self.memory_type = "text"
        
        super().__init__(*args, **kwargs)
        
        if not hasattr(self, 'reasoning'):
            self.setup_modules()

    def setup_modules(self):
        # Initialize modules with correct parameters
        self.planner = RecPlanning(llm=self.llm)
        self.reasoning = RecReasoning(
            profile_type=self.profile_type, 
            memory_type=self.memory_type, 
            llm=self.llm
        )

    def workflow(self):
        # 1. Construct Description
        task_description = f"""
        User ID: {self.task.get('user_id')}. 
        Candidate Items: {self.task.get('candidate_items')}
        """
        # Get candidates for robust fallback
        candidate_ids = self.task.get('candidate_items', [])
        
        # 2. Reason
        # The prompt is constructed in RecReasoning
        result = self.reasoning(task_description)

        try:
            match = re.search(r"\\[.*\\]", result, re.DOTALL)
            if match:
                list_str = match.group()
                # Safely parse the Python list string
                ranked_list = ast.literal_eval(list_str)
                
                # Check if the parsed list contains actual items (not just empty strings)
                if ranked_list and all(isinstance(item, str) and item for item in ranked_list):
                    print('Processed Output (Parsed List):', ranked_list)
                    return ranked_list
                else:
                    # If parsed list is invalid, fall back
                    print("Parsed list invalid (e.g., ['']). Falling back to candidates.")
                    return candidate_ids
            else:
                print("No list found in LLM output. Falling back to candidate list.")
                # Fallback 1: Return the original, unranked candidates
                return candidate_ids
        except Exception as e:
            print(f'Format error during parsing: {e}. Falling back to candidate list.')
            # Fallback 2: Return the original, unranked candidates
            return candidate_ids

if __name__ == "__main__":
    task_set = "amazon" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="your data_dir", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./track2/{task_set}/tasks", groundtruth_dir=f"./track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    simulator.set_llm(InfinigenceLLM(api_key="your api_key"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
