import logging
import re
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase

logger = logging.getLogger("BaselineAgent")

class RecPlanning(PlanningBase):
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        prompt = f'''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Finally, I need to recommend items", "reasoning instruction": "None"}}
Task: {task_description}
'''
        return prompt

class RecReasoning(ReasoningBase):
    def __init__(self, profile_type, memory_type, llm):
        super().__init__(
            profile_type_prompt=profile_type, 
            memory=memory_type, 
            llm=llm
        )

    def __call__(self, task_description):
        prompt = f"""
        You are a recommendation assistant.
        Task: {task_description}
        
        Please rank the candidate items based on the user's likely preference.
        Output the list of Item IDs strictly in this format:
        ['item_id1', 'item_id2', ...]
        """
        result = self.llm(prompt)
        return result

class BaselineRecommendationAgent(RecommendationAgent):
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
        task_desc = f"User ID: {self.task.get('user_id')}. Candidate Items: {self.task.get('candidate_items')}"
        
        # 2. Reason
        result = self.reasoning(task_desc)

        # 3. Parse Output
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                return eval(match.group())
            return []
        except:
            return []