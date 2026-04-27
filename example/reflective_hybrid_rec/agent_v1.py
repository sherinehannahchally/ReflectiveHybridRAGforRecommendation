from __future__ import annotations
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from websocietysimulator.agent.recommendation_agent import RecommendationAgent
except ImportError:
    from websocietysimulator.agent import RecommendationAgent

from .config import ProjectConfig
from .hybrid_scorer import HybridScorer, CandidateFeatures

class HybridAgentV1(RecommendationAgent):
    """
    V1 Agent: Uses Hybrid Search (Dense+Sparse). Note: No Reflections.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = ProjectConfig()
        self.scorer = HybridScorer(self.config, use_reflections=False)

    def insert_task(self, task):
        self.task = task if isinstance(task, dict) else task.to_dict()

    def workflow(self) -> List[str]:
        task = self.task
        user_id = str(task.get("user_id") or task.get("uid"))
        candidate_ids = list(task.get("candidate_items") or task.get("items"))

        candidates = self._build_candidate_features(candidate_ids)
        scores = {}
        
        def process_candidate(cand):
            rating = self.scorer.score_candidates(user_id, [cand])
            return cand.item_id, rating.get(cand.item_id, 3.0)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_cand = {executor.submit(process_candidate, c): c for c in candidates}
            for future in as_completed(future_to_cand):
                try:
                    item_id, score = future.result()
                    scores[item_id] = score
                except:
                    scores[future_to_cand[future].item_id] = 3.0

        ranked = sorted(candidates, key=lambda c: scores.get(c.item_id, 0.0), reverse=True)
        return [c.item_id for c in ranked]

    def _build_candidate_features(self, candidate_ids: List[str]) -> List[CandidateFeatures]:
        candidates = []
        for item_id in candidate_ids:
            try: item = self.interaction_tool.get_item(item_id=item_id)
            except: item = {}
            title = item.get("name") or item.get("title") or "Unknown"
            meta = []
            if "categories" in item: meta.append(f"Categories: {item['categories']}")
            if "text" in item: meta.append(item['text'])
            candidates.append(CandidateFeatures(item_id, title, " ".join(meta), item.get("stars", 0.0)))
        return candidates