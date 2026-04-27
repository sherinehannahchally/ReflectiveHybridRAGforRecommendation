# example/reflective_hybrid_rec/reflective_hybrid_rec_agent.py

from __future__ import annotations
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import RecommendationAgent (handling folder naming structure)
try:
    from websocietysimulator.agent.recommendation_agent import RecommendationAgent
except ImportError:
    from websocietysimulator.agent import RecommendationAgent

from .config import ProjectConfig
from .hybrid_scorer import HybridScorer, CandidateFeatures

class ReflectiveHybridRecAgent(RecommendationAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Load Config & Scorer
        self.config = ProjectConfig()
        self.scorer = HybridScorer(self.config)

    def insert_task(self, task):
        """
        Handle Manual Injection (Dict) vs Simulator Injection (Task Object).
        """
        if isinstance(task, dict):
            self.task = task
        else:
            self.task = task.to_dict()

    def workflow(self) -> List[str]:
        """
        Main Agent Workflow:
        1. Parse Task
        2. Format Candidates
        3. Parallel Scoring
        4. Rank & Return
        """
        task = self.task
        
        # 1. Extract Info
        user_id = str(task.get("user_id") or task.get("uid"))
        candidate_ids = list(task.get("candidate_items") or task.get("items"))

        # 2. Build Candidate Features
        candidates = self._build_candidate_features(candidate_ids)

        # 3. Parallel Scoring
        # We use threading to score all 20 candidates simultaneously.
        # This reduces task time from ~40s to ~3s.
        scores = {}
        
        def process_candidate(cand):
            # This calls Qdrant (Hybrid Search) + Gemini (Reasoning)
            score = self.scorer.score_candidates(user_id, [cand])
            return cand.item_id, score.get(cand.item_id, 0.0)

        # Use 10 workers to blast through the API calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_cand = {
                executor.submit(process_candidate, cand): cand 
                for cand in candidates
            }
            
            for future in as_completed(future_to_cand):
                try:
                    item_id, score = future.result()
                    scores[item_id] = score
                except Exception as e:
                    # Fallback for failed items
                    scores[future_to_cand[future].item_id] = 0.0

        # 4. Rank
        ranked = sorted(
            candidates,
            key=lambda c: scores.get(c.item_id, 0.0),
            reverse=True,
        )

        return [c.item_id for c in ranked]

    def _build_candidate_features(self, candidate_ids: List[str]) -> List[CandidateFeatures]:
        candidates = []
        interaction_tool = self.interaction_tool

        for item_id in candidate_ids:
            try:
                item = interaction_tool.get_item(item_id=item_id)
            except Exception:
                item = {}

            title = item.get("name") or item.get("title") or "Unknown"
            
            meta_parts = []
            if "categories" in item:
                meta_parts.append(f"Categories: {item['categories']}")
            if "text" in item:
                meta_parts.append(item['text'])
            
            candidates.append(
                CandidateFeatures(
                    item_id=item_id,
                    title=title,
                    text=" ".join(meta_parts),
                    avg_stars=item.get("stars", 0.0),
                )
            )

        return candidates