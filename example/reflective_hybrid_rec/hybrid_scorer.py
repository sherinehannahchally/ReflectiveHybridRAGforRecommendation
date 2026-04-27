from __future__ import annotations
import torch
import logging
import threading
from typing import List, Dict
from dataclasses import dataclass
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForMaskedLM
import google.generativeai as genai

from .config import ProjectConfig

logger = logging.getLogger(__name__)

@dataclass
class CandidateFeatures:
    item_id: str
    title: str
    text: str
    avg_stars: float

class HybridScorer:
    def __init__(self, config: ProjectConfig, use_reflections: bool = True) -> None:
        self.cfg = config
        self.device = "cpu"
        self.use_reflections = use_reflections
        self.splade_lock = threading.Lock()

        logger.info(f"Loading SPLADE model on {self.device}...")
        self.splade_tokenizer = AutoTokenizer.from_pretrained(config.qdrant.sparse_model_id)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(
            config.qdrant.sparse_model_id,
            low_cpu_mem_usage=False 
        )
        self.splade_model.to(self.device)
        self.splade_model.eval()

        self.client = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        
        genai.configure(api_key=config.llm.api_key)
        self.gemini = genai.GenerativeModel(config.llm.model_name)

    def _get_splade_vector(self, text: str) -> models.SparseVector:
        if not text: return models.SparseVector(indices=[], values=[])
        with self.splade_lock:
            try:
                tokens = self.splade_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    logits = self.splade_model(**tokens).logits
                
                vec = torch.max(torch.log(1 + torch.relu(logits)) * tokens['attention_mask'].unsqueeze(-1), dim=1)[0].squeeze()
                indices = vec.nonzero().squeeze().cpu().tolist()
                values = vec[indices].cpu().tolist()
                if isinstance(indices, int): indices, values = [indices], [values]
                return models.SparseVector(indices=indices, values=values)
            except Exception as e:
                logger.error(f"SPLADE Error: {e}")
                return models.SparseVector(indices=[], values=[])

    def retrieve_user_context(self, user_id: str, query_text: str) -> str:
        # 1. Generate Vectors
        try:
            dense_vec = genai.embed_content(model=self.cfg.qdrant.dense_model_name, content=query_text, task_type="retrieval_query")['embedding']
            sparse_vec = self._get_splade_vector(query_text)
        except Exception: return ""

        final_context = []

        # 2. SIDE CAR LOOKUP (Reflection) - Only if enabled for V2
        if self.use_reflections:
            try:
                reflection_hits = self.client.search(
                    collection_name=self.cfg.qdrant.reflection_collection,
                    query_vector=models.NamedVector(name="dense", vector=dense_vec),
                    query_filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]),
                    limit=1
                )
                if reflection_hits:
                    final_context.append(f"USER INSIGHTS (High-Level Profile):\n{reflection_hits[0].payload['text']}")
            except Exception: pass

        # 3. MAIN LOOKUP (Raw History)
        try:
            results = self.client.query_points(
                collection_name=self.cfg.qdrant.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vec, using="gemini-dense", filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]), limit=10),
                    models.Prefetch(query=sparse_vec, using="splade-sparse", filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]), limit=10),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=5,
                with_payload=True,
            )
            
            if results.points:
                raw_text = "\n".join([f"- [Rated {p.payload.get('stars','-')}]: {p.payload.get('text','')[:200]}" for p in results.points])
                final_context.append(f"RAW EVIDENCE (Specific Interactions):\n{raw_text}")
                
        except Exception: pass

        return "\n\n".join(final_context)

    def predict_score_with_gemini(self, candidate: CandidateFeatures, context: str) -> float:
        if not context: return 0.5 

        # Below is the structured CoT prompt for stability and accuracy
        prompt = f"""
        Act as a recommendation expert.
        
        Candidate Item: {candidate.title}
        Description: {candidate.text[:300]}
        
        User History:
        {context}
        
        Task:
        1. Analyze how well the candidate matches the user's past preferences (genre, style, tone).
        2. Think step-by-step about why the user would like or dislike this.
        3. Finally, assign a likelihood score between 0.0 and 1.0.
        
        Format your answer strictly as follows:
        Reasoning: [Your full step-by-step reasoning]
        Final Score: [The Number]
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            text = response.text.strip()
            
            # --- Robust Parsing Logic ---
            import re
            # Look for the score, typically at the end
            match = re.search(r"(?:Final Score:)?\s*(\d+(?:\.\d+)?)", text.split('\n')[-1])
            if not match:
                # Fallback: search the whole string for the last number
                matches = re.findall(r"0\.\d+|1\.0|0|1", text)
                if matches:
                    return float(matches[-1])
                return 0.5
            
            val = float(match.group(1))
            return min(max(val, 0.0), 1.0)
        except: return 0.5

    def score_candidates(self, user_id: str, candidates: List[CandidateFeatures]) -> Dict[str, float]:
        scores = {}
        # The agent logic parallelizes this loop using ThreadPoolExecutor
        for cand in candidates:
            q_text = f"{cand.title} {cand.text}"
            context = self.retrieve_user_context(user_id, q_text)
            scores[cand.item_id] = self.predict_score_with_gemini(cand, context)
        return scores