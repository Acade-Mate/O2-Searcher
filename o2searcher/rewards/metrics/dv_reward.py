from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import re
from typing import List, Dict, Any, Tuple
from utils import AbstractAgent
import json
import os

class QueryIndependenceTransformer(AbstractAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_transformer.to(self.device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, 'prompts')
        self._set_system_prompt_from_text(os.path.join(prompts_dir, 'translation_SP.txt'))
        

        self.MAX_QUERIES = 20  # max queries
        self.SIMILARITY_THRESHOLD = 0.98  # similarity threshold
        self.REDUNDANCY_THRESHOLD = 0.8  # redundancy threshold
        self.BASE_SCORE = 0.5  # base score
        self.token_len_threshold = 8
    def _is_chinese(self, text: str) -> bool:
        """check if text contains chinese"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def _translate_to_english(self, queries: List[str]) -> List[str]:
        """translate chinese queries to english"""
        prompt = """Please translate the following queries from Chinese to English. Keep the translation accurate and professional. Return only the translations in a JSON array format.

        Queries:
        {}

        Output format:
        ["translation1", "translation2", ...]"""
        
        # only translate queries containing chinese
        chinese_queries = [q for q in queries if self._is_chinese(q)]
        if not chinese_queries:
            return queries
            
        response = self.response(prompt.format("\n".join(chinese_queries)))
        
        try:

            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
                
            translations = json.loads(cleaned_response)
            
            # create translation map
            translation_map = dict(zip(chinese_queries, translations))
            
            # return translated queries
            return [translation_map.get(q, q) for q in queries]
            
        except json.JSONDecodeError:
            print("translation parsing failed, return original queries")
            return queries
    
    def _calculate_query_weights(self, n: int) -> float:
        """
        calculate query quantity weight
        - weight increases with query quantity between 3-10
        - weight remains high but slightly decreases between 10-20
        """
        if n <= 5:
            return 0.5
        elif n <= 15:
            return 0.5 + (n - 5) * 0.05  # weight from 0.55 to 0.9 for 3-10 queries
        else:
            return 0.9 - (n - 15) * 0.02  # weight slightly decreases from 0.9 for 15-20 queries
    
    def _calculate_token_length_penalty(self, token_len: int) -> float:
        """
        calculate token length penalty factor
        penalty increases with token length below threshold
        """
        if token_len >= self.token_len_threshold:
            return 1.0  # no penalty
        elif token_len <= 0:
            return 0.1 # avoid division by zero or negative, set a minimum penalty factor
        else:
            # linear penalty: maximum penalty for length 1, minimum penalty near threshold
            penalty_factor = token_len / self.token_len_threshold
            # can set a minimum penalty limit to prevent factor from being too small
            return max(0.1, penalty_factor)
    
    def calculate_independence(self, queries: List[str]) -> Dict[str, Any]:
        """calculate query independence"""
        # check query quantity
        if not queries:
            return self._get_empty_result()
        
        if len(queries) > self.MAX_QUERIES:
            queries = queries[:self.MAX_QUERIES]  # truncate to 20 queries
            
        n = len(queries)
        if n < 2:
            return self._get_empty_result()
            
        # translate queries
        translated_queries = self._translate_to_english(queries)
        
        # calculate embeddings
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(
                translated_queries,
                convert_to_tensor=True,
                device=self.device
            )
            if self.device.type == "cuda":
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = embeddings.numpy()
        
        similarity_matrix = cosine_similarity(embeddings)
        independence_matrix = 1 - similarity_matrix
        
        is_all_similar = np.all(similarity_matrix > self.SIMILARITY_THRESHOLD)
        
        quantity_weight = self._calculate_query_weights(n)
        
        pairwise_scores = []
        redundancy_groups = []
        query_independence_scores = {}
        
        for i in range(n):
            scores_i = []
            for j in range(n):
                if i != j:
                    if is_all_similar:
                        score = self.BASE_SCORE * (0.2 ** (n-1))  
                    else:
                        score = float(independence_matrix[i][j])
                    
                    if i < j: 
                        pairwise_scores.append([queries[i], queries[j], round(score, 2)])
                    
                    if score < (1 - self.REDUNDANCY_THRESHOLD): 
                        if i < j:  
                            redundancy_groups.append([queries[i], queries[j]])
                    scores_i.append(score)
            
            if is_all_similar:
                base_score = self.BASE_SCORE * (0.2 ** (n-1))
                query_independence_scores[queries[i]] = round(base_score, 2)
            else:
                avg_score = np.mean(scores_i)
                weighted_score = avg_score * quantity_weight
                query_independence_scores[queries[i]] = round(weighted_score, 2)
        
        # calculate token length (use simple space split as approximation)
        # note: more accurate method is to use model tokenizer
        token_lengths = [len(self.sentence_transformer.tokenizer(q, add_special_tokens=True)['input_ids']) for q in translated_queries]

        # apply token length penalty to each query independence score
        penalized_query_scores = {}
        original_scores_before_penalty = query_independence_scores.copy() # optional: keep original scores for comparison

        for i, query in enumerate(queries):
            original_score = query_independence_scores.get(query, 0) # get original score
            token_len = token_lengths[i]
            penalty_factor = self._calculate_token_length_penalty(token_len)
            penalized_score = original_score * penalty_factor
            penalized_query_scores[query] = round(penalized_score, 2) # update score

        # calculate overall score using penalized scores
        if is_all_similar:
            # if all queries are similar, penalty was already handled at the start, but consider whether token penalty should also be applied
            # here we keep the original, as the is_all_similar penalty is already heavy
             overall_score = self.BASE_SCORE * (0.2 ** (n-1))
        else:
            # calculate average using penalized scores
            penalized_scores_list = list(penalized_query_scores.values())
            # note: here we don't need to multiply by quantity_weight, as it was already applied when calculating individual scores
            # if want quantity_weight to only affect overall score, can multiply here
            overall_score = np.mean(penalized_scores_list) # use penalized scores

        # calculate redundancy and independence coverage (use penalized scores or original scores, depending on definition)
        # here we continue using penalized scores to define independence
        redundant_queries = set(sum(redundancy_groups, [])) # redundancy group based on similarity, not affected by token length
        qrr = len(redundant_queries) / n
        # use penalized scores to determine independent queries
        independent_queries = [q for q, s in penalized_query_scores.items() if s >= 0.5]
        icr = len(independent_queries) / n
        
        return {
            "pairwise_scores": pairwise_scores,
            "overall_independence_score": round(overall_score, 2),
            "redundancy_groups": redundancy_groups,
            "query_redundancy_rate": round(qrr, 2),
            "independence_coverage_rate": round(icr, 2),
            "query_independence_scores": penalized_query_scores,
            "query_quantity_weight": round(quantity_weight, 2),
            "total_queries": n
        }
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "pairwise_scores": [],
            "overall_independence_score": 0.0,
            "redundancy_groups": [],
            "query_redundancy_rate": 1.0,
            "independence_coverage_rate": 0.0,
            "query_independence_scores": {},
            "query_quantity_weight": 0.0,
            "total_queries": 0
        }