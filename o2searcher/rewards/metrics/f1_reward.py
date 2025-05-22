from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
from typing import List, Union
import torch
from utils import AbstractAgent
from scipy.optimize import linear_sum_assignment
import os

class FindingSentenceEvaluator(AbstractAgent):
    def __init__(self, model_name: str = ''):
        super().__init__(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model.to(self.device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, 'prompts')
        self._set_system_prompt_from_text(os.path.join(prompts_dir, 'translation_SP.txt'))

    def _is_chinese(self, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _translate_to_english(self, texts: List[str]) -> List[str]:
        prompt = """Please translate the following texts from Chinese to English. Keep the translation accurate and professional. Return only the translations in a JSON array format.

        Texts:
        {}

        Output format:
        ["translation1", "translation2", ...]"""

        chinese_texts = [text for text in texts if self._is_chinese(text)]
        if not chinese_texts:
            return texts
            
        response = self.response(prompt.format("\n".join(chinese_texts)))
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
                
            translations = json.loads(cleaned_response)
            
            translation_map = dict(zip(chinese_texts, translations))
            
            return [translation_map.get(text, text) for text in texts]
            
        except json.JSONDecodeError:
            print("translation failed, return original text")
            return texts

    def extract_findings(self, text: str) -> List[str]:
        """extract findings from text"""
        # extract <answer> tag content
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.finditer(answer_pattern, text, re.DOTALL)
        answer_match = list(answer_match)
        if len(answer_match) <= 1:
            return []
        
        content = answer_match[-1].group(1).strip()
        if not content.strip() or content.strip().lower() == 'and':
            return []
        
        # split each point
        findings = [point.strip() for point in content.split('-') if point.strip()]
        return findings
            
    def calculate_scores(self,
                        generated_points: Union[str, List[str]],
                        reference_points: List[str],
                        threshold: float = 0.85) -> dict:
        """calculate similarity score between generated text and reference text"""

        # if input is string, extract findings
        if isinstance(generated_points, str):
            extracted_points = self.extract_findings(generated_points)
        else:
            extracted_points = generated_points # if already list, use directly

        # 1. filter invalid answer
        # remove empty string or only contain space
        valid_generated_points = [p for p in extracted_points if p and p.strip()]
        # can add more complex filter logic here, such as filter out only 'and'
        if not valid_generated_points:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        # check if reference points is empty
        if not reference_points:
             return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} # if no reference points, cannot calculate

        # use filtered valid points for further calculation
        generated_points = valid_generated_points

        num_gen_points = len(generated_points)
        num_ref_points = len(reference_points)
        
        # translate chinese text
        translated_gen_points = self._translate_to_english(generated_points)
        translated_ref_points = self._translate_to_english(reference_points)

        # get text embeddings
        gen_embeddings = self.sentence_model.encode(translated_gen_points)
        ref_embeddings = self.sentence_model.encode(translated_ref_points)

        # calculate similarity matrix
        similarity_matrix = cosine_similarity(gen_embeddings, ref_embeddings)
        
        # for each reference point, if multiple generate point similarity exceeds threshold,
        # only keep the highest similarity, others set to 0
        for j in range(similarity_matrix.shape[1]):  # for each reference point
            col = similarity_matrix[:, j]
            matches = col >= threshold
            if np.sum(matches) > 1:  # if multiple matches
                # find the best match
                best_idx = np.argmax(col)
                # set others to 0
                matches = np.zeros_like(matches)
                matches[best_idx] = True
                similarity_matrix[:, j] = matches * col

        # use hungarian algorithm for optimal matching
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # find all matches with similarity exceeds threshold
        valid_matches = [(i, j) for i, j in zip(row_ind, col_ind) 
                         if similarity_matrix[i, j] >= threshold]
        
        # calculate precision and recall
        precision = len(valid_matches) / num_gen_points if num_gen_points > 0 else 0.0
        recall = len(valid_matches) / num_ref_points if num_ref_points > 0 else 0.0

        # calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

def f1_reward_fn(generated_text: str, reference_points: np.ndarray, threshold: float = 0.85) -> dict:
    """wrapper function for calculating F1 reward score"""
    evaluator = FindingSentenceEvaluator()
    return evaluator.calculate_scores(generated_text, reference_points.tolist(), threshold)

if __name__ == '__main__':
    # test data
    test_input = """<answer>  </answer> \n <answer>

    </answer>"""
    
    test_input_2 = """<answer>  </answer> \n <answer>
    </answer>"""


    test_merged_findings = np.array([

    ], dtype=object)
    
    # calculate metrics
    output = f1_reward_fn(test_input, test_merged_findings, threshold=0.7)
    output_2 = f1_reward_fn(test_input_2, test_merged_findings, threshold=0.7)
    print(f"\nResults for original text:", output)
    print(f"\nResults for duplicate text:", output_2)
