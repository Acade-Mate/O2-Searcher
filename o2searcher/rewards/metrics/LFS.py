from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import re
import json
import os
from o2searcher.rewards.metrics.utils import AbstractAgent
import uvicorn

class FindingsRecallCalculator(AbstractAgent):
    def __init__(self, model_name):
        super().__init__(model_name)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, 'prompts')
        sp_path = os.path.join(prompts_dir, 'lfs_SP.txt')
        template_path = os.path.join(prompts_dir, 'lfs_prompt_template.txt')
        if not os.path.exists(sp_path):
            raise FileNotFoundError(f"System prompt file not found: {sp_path}")
        self._set_system_prompt_from_text(sp_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Prompt template file not found: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def calculate_recall(self, input_findings: str, target_findings: List[str]) -> Tuple[List[List[str]], dict]:
        input_findings_list = [f.strip() for f in input_findings.split('\n') if f.strip()]
        target_findings_list = [f.strip() for f in target_findings if f.strip()]

        input_list_str = '\n'.join(f"{i+1}. {f}" for i, f in enumerate(input_findings_list))
        target_list_str = '\n'.join(f"{i+1}. {f}" for i, f in enumerate(target_findings_list))
        json_format = '''[
           ["input_finding_1", "matched_target_finding_1"],
           ["input_finding_2", "matched_target_finding_2"],
           ...
            ]'''

        prompt = self.prompt_template.format(
            input_list_str=input_list_str,
            target_list_str=target_list_str,
            json_format=json_format
        )

        response = self.response(prompt)
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            matches = json.loads(cleaned_response)
            if not isinstance(matches, list):
                raise ValueError("Response is not a list")
            metrics = {
                "input_findings_count": len(input_findings_list),
                "target_findings_count": len(target_findings_list),
                "matched_count": len(matches)
            }
            return matches, metrics
        except json.JSONDecodeError:
            return [], {
                "input_findings_count": len(input_findings_list),
                "target_findings_count": len(target_findings_list),
                "matched_count": 0
            }

def extract_key_findings(text: str) -> str:
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    valid_findings = [m.strip() for m in matches if m.strip()]

    
    if valid_findings:
        for i in range(len(valid_findings)):
            valid_findings[i] = valid_findings[i].replace('- ', '')
        return valid_findings[-1] 
    
    print("Warning: No valid key findings found in input text")
    return ""

def calculate_findings_metric(input_text: str, merged_findings: List[str], model_name: str = 'deepseek-v3') -> Tuple[float, float, float]:
    try:
        input_findings = extract_key_findings(input_text)
        print('input_findings', input_findings)
        if not input_findings:
            print("Warning: No key findings found in input text")
            return 0.0, 0.0, 0.0
        calculator = FindingsRecallCalculator(model_name)
        matches, metrics = calculator.calculate_recall(input_findings, merged_findings)
        print(matches)
        if metrics['target_findings_count'] == 0 or metrics['input_findings_count'] == 0:
            return 0.0, 0.0, 0.0
        precision = len(set([m[0] for m in matches])) / metrics['input_findings_count']
        recall =    len(set([m[1] for m in matches])) / metrics['target_findings_count']
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    except:
        return 0.0, 0.0, 0.0

