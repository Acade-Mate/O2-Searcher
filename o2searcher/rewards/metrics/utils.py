import re
import json
import httpx
import openai
from typing import List, Dict


def get_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as rf:
        text = rf.read()

    return text


def write_results2file(res: str, save_to: str):
    with open(save_to, 'w', encoding='utf-8') as wf:
        wf.write(res)


def get_config(model: str):
    with open('./o2searcher/config.json', 'r', encoding='utf-8') as rf:
        api_configs: Dict[str, Dict] = json.load(rf)

    model_name = api_configs[model]['model_name']
    api_key = api_configs[model]['api_key_var']
    base_url = api_configs[model]['base_url']
    proxy_url = api_configs[model].get('proxy_url', None)

    return model_name, api_key, base_url, proxy_url


def extract_json_to_dict(text):
    json_pattern = r'```(?i:json)\s*(.*?)\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            result_dict = json.loads(json_str)
            return result_dict
        except json.JSONDecodeError as e:
            print(json_str)
            print(f'Json decode error: {text}')
            return {"error": f"Json decode error: {str(e)}"}
    else:
        print(text)
        return {"error": "Can not find json data"}
    

def extract_points(text: str) -> List[str]:
    try:
        matches = re.findall(r'\d+\.\s+(.*?)\s*(?=\d+\.|$)', text, re.DOTALL)
    except TypeError as e:
        print(text)
        print(type(text))
        raise e
    return [match.strip() for match in matches]


def extract_key_findings_from_response(text: str) -> List[str]:
    text_pattern = r'```(?i:TEXT)\s*(.*?)\s*```'
    match = re.search(text_pattern, text, re.DOTALL)
    if match:
        key_findings = match.group(1)
        return key_findings.split('\n')
    else:
        return 
    

def extract_key_findings_from_answer(text: str) -> str:
    match = re.search(r'<key_findings>(.*?)</key_findings>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return
    pass


def get_points(outline: Dict[str, Dict[str, str]]) -> List[str]:
    points = []
    for sec in outline.values():
        for subsec in sec.values():
            points.extend(extract_points(subsec))
    
    return points


def get_points_num(outline: Dict[str, Dict[str, str]]) -> int:
    points_num = 0
    for sec in outline.values():
        for subsec in sec.values():
            points_num += len(extract_points(subsec))
    
    return points_num


def extract_learnings(full_response: List[Dict[str, str]]):
    learnings: List[str] = []
    for elem in full_response:
        content = elem['content']
        match = re.search(r'<learnings>(.*?)</learnings>', content, re.DOTALL)
        if match:
            learnings.append(match.group(1).strip())

    return learnings


def extract_answer(full_response: List[Dict[str, str]]):
    final_response = full_response[-1]
    content = final_response['content']
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return


class AbstractAgent:
    def __init__(self, model_name):
        model_name, api_key, base_url, proxy_url = get_config(model_name)

        if proxy_url:
            self.client = openai.OpenAI(
                api_key=api_key,
                # base_url=base_url,
                http_client=httpx.Client(proxy=proxy_url)
            )
        else:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )

        self.model_name = model_name


    def _set_system_prompt_from_text(self, file_path: str):
        self.system_prompt = get_text(file_path)


    def response(self, text: str, temperature=0, top_p=1):
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        except AttributeError:
            messages = [
                {
                    "role": "user",
                    "content": text
                }
            ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )
            return res.choices[0].message.content
        except Exception as e:
            raise(e)