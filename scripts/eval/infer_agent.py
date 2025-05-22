import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from o2searcher.prompts import error_prompt, extra_prompt
import requests

import time
import logging

import argparse
import re
import json
import tqdm


SYSTEM_PROMPT = "As a expert researcher, provide comprehensive key findings for open-ended queries and precise answers to other specific questions. \
    Each time you receive new information, you MUST first engage in reasoning within the <think> and </think> tags. \
    After reasoning, if you realize that you lack certain knowledge, you can invoke a SEARCH action with distinct queries (one to five) using the <search>\n<query>QUERY</query>\n<query>QUERY</query>\n</search> format to obtain relevant learnings, \
    which will be presented between the <learnings> and </learnings> tags.\n \
    You are allowed to perform searches as many times as necessary. If you determine that no additional external knowledge is required, you can directly present the output within the <answer> and </answer> tags."

target_sequences = ["</search>\n", "</search>", "</search>\n\n", "</answer>", "</answer>\n", "</answer>\n\n"]


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_ids.append([1472,1836])
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    return model, tokenizer, stopping_criteria


class ResearchAgent:
    def __init__(self, model_path, search_urls=None):
        self.max_step = 5
        self.num_contents = 3
        self.model, self.tokenizer, self.stopping_criteria = load_model(model_path)
        self.search_urls = search_urls
        self.max_new_tokens = 2048
        self.return_raw = False
        self.raw_contents = []
    
    def extract_label_content(self, response, label='search'):
        if label == 'search':
            queries = re.findall(r'<query>(.*?)</query>', response, re.DOTALL)
            queries = [p.strip() for p in queries] if queries else []
            success = len(queries) > 0 
            return success, queries
        else:
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            return True if match else False

    def check_status(self, response):
        success, queries = self.extract_label_content(response, label='search')
        done = self.extract_label_content(response, label='answer')
        if not done:
            logging.info(f'SEARCH: {queries}')
        return success, queries, done

    def search(self, queries, ability):
        payload = {
            "queries": [queries], # [[list of queries] * num]
            "topk": self.num_contents,
        }
        
        if not self.return_raw or ability != 'openended':
            learning_str = requests.post(self.search_urls[ability], json=payload).json()[0]
        else:
            payload['return_raw'] = True
            result = requests.post(self.search_urls[ability], json=payload).json()
            learning_str = result['learnings'][0]
            self.raw_contents.extend(result['raw_contents'])
        extra_info = extra_prompt.format(learning_str=learning_str)
        return extra_info
    
    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

        inputs = self.tokenizer([text], return_tensors='pt').to('cuda')
        generated_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria,
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if '<search>' in response and '</search>' not in response: response += '>'
        return response
    
    def step(self, messages: list, ability: str):
        response = self.generate(messages)
        messages.append({'role': 'assistant', 'content': response})
        print(response)

        success, queries, done = self.check_status(response)

        if not done:
            if not success:
                messages.append({'role': 'user', 'content': error_prompt})
            else:
                extra_info = self.search(queries, ability)
                messages.append({'role': 'user', 'content': extra_info})

        return done, messages
    
    def forward(self, query, ability='openended'):
        logging.info(f'Initial query: {query}')
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Initial query: '+ query}
        ]
        done = False
        for _ in range(self.max_step):
            done, messages = self.step(messages, ability)
            if done:
                break

        return messages
    

def save_messages(messages: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))
    logging.info(f"Writed messages: {path}")


def main(args):
    researcher = ResearchAgent(args.model_path, args.search_urls)

    if os.path.exists(args.query):
        with open(args.query, 'r', encoding='utf-8') as f:
            queries = f.readlines()

        for query in tqdm.tqdm(queries):
            
            messages = researcher.forward(query, args.ability)
            save_name = query.replace(' ', '_').replace(',', '_').replace('?', '')  
            save_messages(messages, f'outputs/{args.tag}/qwen2.5-3b/{save_name[:100]}.json')
        
        return

    researcher.return_raw = True
    messages = researcher.forward(args.query, args.ability)    
    save_messages(messages, f'outputs/{args.tag}/messages_{current_time}.json')
    save_messages(researcher.raw_contents, f'outputs/{args.tag}/raw_contents_{current_time}.json')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', type=str, default="")
    parser.add_argument('-m', '--model_path', type=str, default='')
    parser.add_argument('-a', '--ability', type=str, default='openended')
    parser.add_argument('-t', '--tag', type=str, default='150step')
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'config.json')) as f:
        search_urls = json.load(f)
    args.search_urls = search_urls

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  - %(message)s', handlers=[logging.StreamHandler()])

    main(args)
