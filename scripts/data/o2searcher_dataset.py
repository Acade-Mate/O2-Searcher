"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any
import json
import re
import numpy as np

import pandas as pd
import datasets


SYSTEM_PROMPT = "As a expert researcher, provide comprehensive key findings for open-ended queries and precise answers to other specific questions. Each time you receive new information, you MUST first engage in reasoning within the <think> and </think> tags. After reasoning, if you realize that you lack certain knowledge, you can invoke a SEARCH action with distinct queries (one to five) using the <search>\n<query>QUERY</query>\n<query>QUERY</query>\n</search> format to obtain relevant learnings, which will be presented between the <learnings> and </learnings> tags.\nYou are allowed to perform searches as many times as necessary. If you determine that no additional external knowledge is required, you can directly present the output within the <answer> and </answer> tags."
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepResearcher training')
    parser.add_argument('--local_dir', default='./o2searcher/data/',
                       help='Local directory to save processed datasets')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    local_dir = args.local_dir
    split = args.split

    def process_fn_qa(row):
        data = {
            "data_source": row['data_source'],
            "prompt": np.array([
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': 'Initial query: ' + row['question']}
                ], dtype=object),
            "ability": 'closedended',
            "reward_model": {
                "style": "rule",
                "ground_truth":  {'target': row['golden_answers']}
            },
            "extra_info": {
                'split': split,
                'index': row['extra_info']['index']
            }
        }
        return data

    def process_fn(query: str, gt: List[Dict], idx: int) -> Optional[Dict[str, Any]]:
        data = {
            "data_source": "openended",
            "prompt": np.array([
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'Initial query: {query}'}
                ], dtype=object),
            "ability": "openended",
            "reward_model": {
                "style": "rule",
                "ground_truth":  {'target': np.array(gt, dtype=object)}
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data

    # Initialize datasets
    dataframes = []
    if split == 'train':
        datanames = ['nq_hotpotqa'] + ['openended']*4
    else:
        datanames = ['nq_hotpotqa']
    for data_source in datanames:
        if data_source == 'nq_hotpotqa':
            parquet_file = os.path.join(args.local_dir, data_source, f'{split}.parquet')
            dataframe = pd.read_parquet(parquet_file)
            if split == 'train':
                dataframe = dataframe.sample(n=1200, random_state=42)
            dataframe = dataframe.apply(process_fn_qa, axis=1).apply(pd.Series)
        else:
            json_file = os.path.join(args.local_dir, data_source, f'{split}_gt.json')
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            processed_data = []
            for idx, (query, gt) in enumerate(data.items()):
                processed_example = process_fn(query, gt, idx)
                processed_data.append(processed_example)
            dataframe = pd.DataFrame(processed_data)

        print(f'{data_source}: {len(dataframe)}')
        dataframes.append(dataframe)
    dataframe = pd.concat(dataframes)
    print(f'original dataset len: {len(dataframe)}')
    dataframe.to_parquet(os.path.join(local_dir, 'hybrid', f'{split}.parquet'))

