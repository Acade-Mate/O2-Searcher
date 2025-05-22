from .rewards_format import format_reward_fn
from .rewards_score import f1_reward_fn, dv_reward_fn
from typing import Union, List, Dict, Any
import random

def o2searcher_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], queries = None):
    format_reward_output = format_reward_fn(solution_str)
    f1_reward = f1_reward_fn(solution_str, ground_truth['target'].tolist())
    queries = [q for q in queries.split('\n') if q.strip()]
    dv_reward = dv_reward_fn(queries)
    weights = [1, 1, 0.5]
    total_reward = 0.5*(weights[0]*format_reward_output.reward + weights[1]*f1_reward + weights[2]*dv_reward)/sum(weights)
    print(format_reward_output, f'f1: {f1_reward}', f'dv: {dv_reward}')
    return total_reward