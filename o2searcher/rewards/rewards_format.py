from dataclasses import dataclass, field
from o2searcher import prompts
import numpy as np
import json
from typing import Union, List, Dict, Any
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FormatOutput:
    reward: float = 0.0
    metrics: dict = None


def extract_answer(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return None
    
    return matches[-1].group(1).strip()

def calculate_diversity_reward(content_list, similarity_threshold=0.6, top_k=1):
    if len(content_list) <= 1:
        return 1.0  # Maximum reward for single/empty content
    
    try:
        # Vectorize content using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(content_list)
        
        # Calculate pairwise cosine similarities
        pairwise_similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
        np.fill_diagonal(pairwise_similarity, 0)  # Zero out self-similarities
        
        # Get upper triangle values (excluding diagonal)
        upper_triangle = pairwise_similarity[np.triu_indices_from(pairwise_similarity, k=1)]
        
        if len(upper_triangle) == 0:
            return 1.0
        
        # New similarity score calculation that better handles few high similarities
        # 1. Count of highly similar pairs (above threshold)
        high_sim_count = np.sum(upper_triangle > similarity_threshold)
        
        # 2. Mean of top-k similarities (handles small clusters)
        top_k_sim = np.mean(np.sort(upper_triangle)[-top_k:]) if len(upper_triangle) >= top_k else np.max(upper_triangle)
        
        # 3. Overall mean similarity
        mean_sim = np.mean(upper_triangle)
        
        # Combine metrics with weights (adjust these based on your needs)
        similarity_score = (
            0.5 * top_k_sim +          # Emphasize cluster similarity
            0.3 * (2 * high_sim_count / len(upper_triangle)) +  # Ratio of high similarities
            0.2 * mean_sim             # Overall similarity
        )
        
        # Transform to diversity reward (higher score = less diversity)
        # Using a non-linear transformation to penalize high similarities more
        diversity_reward = max(0, 1 - np.power(similarity_score, 1.5))
        
        return diversity_reward
        
    except Exception as e:
        print(f"Error calculating diversity reward: {str(e)}")
        return 0.5  # Neutral fallback

def calculate_format_reward(model_answer):
    if not model_answer.strip() or model_answer.strip().lower() == 'and':
        return FormatOutput(reward=0.0)
    
    lines = [line.strip() for line in model_answer.split('\n') if line.strip()]
    if not lines:
        return FormatOutput(reward=0.0)

    valid_bullets = 0
    content_list = []
    format_errors = 0
    
    for line in lines:
        if line.startswith('- '):
            content = line[2:].strip()
            if content:
                valid_bullets += 1
                content_list.append(content)
            else:
                format_errors += 1
        else:
            format_errors += 1
    
    format_reward = 1 - (format_errors / len(lines))
    completeness_reward = min(valid_bullets / 10, 1)  # 10条得满分
    diversity_reward = calculate_diversity_reward(content_list)
    
    unique_ratio = len(set(content_list)) / max(1, len(content_list))
    duplicate_penalty = 1 - unique_ratio
    
    weights = [0.5, 0.3, 0.5]
    reward = (
        (weights[0] * format_reward +
        weights[1] * completeness_reward +
        weights[2] * diversity_reward) / sum(weights) - 
        3*duplicate_penalty
    )
    
    return FormatOutput(
        reward=max(0, min(1, reward)),
        metrics={
            'format': weights[0] * format_reward,
            'completeness': weights[1] * completeness_reward,
            'diversity': weights[2] * diversity_reward,
            'duplicate_penalty': -3*duplicate_penalty
        }
    )

def format_reward_fn(solution_str: str):
    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return FormatOutput(reward=0.0)

    format_reward_output = calculate_format_reward(model_answer)
    return format_reward_output


if __name__ == '__main__':
    test_sample = '- ' + '\n- '.join([\
        # "Social media has a double-edged sword effect on adolescent mental health, providing support but also potentially causing negative impacts.",
        "Social media has a double-edged sword effect on adolescent mental health, providing support but also potentially causing negative impacts.",
        "Frequent use of social media may lead to insomnia in adolescents, becoming an accomplice to sleeplessness.",
        "Social media may spread anxiety, affecting adolescents' social anxiety conditions.",
        "The impact of social media on adolescent mental health may be similar to internet addiction, posing potential harm.",
        "The use of social media may pose risks to girls' mental health, possibly related to cyberbullying and lack of sleep.",
        "There is harmful content on social media that needs to be regulated to reduce its negative impact on adolescents.",
        "Research shows that teenagers who use social media for more than 3 hours a day are more likely to experience depressive symptoms, with this risk potentially doubling.",
        # "Reuters reports that the use of social media may severely damage adolescent mental health.",
        "The Ministry of Education warns that social media may have a negative impact on adolescent mental health and needs attention.",
        "Excessive use of social networks may lead to social disorders in adolescents.",
        "Mayo Clinic and other institutions focus on the impact of social media use on adolescents, emphasizing the need to pay attention to its potential risks to mental health.",
        "Current research indicates that the use of social media may have a negative impact on adolescent mental health, but the specific long-term risks are not yet clear.",
        "Hans Publishers and related research point out that there is a certain correlation between the use of social networking sites and adolescent mental health, but more specific cases and detailed information are needed to support these findings.",
        "Despite research progress, there is currently a lack of specific research content on the differences in long-term potential risks of different social media platforms on adolescent mental health.",
        "New York State Attorney General James and other officials advocate for legislation to protect children in response to the long-term mental health risks of social media on adolescents.",
        "Attorneys General from 33 U.S. states are suing Meta, claiming its products harm children's mental health."
    ])
    print(calculate_format_reward(test_sample))