import os
import json
import re
import requests
import numpy as np
from tqdm import tqdm
import glob
import ast
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--generate_dir', type=str, default="./outputs/150step")
parser.add_argument('--output_dir', type=str, default="./outputs/")
parser.add_argument('--metric_port', type=int, default=11000)
parser.add_argument('--type', type=str, default='f1')
args = parser.parse_args()

# 文件路径
OUTPUT_DIR = args.output_dir
LLM_DIR = args.generate_dir
if args.type == 'f1':
    EVALUATION_URL = f"http://127.0.0.1:{args.metric_port}/calculate_finding_scores"
elif args.type == 'lfs':
    EVALUATION_URL = f"http://127.0.0.1:{args.metric_port}/calculate_finding_scores_llm"
else:
    raise NotImplementedError

GT_FILE = "./o2searcher/data/openended/test_gt.json"
DIF_FILE = "./o2searcher/data/openended/difficulty_labels.json"
models = ["qwen2.5-3b"]


def load_gt_data():
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_label_data():
    with open(DIF_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_from_conversation_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        conversation = json.load(f)
    
    question = None
    for message in conversation:
        if message.get("role") == "user" and "Initial query:" in message.get("content", ""):
            content = message.get("content", "")
            question = content.split("Initial query:", 1)[1].strip()
            question = question.replace('(open-ended)', '').strip()
            break
    
    answer = None
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            pattern = r'<(?:answer|key_findings)>(.*?)</(?:answer|key_findings)>'
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                answer = f"<answer>{matches[0]}</answer>"
                break
    return question, answer

def format_answer(answer_text):
    if not answer_text:
        return None
    
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, answer_text, re.DOTALL)
    
    if not matches:
        return None
    
    answer_content = matches[0].strip()
    
    if answer_content.startswith('[') and answer_content.endswith(']'):
        try:
            items = ast.literal_eval(answer_content)
            formatted_items = []
            for item in items:
                clean_item = item.strip().strip('"').strip()
                if clean_item.startswith("Key finding"):
                    clean_item = clean_item.split(':')[1].strip()
                    formatted_items.append(f"- {clean_item}")
                else:
                    if item and isinstance(item, str) and item.strip():
                        formatted_items.append(f"- {item.strip()}")
        except:
            pattern = r'["\']\s*(?:Key finding \d+: )?(.*?)[\'"]\s*,?'
            items = re.findall(pattern, answer_content)
            formatted_items = [f"- {item.strip()}" for item in items if item.strip()]

            if not formatted_items:
                lines = answer_content.split('\n')
                formatted_items = []
                for line in lines:
                    line = line.strip()
                    if line and line not in ['[', ']', '-']:
                        line = line.strip(',').strip('"').strip("'").strip('-').strip()
                        if line:
                            formatted_items.append(f"- {line}")
    else:
        lines = answer_content.split('\n')
        formatted_items = []
        for line in lines:
            line = line.strip()
            if line and line not in ['-', '[', ']']:
                if line.startswith('-'):
                    line = line[1:].strip()
                formatted_items.append(f"- {line}")
    
    formatted_answer = "\n        ".join(formatted_items)
    return f"""<answer> </answer> <answer>{formatted_answer}
    </answer>"""

def evaluate_answer(generated_text, reference_points, threshold=0.85):
    data = {
        "generated_text": generated_text,
        "reference_points": reference_points,
        "threshold": 0.75
    }
    
    try:
        response = requests.post(EVALUATION_URL, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"评估API返回错误: {response.status_code}")
            return None
    except Exception as e:
        print(f"调用评估API时出错: {e}")
        return None

def main():
    gt_data = load_gt_data()
    print(f"成功加载参考答案，共有 {len(gt_data)} 个问题")
    label_data = load_label_data()
    
    results = []
    
    used_gt_questions = set()
    
    json_files = []
    
    for model in models:
        model_dir = os.path.join(LLM_DIR, model)
        if os.path.exists(model_dir):
            model_files = glob.glob(os.path.join(model_dir, "*.json"))
            json_files.extend(model_files)
    
    print(f"发现 {len(json_files)} 个LLM结果文件")
    
    for file_path in tqdm(json_files):
        try:
            question, answer = extract_from_conversation_json(file_path)
            
            if not question or not answer:
                print(f"文件 {file_path} 缺少问题或答案")
                continue
            
            formatted_answer = format_answer(answer)
            if not formatted_answer:
                print(f"文件 {file_path} 无法提取答案内容")
                continue
            
            matched_gt_question = None
            reference_points = gt_data.get(question, [])
            if reference_points:
                matched_gt_question = question
            else:
                normalized_q = re.sub(r'[^\w\s]', '', question).lower()
                for gt_q, gt_refs in gt_data.items():
                    normalized_gt_q = re.sub(r'[^\w\s]', '', gt_q).lower()
                    if normalized_q in normalized_gt_q or normalized_gt_q in normalized_q:
                        reference_points = gt_refs
                        matched_gt_question = gt_q
                        break
            
            if not reference_points:
                print(f"问题 '{question}' 没有找到匹配的参考答案")
                continue
            
            if matched_gt_question:
                used_gt_questions.add(matched_gt_question)
            
            model_name = os.path.basename(os.path.dirname(file_path))
            print('formatted_answer: ', formatted_answer)
            
            evaluation_result = evaluate_answer(formatted_answer, reference_points)
            if evaluation_result:
                file_name = os.path.basename(file_path)
                result = {
                    "file": file_name,
                    "model": model_name,
                    "question": question,
                    "matched_gt": matched_gt_question,
                    "metrics": evaluation_result,
                    "mode": label_data[question]
                }
                results.append(result)
                print(f"{model_name} - {file_name} 评估完成，F1分数: {evaluation_result.get('f1', 'N/A')}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    unused_gt_questions = set(gt_data.keys()) - used_gt_questions
    print(f"\n有 {len(unused_gt_questions)} 个GT问题未被匹配:")
    for i, q in enumerate(unused_gt_questions, 1):
        print(f"{i}. {q}")
    
    with open(os.path.join(OUTPUT_DIR, "unused_gt_questions.json"), 'w', encoding='utf-8') as f:
        json.dump(list(unused_gt_questions), f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "llm_evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if results:
        model_scores = {}
        model_hard_scores = {}
        model_easy_scores = {}
        for result in results:
            model = result.get("model", "unknown")
            score = result.get("metrics", {}).get("f1", 0)
            if model not in model_scores:
                model_scores[model] = []
                model_hard_scores[model] = []
                model_easy_scores[model] = []
            model_scores[model].append(score)
            if result['mode'] == 'hard':
                model_hard_scores[model].append(score)
            if result['mode'] == 'easy':
                model_easy_scores[model].append(score)
        
        print("\n各模型平均F1分数:")
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"{model}: {avg_score:.4f}; Hard: {sum(model_hard_scores[model])/(len(model_hard_scores[model])+1e-8)}; Easy: {sum(model_easy_scores[model])/(len(model_easy_scores[model])+1e-8)} (样本数: {len(scores)})")
        
        all_scores = [r.get("metrics", {}).get("f1", 0) for r in results if "metrics" in r]
        if all_scores:
            overall_avg = sum(all_scores) / len(all_scores)
            print(f"\n总体平均F1分数: {overall_avg:.4f}, 共评估 {len(all_scores)} 个结果")
        
        print(f"参考答案总数: {len(gt_data)}, 已使用: {len(used_gt_questions)}, 未使用: {len(unused_gt_questions)}")
    else:
        print("\n没有评估结果")

if __name__ == "__main__":
    main()