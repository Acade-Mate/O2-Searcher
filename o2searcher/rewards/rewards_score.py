import requests
from typing import List
from o2searcher.rewards.config import *



def dv_reward_fn(queries: List[str], do_print=False) -> float:
    payload = {
        "queries": queries
    }
    try:
        output = requests.post(DV_URL, json=payload).json()
        if do_print: print(output)
        reward = output['overall_independence_score']
        return reward
    except Exception as e:
        print(f"[WARNING] Independence score error! {str(e)}")
        return 0.0

def f1_reward_fn(solution_str: str, ground_truth, do_print=False) -> float:
    payload = {
        "generated_text": solution_str,
        "reference_points": ground_truth,
        "threshold": 0.75
    }
    try:
        output = requests.post(F1_URL, json=payload).json()
        if do_print: print(output)
        reward = output['f1']
        return reward
    except Exception as e:
        print(f"[WARNING] F1 score error! {str(e)}")
        return 0.0
    

if __name__ == '__main__':
    test_queries = [
        "具体机制：较弱大模型通过数据增强和正则化技术提供有效知识的详细过程",
        "实际案例：DeepSeek的量化蒸馏技术的具体应用和效果评估",
        "蒸馏参数的优化策略：温度参数、损失函数和超参数调整的具体方法和效果"
    ]
    test_solution = '''<answer>and</answer><answer>and</answer><answer>- Multimodal learning enhances disease diagnosis and treatment planning by integrating diverse data types such as clinical notes, imaging studies, and laboratory results, leading to more comprehensive and accurate patient assessments.
        - Large language models can analyze and synthesize multimodal data, providing clinicians with deeper insights and more informed decision-making capabilities, thereby improving the accuracy of diagnoses and treatment recommendations.
        - Multimodal learning speeds up the drug discovery process by analyzing vast amounts of scientific literature and clinical trial data, identifying potential treatment candidates more efficiently than traditional methods, which is crucial in precision medicine.
        - Multimodal machine learning in precision health improves prediction accuracy and clinical decision-making by integrating diverse data sources such as imaging, electronic health records (EHR), and genomic data. This helps in detecting and diagnosing conditions like neurodegenerative diseases and cancers.
        - Early fusion techniques help in creating a comprehensive patient profile, improving the detection and diagnosis of conditions like neurodegenerative diseases and cancers.
        - Intermediate fusion involves step-by-step combination of different data features, allowing for more flexible and expressive model architectures, which is particularly useful in integrating EHR and text data, enhancing the ability to capture nuanced clinical information.
        - Late fusion trains separate models for each data source and combines their outputs, providing robustness and interpretability, especially in scenarios where different data types have varying levels of importance or reliability.
        </answer>
    '''
    ground_truth = ['Multimodal learning enhances precision medicine by integrating diverse data types, such as clinical text and imaging, to improve disease diagnosis and treatment planning, as evidenced by the innovative multimodal predictive models published in Nature.', 
        'The rise of medical multimodal large models, such as those developed by Baidu Intelligent Cloud, provides new opportunities for precision medicine by enabling more accurate diagnosis of complex diseases, including rare digestive tract conditions.', 
        'Multimodal AI systems have demonstrated significant breakthroughs in the precise diagnosis of diseases like dementia and lung infections, showcasing their potential in clinical decision-making and treatment strategy formulation.', 
        "Research teams, such as Professor Dong Bin's, are actively exploring the use of multimodal AI models to support medical clinical decisions, highlighting the practical applications of multimodal learning in precision medicine.", 
        'The integration of multimodal data in AI systems is leading to advancements in precision health, as seen in the successful early identification of stroke using multimodal deep learning models based on language and motion data.', 
        'Despite the promising applications, multimodal learning in precision medicine faces technical and computational challenges, which require further research and development to fully realize its potential in clinical settings.', 
        'The integration of multimodal data resources, as supported by national data initiatives, empowers precision medicine by providing comprehensive data for more informed decision-making in healthcare.', 
        'The use of multimodal data in personalized cancer care, as demonstrated by Philips, showcases the practical application of multimodal learning in creating tailored treatment plans.', 
        'Multimodal learning, by integrating various data sources such as imaging, genomics, and clinical data, can enhance the accuracy of disease diagnosis, as evidenced by breakthroughs in the precise diagnosis of dementia.', 
        'Multimodal AI has wide applications in clinical disease diagnosis, providing more comprehensive diagnostic information through the integrated analysis of different types of data.', 
        'Although multimodal large models show potential in healthcare, their specific applications in treatment planning are still limited and require further research and exploration.', 
        'Precision medicine emphasizes accurate diagnosis, and the application of multimodal technology can support this goal by improving diagnostic accuracy, particularly in cancer research.', 
        'The China Computer Federation points out that the deep integration of artificial intelligence and precision medicine helps to improve the accuracy and efficiency of diagnosis and treatment, highlighting the importance of multimodal learning in precision medicine.', 
        "Multimodal AI has important application prospects in precision immunotherapy, helping to end the 'trial and error' era of immunotherapy.", 
        'Multimodal AI is applied in the precise diagnosis and treatment of breast cancer, demonstrating its potential in cancer therapy.', 
        "Stanford's Li Ruijiang team used a multimodal AI model to accurately predict the effectiveness of immunotherapy, advancing tumor treatment."]

    import time
    start = time.time()
    dv = dv_reward_fn(test_queries, do_print=True)
    end = time.time()
    print(f'total time cost: {end - start} s, dv: {dv}')

    start = time.time()
    f1 = f1_reward_fn(test_solution, ground_truth, do_print=True)
    end = time.time()
    print(f'total time cost: {end - start} s, f1: {f1}')