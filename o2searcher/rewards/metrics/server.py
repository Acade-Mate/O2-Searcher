from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import uvicorn
import numpy as np
from o2searcher.rewards.metrics.f1_reward import FindingSentenceEvaluator
from o2searcher.rewards.metrics.dv_reward import QueryIndependenceTransformer
from o2searcher.rewards.metrics.LFS import calculate_findings_metric
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=11000)
args = parser.parse_args()

app = FastAPI(title="Metrics")


model_name = 'doubao-32k'
calculator = QueryIndependenceTransformer(model_name)
evaluator = FindingSentenceEvaluator(model_name)

class FindingRequest(BaseModel):
    generated_text: str
    reference_points: List[str]
    threshold: float = 0.85

class BatchFindingRequest(BaseModel):
    generated_texts: List[str]
    reference_points: List[str]
    threshold: float = 0.85

class QueryRequest(BaseModel):
    queries: List[str]

@app.post("/calculate_finding_scores")
async def calculate_scores(request: FindingRequest) -> Dict[str, Any]:
    return evaluator.calculate_scores(
        request.generated_text,
        request.reference_points,
        request.threshold
    )

@app.post("/batch_calculate_finding_scores")
async def batch_calculate_scores(request: BatchFindingRequest) -> List[Dict[str, Any]]:
    results = []
    for generated_text in request.generated_texts:
        scores = evaluator.calculate_scores(
            generated_text,
            request.reference_points,
            request.threshold
        )
        results.append(scores)
    return results

@app.post("/calculate_query_independence")
async def calculate_independence(request: QueryRequest) -> Dict[str, Any]:
    """calculate dv reward"""
    return calculator.calculate_independence(request.queries)


class DataModel(BaseModel):
    generated_text: str
    reference_points: List[str]

@app.post("/calculate_finding_scores_llm")
async def calculate_finding_scores(data: DataModel):
    try:
        precision, recall, f1 = calculate_findings_metric(
            data.generated_text,
            data.reference_points,
            model_name=model_name
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=args.port, reload=False)
