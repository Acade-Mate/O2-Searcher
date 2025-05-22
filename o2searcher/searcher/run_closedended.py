import httpx
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import uvicorn
import os
import asyncio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=10001)
parser.add_argument('--local_url', type=str, default="http://127.0.0.1:8000/retrieve")

args = parser.parse_args()


# --- Configuration ---
SERVICE_PORT = args.port
DENSE_RETRIEVER_URL = args.local_url
DEFAULT_TOP_K = 3
# --- Configuration End ---

class WikiSearchRequest(BaseModel):
    queries: List[List[str]]
    topk: int = DEFAULT_TOP_K

app = FastAPI(title="Wiki Search Service Wrapper")

@app.on_event("startup")
async def startup_event():
    app.state.http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.http_client.aclose()

@app.post("/wiki_search", response_model=List[str])
async def wiki_search(request: WikiSearchRequest):
    client: httpx.AsyncClient = app.state.http_client

    if not request.queries or not isinstance(request.queries, list) or not all(isinstance(q_list, list) for q_list in request.queries):
        raise HTTPException(status_code=400, detail="Invalid queries format. Expected List[List[str]].")

    if request.topk <= 0:
        raise HTTPException(status_code=400, detail="Invalid topk value. Must be positive.")

    flat_queries = [query for batch in request.queries for query in batch]

    if not flat_queries:
        return ["" for _ in request.queries] 

    payload = {
        "queries": flat_queries,
        "topk": request.topk,
        "return_scores": True
    }

    try:
        response = await client.post(DENSE_RETRIEVER_URL, json=payload, timeout=60.0)
        response.raise_for_status()
        dense_results_data = response.json()

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Error connecting to dense retrieval service: {exc}"
        )
    except httpx.HTTPStatusError as exc:
        detail = f"Dense retrieval service returned error: {exc.response.status_code}"
        try:
            detail = exc.response.json()
        except:
            pass
        raise HTTPException(
            status_code=exc.response.status_code if exc.response.status_code >= 400 else 500,
            detail=detail
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    if "result" not in dense_results_data or not isinstance(dense_results_data["result"], list):
        raise HTTPException(status_code=500, detail="Unexpected response format from dense retrieval service.")

    retrieved_query_results = dense_results_data["result"]

    if len(retrieved_query_results) != len(flat_queries):
        print(f"Warning: Mismatch between query count ({len(flat_queries)}) and result count ({len(retrieved_query_results)}).")
        while len(retrieved_query_results) < len(flat_queries):
            retrieved_query_results.append([])

    final_results = []
    start_idx = 0

    for batch in request.queries:
        batch_len = len(batch)
        batch_results = retrieved_query_results[start_idx : start_idx + batch_len] if batch_len > 0 else []

        batch_content = []
        for query_result_list in batch_results:
            query_contents = []
            for doc in query_result_list[:request.topk]:
                content = ""
                if isinstance(doc, dict):
                    document = doc.get('document', {})
                    if isinstance(document, dict):
                        content = document.get('contents', '')
                        print(f"DEBUG: content: {content}")
                    else:
                        print(f"DEBUG: 'document' field is not a dict: {document}")
                else:
                    print(f"DEBUG: doc was not a dict: {doc}")
                query_contents.append(content)

            while len(query_contents) < request.topk:
                query_contents.append(" ")

            query_str = " ".join(query_contents)
            batch_content.append(query_str)

        batch_str = "\n".join(batch_content)
        final_results.append(batch_str)
        start_idx += batch_len

    return final_results

if __name__ == "__main__":
    print(f"Starting Wiki Search Service on http://0.0.0.0:{SERVICE_PORT}")
    print(f"Proxying to Dense Retriever at: {DENSE_RETRIEVER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)