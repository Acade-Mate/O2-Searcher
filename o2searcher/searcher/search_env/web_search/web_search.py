import meilisearch
from fastapi import FastAPI
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI()

class SearchRequest(BaseModel):
    queries: List[str]

class SearchResponse(BaseModel):
    raw_content: str

client = meilisearch.Client('http://localhost:7700', 'Web_Knowledge_Corpus')
index = client.index('Web_Corpus')


@app.post("/search")
async def search(request: SearchRequest):
    all_results = []
    
    for query in request.queries:
        search_results = index.search(query, {'limit': 3}) 
        
        query_results = []
        if search_results['hits']:
            for hit in search_results['hits']:
                query_results.append(SearchResponse(
                    raw_content=hit.get('compressed_contents', '') 
                ))
            
    
        while len(query_results) < 3:
            query_results.append(SearchResponse(
                raw_content=""
            ))
            
        all_results.append(query_results)
    
    return all_results
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
