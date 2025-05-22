
from fastapi import FastAPI, Request
import asyncio
from o2searcher.searcher.generator import GPTGenerator
from o2searcher.searcher.prompts import learnings_prompts, compress_prompts
from tavily import AsyncTavilyClient
from dataclasses import dataclass
from typing import List
import json
import os


@dataclass
class SearchResult:
    refs: List[str]
    contents: List[str]


class Searcher:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, 'api_keys.json')) as f:
            self.api_keys = json.load(f)
        self.current_key_index = 0
        self.searcher = AsyncTavilyClient(api_key=self.api_keys[self.current_key_index])
        self.agent = GPTGenerator()
        self.max_retries = len(self.api_keys)

    async def compress_raw_content(self, query, contents):
        print(f"Compressing the raw web contents...")

        messages = [
            {'role': 'system', 'content': compress_prompts.system_prompt},
            {'role': 'user','content': compress_prompts.prompt_template.format(query=query, contents=contents)}
        ]
        compressed_contents = await self.agent.generate(messages)

        return compressed_contents

    async def search(self, query: str, num_contents: int = 3, compress_raw_content: bool = True) -> SearchResult:
        print(f"Using web tool to retrieving contents...")
        
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                response = await self.searcher.search(
                    query, 
                    include_raw_content=True, 
                    max_results=num_contents, 
                    include_answer=False
                )
                
                if compress_raw_content:
                    compression_tasks = [
                        self.compress_raw_content(query, res['raw_content'])
                        for res in response['results']
                    ]
                    contents = await asyncio.gather(*compression_tasks)
                else:
                    contents = [res['raw_content'] for res in response['results']]
                
                return SearchResult(
                    refs=[x['url'] for x in response['results']],
                    contents=[
                        f"{res['title']}: {content}" 
                        for res, content in zip(response['results'], contents)
                    ]
                )
            except Exception as e:
                last_exception = e
                retry_count += 1
                print(f"Search failed with API key {self.current_key_index + 1}/{len(self.api_keys)}. Error: {str(e)}")
                
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                self.searcher = AsyncTavilyClient(api_key=self.api_keys[self.current_key_index])
                print(f"Switching to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        
        raise Exception(f"All API keys failed. Last error: {str(last_exception)}")


SEARCHER = Searcher()
LLM = GPTGenerator()
app = FastAPI()

MAX_CONCURRENT = 128

async def learn(query, contents: str, num_learnings=3) -> list:
        messages = [
            {'role': 'system', 'content': learnings_prompts.system_prompt},
            {'role': 'user', 'content': learnings_prompts.prompt_template.format(query=query, contents=contents, num_learnings=num_learnings)}
        ]
        return await LLM.generate(messages)

@app.post("/search")
async def search(requests: Request):
    payload = await requests.json()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return_raw = payload.get('return_raw', False)
    raw_contents = []

    async def _process_query(query, num_contents):
        async with semaphore:
            result = await SEARCHER.search(query, num_contents=num_contents, compress_raw_content=True)
            if return_raw:
                raw_contents.append(result)
            contents_str = ''.join([f'\n{content}\n' for content in result.contents]).replace('{', '').replace('}', '')
            learnings = await learn(query, contents_str)
            print(f"Query: {query}; Learnings: {learnings}")
            return learnings
    
    all_queries = payload['queries']
    num_contents = payload.get('topk', 3)    
    tasks = [_process_query(query, num_contents) for queries in all_queries for query in queries]
    learnings = await asyncio.gather(*tasks)

    results = []
    start_idx = 0
    for queries in all_queries:
        learnings_str = '\n'.join(learnings[start_idx:start_idx + len(queries)])
        results.append(learnings_str)
        start_idx += len(queries)

    if return_raw:
        return {'learnings':results, 'raw_contents': raw_contents}
    else:
        return results


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=10102)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

