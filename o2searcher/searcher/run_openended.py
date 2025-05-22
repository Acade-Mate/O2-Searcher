from fastapi import FastAPI, Request, HTTPException
from o2searcher.searcher.generator import GPTGenerator
from o2searcher.searcher.prompts import learnings_prompts, compress_prompts
import httpx
from dataclasses import dataclass
from typing import List, Optional, Dict
import asyncio
import json
import aiohttp
import logging
from bs4 import BeautifulSoup
import signal
from collections import OrderedDict
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local_url', type=str, default="http://127.0.0.1:10000/search")
parser.add_argument('--link_tag', type=str, default='href')
parser.add_argument('--model_name', type=str, default='qwen2.5')
parser.add_argument('--port', type=int, default=10102)
args = parser.parse_args()

# Constants
SEARCH_URL = args.local_url  # search endpoint for meilisearch
LINK_TAG = args.link_tag
DEFAULT_MODEL = args.model_name

MAX_CONCURRENT_REQUESTS = 64
MAX_CONCURRENT_CONTENT = 8
EXCLUDED_EXTENSIONS = ('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.gif')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
SEARCH_CN = False
RETURN_CACHE = False

# Initialize components
LLM = GPTGenerator()
CLIENT = httpx.AsyncClient()
app = FastAPI()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryBank:
    def __init__(self, max_size=4200000, json_file='./o2searcher/data/memory.json'):
        self.max_size = max_size
        self.json_file = json_file
        self.memory = OrderedDict()
        self.lock = asyncio.Lock()
        
        self._load_from_json()

    def _load_from_json(self):
        if not os.path.exists(self.json_file):
            return

        try:
            with open(self.json_file, encoding='utf-8') as f:
                data = json.load(f)
                filtered_data = {
                    k: v for k, v in data.items() 
                    if v is not None and v.get('result') not in (None, "")
                }
                
                items = list(filtered_data.items())[-self.max_size:]
                self.memory = OrderedDict(items)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load memory bank: {e}")
        except AttributeError:
            print(f"Warning: Invalid data format in {self.json_file}")
            self.memory = OrderedDict()

    def _save_to_json(self):
        filtered_memory = {
            k: v for k, v in self.memory.items() 
            if v is not None and v.get('result') not in (None, "")
        }
        
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_memory, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save memory bank: {e}")

    async def get(self, query: str) -> Optional[str]:
        # TODO: embedding similary for content retrival
        async with self.lock:
            entry = self.memory.get(query)
            if entry and entry.get('result') not in (None, ""):
                return entry['result']
            return None

    async def set(self, query: str, result: str):
        if result in (None, ""):
            return

        async with self.lock:
            if len(self.memory) >= self.max_size:
                self.memory.popitem(last=False)
            
            self.memory[query] = {'result': result}


MEMORY_BANK = MemoryBank()


async def compress_raw_content(query: str, contents: str, model_name: str) -> str:
    """Compress raw web content using GPT."""
    try:
        messages = [
            {'role': 'system', 'content': compress_prompts.system_prompt},
            {'role': 'user', 'content': compress_prompts.prompt_template.format(
                query=query, 
                contents=contents
            )}
        ]
        return await LLM.generate(messages, model_name)
    except Exception as e:
        logger.error(f"Error compressing content for query '{query}': {str(e)}")
        return ""


async def _translate_to_chineses(query, model_name) -> str:
    system_prompt = """Please translate the following English query into Chinese, requiring concise and professional language, ensuring accurate terminology and correct grammar, and without adding additional explanations."""
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'The english query: {query}'}
    ]
    result = await LLM.generate(messages, model_name)    

    return result.strip()


async def _translate_query_with_semaphore(semaphore, query, model_name):
    async with semaphore:
        return await _translate_to_chineses(query, model_name)


async def fetch_content(session: aiohttp.ClientSession, query: str, web: Dict[str, str], model_name: str) -> str:
    """
    Fetch content from the provided webs link asynchronously.
    Returns a dictionary with title, url and raw_content or None if failed.
    """
    raw_content = web.get('raw_content', None)
    if raw_content is not None:
        return raw_content
    
    link = web.get(LINK_TAG, "")
    
    if not link or not link.startswith("http"):
        logger.error(f"Invalid link: {link}")
        return None

    if any(link.lower().endswith(ext) for ext in EXCLUDED_EXTENSIONS):
        logger.warning(f"Excluded non-HTML resource: {link}")
        return None

    try:
        async with session.get(link, headers=HEADERS, timeout=20) as response:
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type:
                logger.warning(f"Skipping non-HTML content: {link} ({content_type})")
                return None
                
            html = await response.text()
            soup = BeautifulSoup(html, "lxml")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            body = soup.get_text(separator='\n', strip=True)
            logger.debug(f"Fetched content from {link}")

            compressed_content = await compress_raw_content(query, body[:32000], model_name)
            return compressed_content

    except asyncio.TimeoutError:
        logger.warning(f"Timeout when fetching content from {link}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch content from {link}: {str(e)}")
        return None


async def _extract(query: str, data: List[Dict[str, str]], num_contents: int, model_name: str) -> List[str]:
    """Extract and compress content from search results."""
    if not data:
        return []
        
    processed_data = []
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONTENT)
        
        async def limited_fetch(web):
            async with semaphore:
                return await fetch_content(session, query, web, model_name)
        
        tasks = [limited_fetch(web) for web in data[:num_contents] if web]
        results = await asyncio.gather(*tasks)
        
        processed_data = [res for res in results if res]
    
    return processed_data[:num_contents]

async def _learn(query: str, contents: str, model_name, num_learnings: int = 3) -> List[str]:
    """Generate learnings from compressed content."""
    try:
        messages = [
            {'role': 'system', 'content': learnings_prompts.system_prompt},
            {'role': 'user', 'content': learnings_prompts.prompt_template.format(
                query=query, 
                contents=contents, 
                num_learnings=num_learnings
            )}
        ]
        response = await LLM.generate(messages, model_name)
        return response
    except Exception as e:
        logger.error(f"Error generating learnings for query '{query}': {str(e)}")
        return ''

async def _process_query(query: str, data: List[Dict[str, str]], num_contents: int, model_name: str) -> List[str]:
    """Process a single query end-to-end."""
    if RETURN_CACHE:
        cached_result = await MEMORY_BANK.get(query)
        if cached_result is not None:
            logger.info(f"Returning cached result for query: {query}")
            return cached_result
    
    compressed_contents = await _extract(query, data, num_contents, model_name)
    if not compressed_contents:
        return ''
        
    contents_str = '\n'.join(compressed_contents).replace('{', '').replace('}', '')
    learnings_str = await _learn(query, contents_str[:32000], model_name)

    await MEMORY_BANK.set(query, learnings_str)

    logger.info(f"Processed query: {query}, Learnings: {learnings_str}")
    return learnings_str


async def _process_query_with_semaphore(semaphore, query, data, num_contents, model_name):
    async with semaphore:
        return await _process_query(query, data, num_contents, model_name)
    

@app.post("/search")
async def search(request: Request):
    """
    Endpoint to process multiple search queries and return learnings.
    """
    try:
        payload = await request.json()
        all_queries = payload['queries']
        num_contents = payload.get('topk', 3)
        model_name = payload.get('model_name', DEFAULT_MODEL)

        # Validate input
        if not all_queries or not isinstance(all_queries, list):
            raise ValueError("Invalid queries format")
            
        query_list = [query for queries in all_queries for query in queries]
        if SEARCH_CN:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            tasks = [
                _translate_query_with_semaphore(semaphore, query, model_name) 
                for query in query_list
            ]
            new_query_list = await asyncio.gather(*tasks)
        else:
            new_query_list = query_list

        # Fetch search results
        response = await CLIENT.post(
            SEARCH_URL, 
            json={'queries': new_query_list},
            timeout=120
        )
        response.raise_for_status()
        data_list = response.json()

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        # Process all queries in parallel
        tasks = [
            _process_query_with_semaphore(semaphore, query, data, num_contents, model_name) 
            for query, data in zip(query_list, data_list)
        ]
        learnings = await asyncio.gather(*tasks)

        # Group results by original query batches
        results = []
        start_idx = 0
        for queries in all_queries:
            batch_learnings = learnings[start_idx:start_idx + len(queries)]
            learnings_str = '\n'.join(batch_learnings)
            results.append(learnings_str)
            start_idx += len(queries)

        return results
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Search API error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return None


def handle_sigint(signum, frame):
    """Handle SIGINT signal (Ctrl+C)"""
    logger.info("Received interrupt signal, saving memory bank...")
    MEMORY_BANK._save_to_json()
    # Exit the program
    os._exit(0)


if __name__ == "__main__":
    import uvicorn

    # Register signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    uvicorn.run(app, host="0.0.0.0", port=args.port, limit_concurrency=128)