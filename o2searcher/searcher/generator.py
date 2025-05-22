import asyncio
import httpx
import json
from openai import AsyncOpenAI


class GPTGenerator: 
    def __init__(self, max_completion_token: int = 12288, max_try_times: int = 5):
        with open('./o2searcher/config.json', 'r', encoding='utf-8') as rf:
            self.api_configs = json.load(rf)

        self.max_completion_token = max_completion_token
        self.max_try_times = max_try_times

        self.acc_p_tk = 0
        self.acc_c_tk = 0

    def create_model(self, model: str = 'Qwen/Qwen2.5-72B-Instruct'):
        self.model = self.api_configs[model]['model_name']
        api_key = self.api_configs[model]['api_key_var']
        base_url = self.api_configs[model]['base_url']
        proxy_url = self.api_configs[model].get('proxy_url', None)

        if proxy_url is not None:
            self.client = AsyncOpenAI(
                api_key=api_key,
                http_client=httpx.AsyncClient(proxy=proxy_url)
            )
        else:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key
            )
        
    async def calculate_tokens(self, response) -> None:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        self.acc_p_tk += prompt_tokens
        self.acc_c_tk += completion_tokens
        
        print(f"Prompt tokens: {prompt_tokens}/{self.acc_p_tk}, Completion tokens: {completion_tokens}/{self.acc_c_tk}")

    async def generate(self, messages: list, model: str = 'doubao-32k') -> str:
        self.create_model(model)
        try_times = 0
        while True:
            if try_times == self.max_try_times:
                return ''
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_completion_token
                )
                
                await self.calculate_tokens(response)
                content = response.choices[0].message.content
                return content
                
            except Exception as e:
                print(e)
                try_times += 1
                await asyncio.sleep(1)