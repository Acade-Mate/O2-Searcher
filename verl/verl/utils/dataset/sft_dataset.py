# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 json_files: Union[str, List[str]],
                 tokenizer,
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(json_files, List):
            json_files = [json_files]

        self.json_files = json_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_length = max_length

        self._read_files()


    def _read_files(self):
        data_list = []
        for json_file in self.json_files:
            with open(json_file, "r") as f:
                data_list.extend(json.load(f))
        
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.data_list[item]['messages']

        chat_str = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        input_ids_output = tokenizer(chat_str, return_tensors='pt', add_special_tokens=False)
        input_ids = input_ids_output['input_ids'][0]
        attention_mask = input_ids_output['attention_mask'][0]

        roles = {'system': '<|im_start|>system', 'user': '<|im_start|>user', 'assistant': '<|im_start|>assistant'}
        # get loss masks
        loss_mask = attention_mask.clone()
        start_idx = 0
        check_str = ''
        for sentence in messages:
            sentence_str = roles[sentence['role']]+'\n'+sentence['content']+'<|im_end|>\n'
            check_str += sentence_str
            prompt_length = len(tokenizer(sentence_str, return_tensors='pt', add_special_tokens=False)['input_ids'][0])
            if sentence['role'] in ['system', 'user']:
                loss_mask[start_idx:start_idx + prompt_length] = 0
            elif sentence['role'] == 'assistant':
                loss_mask[start_idx:start_idx+3] = 0
            start_idx += prompt_length
        assert check_str == chat_str

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
