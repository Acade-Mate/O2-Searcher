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
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def test_rl_dataset():
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer('')
    local_path = ''
    dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=10240)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)
    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    data = dataset[-1]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(f'type: type{output}')
    print(f'\n\noutput: {output}')


if __name__ == '__main__':
    test_rl_dataset()
