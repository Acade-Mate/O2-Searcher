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

from transformers import AutoTokenizer
from verl.utils import hf_tokenizer
from verl.utils.dataset.sft_dataset import SFTDataset


def test_sft_dataset():
    tokenizer = hf_tokenizer('')
    local_path = ''
    dataset = SFTDataset(json_files=local_path,
                         tokenizer=tokenizer,
                         max_length=10240)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    assert len(output) > 1
    assert type(output) == str


if __name__ == '__main__':
    test_sft_dataset()