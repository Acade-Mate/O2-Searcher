# O²-Searcher - local search environment

## Getting Started 🎯
### Installation

```bash
conda create -n searcher python=3.10
conda activate searcher

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi

# install meilisearch
pip install meilisearch

```

## 2. For Web Knowledge Search on Web pages


```bash
cd web_search

curl -L https://install.meilisearch.com | sh

./meilisearch --master-key="Web_Knowledge_Corpus"

python web_data_upload.py

python web_search.py
```

## 3. For Structured Knowledge Search on Wikipedia

3.1 Download the wikipedia data

```bash
cd wiki_search

save_path=/the/path/to/save

python wiki_download.py --save_path $save_path

cat $save_path/part_* > $save_path/e5_Flat.index

gzip -d $save_path/wiki-18.jsonl.gz
```

3.2 Build the [dense retriever](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md) (We recommend to run this on GPU)

```bash
python wiki_search.py
```

