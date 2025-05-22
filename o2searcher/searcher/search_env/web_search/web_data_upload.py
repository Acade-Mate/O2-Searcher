import json
import uuid
import time
import meilisearch

client = meilisearch.Client('http://localhost:7700', 'Web_Knowledge_Corpus')

with open('./data/Web_data.json', encoding='utf-8') as json_file:
    Web_Corpus = json.load(json_file)

total_docs = len(Web_Corpus)
batch_count = 50
batch_size = (total_docs + batch_count - 1) // batch_count

for i in range(0, total_docs, batch_size):
    batch = Web_Corpus[i:i+batch_size]
    client.index('Web_Corpus').add_documents(batch)
    print(f"Have processed {i//batch_size + 1}/{batch_count} batches, added {len(batch)} articles")
    time.sleep(0.5)  

print(f"Have processed {total_docs} articles and added them to MeiliSearch")

