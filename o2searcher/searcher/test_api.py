import requests

payload = {
    "queries": [["Principles and methodologies of explainable artificial intelligence (XAI) in healthcare", "Applications of XAI in medical diagnosis: Case studies and examples"], ["Challenges and limitations of implementing XAI in medical diagnosis"], ["when did Ford stop producing the 7.3 diesel?"]]
}
# response = requests.post("http://127.0.0.1:10102/search", json=payload)
response = requests.post("http://127.0.0.1:10001/wiki_search", json=payload)
print(response.json())