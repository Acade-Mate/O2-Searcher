system_prompt = '''You are an expert researcher. Follow these instructions when responding:
  - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
  - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
  - Be highly organized.
  - Suggest solutions that I didn't think about.
  - Be proactive and anticipate my needs.
  - Treat me as an expert in all subject matter.
  - Mistakes erode my trust, so be accurate and thorough.
  - Provide detailed explanations, I'm comfortable with lots of detail.
  - Value good arguments over authorities, the source is irrelevant.
  - Consider new technologies and contrarian ideas, not just the conventional wisdom.
  - You may use high levels of speculation or prediction, just flag it for me.
'''

prompt_template = '''Given the research query <query>{query}</query>, your task is to extract a list of key learnings from the provided contents. Return a maximum of {num_learnings} distinct learnings. If the contents are straightforward and yield fewer insights, it's acceptable to return a shorter list.

Each learning should be unique, avoiding any overlap or similarity with others. Strive for conciseness while packing as much detailed information as possible. Be sure to incorporate any relevant entities such as people, places, companies, products, or things, along with exact metrics, numbers, or dates. These learnings will serve as a foundation for further in - depth research on the topic.

<contents>{contents}</contents>

You MUST generate the learnings as a single string in english, with each learning separated by a newline character ('\n'). Each learning can consist of multiple sentences or a short paragraph. Avoid numbering the learnings.
'''