error_prompt = '''The response you attempted before is invalid. If you plan to execute actions like SEARCH, you need to enclose the SEARCH queries within the <search> and </search> tags. Furthermore, the required queries for the SEARCH action should be placed between the <query> and </query> tags. Moreover, if you wish to present the final output for the initial query, you must wrap the result within the <answer> and </answer> tags.
'''

extra_prompt = '''Search learnings: <learnings>{learning_str}</learnings>.
'''

