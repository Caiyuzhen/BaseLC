from langchain.llms import OpenAI

llm=OpenAI()
generate_res = llm.generate(["讲一个笑话", "讲一个好故事"])
print(generate_res[0][0].text)