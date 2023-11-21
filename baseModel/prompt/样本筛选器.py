from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


"""
	样本选择器
		WHY?
			当样本数量非常多的时候, 我们可以通过样本选择器来选择一部分样本, 这样可以提高模型的效率
			缩短发送给大模型的提示词的数量, 更省钱
		HOW?
			SemanticSimilarityExampleSelector 语义相似度筛选器
"""

EXAMPLES = [
	{"question": "你好吗?", "answer": "当然!我很好!"},
    {"question": "今天天气怎么样?", "answer": "天气很不错!"},
    {"question": "今天的食物怎么样?", "answer": "食物很美味!"},
    {"question": "你会说英语吗?", "answer": "是的，我可以回答英语问题。"},
    {"question": "如何解决数学题?", "answer": "我可以尝试帮你解决数学问题。"},
    {"question": "你能推荐一本书吗?", "answer": "当然，我推荐《哈利·波特》系列。"},
    {"question": "今天有什么新闻?", "answer": "我无法提供最新新闻，但可以讨论近期的热点话题。"},
]


def get_user_input():
	input_content = input("请输入问题: ")
	return input_content


def run_llm(input_content):
    llm = OpenAI()
    
    example_prompt = PromptTemplate(
		input_variables=["question", "answer"],
		template="Question: {question}\n{answer}" # 🔥相当于利用上面 EXAMPLES 的数据进行格式化
	)
    

    example_selector = SemanticSimilarityExampleSelector.from_examples( # 👈 语义相似度筛选器
  		examples=EXAMPLES, # OpenAIEmbeddings(), # The embedding class used to produce embeddings which are used to measure semantic similarity.
        embedding_model=OpenAIEmbeddings(), # Chroma, # The VectorStore class that is used to store the embeddings and do a similarity search over.
        vector_store_class=Chroma, # 向量数据库和搜索库, 用于储存和管理 OpenAIEmbeddings 生成的向量
        k=3  # 选择最相似的3个示例
	)
    
    prompt = FewShotPromptTemplate(
		examples = EXAMPLES,
		example_prompt = example_prompt,
		suffix="Question: {input}", # 🔥 以 {input} 作为问题的输入
		input_variables=["input"],
  		example_selector=example_selector, # 使用语义相似度选择器
	)

    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.run(input_content) # 🌟放入用户输入的问题
    return res


if __name__ == "__main__":
    input_content = get_user_input()
    result = run_llm(input_content)
    print(result)
