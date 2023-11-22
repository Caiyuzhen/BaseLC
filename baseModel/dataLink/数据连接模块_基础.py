"""
	WHY?
		访问模型中没有的数据
	HOW?
		文档加载器 => 从多种数据源中加载文档(网页、pdf、ppt)
		文档转换器 => 拆分文档(文本分割器), 处理文档加载器加载出来的文档
		文本 embedding 模型 => 将非结构化的文本转化为浮点数的列表
		向量存储站 => 存储向量数据
		检索器 => 查询、搜索 embedding 数据
"""

# 🚀 Step 1 => 文档转换
# 【🌟 加载结构化的数据】————————————————————————————————————————————————————————————
from langchain.document_loaders import CSVLoader

# 定义路径
csv_loader = CSVLoader(file_path='./data.csv')

# Load the documents
documents = csv_loader.load()

# print(documents) # 打印出 6 个 document, 对应 CSV 文件的 6 行
len(documents)
data = documents[0].page_content
print(data) # 打印出类似字典格式的数据



# 【🌟 加载非结构化的数据】————————————————————————————————————————————————————————————
# 如图片、ppt 、html、pdf、纯文本等, 所有文件格式可见: https://python.langchain.com/docs/integrations/document_loaders/



# 【🌟 URL 加载器】————————————————————————————————————————————————————————————
# UnstructuredURLLoader





# 🚀 Step 2 => 文档转换, 比如把文档拆分成更小的颗粒（避免传入 llm 的 token 太多）
# 文本分割器（以句子为单位把文档拆分成小块, 拆分成一定的数量), 然后成为一个新的文档



# 🚀 Step 3 =>  文本词嵌入, 将一个词映射到一个高维向量中（次向量 Vector, 比如"我"跟"你"的向量距离会比"我"跟"他"的距离更接近）, 进行数字化的表达, 然后嵌入模型 openAI | Cohere | Hugging Face 的等
""" 
	例子:
		"我" -> [0, 0.3, -0.1]
		"你" -> [-0.6, 0.12, 0.16]
		"他" -> [0.01, 0.298, -0.99]
"""


# 🚀 Step 4 =>  存储向量数据


# 🚀 Step 5 =>  检索向量数据
# 检索器
