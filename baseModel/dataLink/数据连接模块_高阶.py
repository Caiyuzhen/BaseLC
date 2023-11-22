"""
	Flow:
		小说节选 -> 加载 -> 产生词向量 -> 存储向量 -> 检索整部小说中相似的向量
"""

# 数据加载器 Document loader
# pip3 install pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 【🌟 第 1 步: 加载文档】 ————————————————————————————————————————————————————————————————
pdf_loader = PyPDFLoader("test.pdf")
# documents = pdf_loader.load()
# pdf_len = len(documents) # 默认把 pdf 的每一页对应成一个 document
# print(pdf_len) # 4 页 document



# 【🌟 第 2 步: 加载并分割文档】 ————————————————————————————————————————————————————————————————
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 100, # 设置一个非常小的文本块大小
 	chunk_overlap = 20, # 重叠部份有多少个字符
	length_function = len,
)

pages = pdf_loader.load_and_split(text_splitter=text_splitter) # text_splitter 提供文本分割器再加载
pdf_len = pages(pages)
print(pdf_len)
