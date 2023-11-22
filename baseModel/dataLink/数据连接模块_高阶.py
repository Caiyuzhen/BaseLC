"""
	Flow:
		小说节选 -> 加载 -> 产生词向量 -> 存储向量 -> 检索整部小说中相似的向量
"""

# 数据加载器 Document loader
# pip3 install pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# 【🌟 第 1 步: 加载文档】 ————————————————————————————————————————————————————————————————
pdf_loader = PyPDFLoader("./test2.pdf")
# documents = pdf_loader.load()
# pdf_len = len(documents) # 默认把 pdf 的每一页对应成一个 document
# print(pdf_len) # 4 页 document



# 【🌟 第 2 步: 加载并分割文档】 ————————————————————————————————————————————————————————————————
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 100, # 会把文档先分割成小块
 	chunk_overlap = 20, # 然后再组合成中等大小的块, chunk_overlap 表示组合后重叠的部份有多少个字符
	length_function = len,
)

pages = pdf_loader.load_and_split(text_splitter=text_splitter) # text_splitter 提供文本分割器再去加载文档
pdf_len = len(pages)
# print(pdf_len)
# print(pages[6]) # 随机查看一页的内容



# 【🌟 第 3 步: 文本词嵌入】 ————————————————————————————————————————————————————————————————
embeddings_model = OpenAIEmbeddings() # 👉 需要依赖 pip3 install tiktoken => 把文本分片
embeddings = embeddings_model.embed_documents([pages[6].page_content]) # 👈相当于取出第 6 页的内容, 生成词向量
# print(len(embeddings[0])) #1536 => 用 1536 个浮点数去描述这个【词向量】



# 【🌟 第 4 步: 存储向量】 ————————————————————————————————————————————————————————————————
db = Chroma.from_documents(pages, OpenAIEmbeddings()) # 🔥利用 OpenAI 的词嵌入能力 OpenAIEmbeddings => 最终返回 db, 就是做好词嵌入的数据库



# 【🌟 第 5 步: 数据检索】 ————————————————————————————————————————————————————————————————
query = "希腊"
docs = db.similarity_search(query) # 🔥拿到 pdf 内所有跟希腊相关的内容
print(len(docs))
print(docs[0])
