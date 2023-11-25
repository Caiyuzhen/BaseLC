from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper #封装类


file_path = "./test2.pdf"
local_persist_path = "./vector_store" #定义index 向量数据要存在本地的数据库路径


# 1.加载环境变量, 得到 openAI key
load_dotenv()
 
 
 

# 🏆 拼接本地向量数据库的名称
def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name) # 把【向量数据库路径】跟【要保存为的向量索引名】做一个拼接


# 🌟 加载 Pdf, 保存为本地向量数据库
def load_pdf_and_save_to_index(file_path, index_name): # 【要传入的 pdf 路径】、【要保存为的向量索引名】

	# 2.加载 pdf
	loader = PyPDFLoader(file_path)

	# 3.初始化【向量数据库索引生成器】, 会返回一个【索引的包装类】
	index = VectorstoreIndexCreator(
		vectorstore_kwargs={"persist_directory": get_index_path(index_name)} #传入存储路径
		).from_loaders([loader]) # 加载 pdf 文件

	index.vectorstore.persist()#执行向量的本地存储 => 🌟 本质是基于 chroma

	# 4.查询数据, query_with_sources, 返回回答
	# answer = index.query_with_sources("希腊政府做了什么?", chain_type="map_reduce") # 每次只丢一部分数据给大模型, 最后一得到一个汇总的东西
	# print(answer)
 

# 🚀 加载已经保存好的向量数据库
def load_index(index_name):
    index_path = get_index_path(index_name) # 拿到之前保存的向量数数据库的路径
    embedding = OpenAIEmbeddings()# 因为用的是 openAI 的 embedding 服务, 所以也要用 openAI 的 embedding 来加载
    vectordb = Chroma(
		persist_directory = index_path,
  		embedding_function =  embedding
	)
    return VectorStoreIndexWrapper(vectorstore=vectordb) # 因为返回的是 VectorStoreIndexWrapper 封装类
 
 
# 🔍 封装查询语句
def query_index_lc(index, query):
    answer = index.query_with_sources(query, chain_type="map_reduce") # 每次只丢一部分数据给大模型, 最后一得到一个汇总的东西
    return answer['answer']
  
  
  
# if __name__ == "__main__":
    # 把某个文件保存为本地向量数据库
    # load_pdf_and_save_to_index(file_path, "test002") # 文件路径、保存在本地的向量数据库的名称
    
    # 加载某个本地向量数据库并进行提问 (未封装的写法)
    # index = load_index("test002")
    # answer = index.query_with_sources("希腊政府做了什么?", chain_type="map_reduce") # 每次只丢一部分数据给大模型, 最后一得到一个汇总的东西
    
    # 封装后的写法 🌟
    # index = load_index("test002")
    # print(query_index_lc(index, "希腊政府做了什么"))