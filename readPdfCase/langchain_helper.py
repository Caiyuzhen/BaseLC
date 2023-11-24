from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from dotenv import load_dotenv
import os


file_path = "./test2.pdf"
local_persist_path = "./vector_store" #定义index 向量数据要存在本地的数据库路径


def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name) # 把【向量数据库路径】跟【要保存为的向量索引名】做一个拼接


def load_pdf_and_save_to_index(file_path, index_name): # 【要传入的 pdf 路径】、【要保存为的向量索引名】
	# 1.加载环境变量
	load_dotenv()

	# 2.加载 pdf
	loader = PyPDFLoader(file_path)


	# 3.初始化【向量数据库索引生成器】, 会返回一个【索引的包装类】
	index = VectorstoreIndexCreator(
		vectorstore_kwargs={"persist_directory": local_persist_path} #传入存储路径
		).from_loaders([loader]) # 加载 pdf 文件

	index.vectorstore.persist()#执行向量的本地存储 => 基于 chroma

	# 4.查询数据, query_with_sources, 返回回答
	# answer = index.query_with_sources("希腊政府做了什么?", chain_type="map_reduce") # 每次只丢一部分数据给大模型, 最后一得到一个汇总的东西
	# print(answer)