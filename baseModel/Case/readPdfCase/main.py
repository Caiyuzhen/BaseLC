# 利用 fastApi 写 web 应用
# uvicorn main:app --reload 启动服务器

from fastapi import FastAPI,File,Form,UploadFile
from typing import Annotated
import os
from langchain_helper import *
from pydantic import BaseModel


app = FastAPI()

file_upload_path = "./uploads"# 文件上传路径

# 主页
@app.get("/")
def read_root():
    return "Home page"


# 🌟🌟 上传文件 -> 调用 langchain 服务 -> 保存为向量 ————————————————————————————————————————————————————————————————————————
@app.post("/upload")
def upload_pdf(
	file: Annotated[UploadFile, File()], # 文件
 	index_name: Annotated[str, Form()] # 🌟 用户需要上传文件向量索引名称, 利用 fastApi 的表单上传能力
):
     # 🌟 拿到文件要保存到的文件位置
    file_upload_target_path = os.path.join(file_upload_path, file.filename)
    with open(file_upload_target_path, "wb") as f: # 打开文件
        contents = file.file.read() # 拿到文件内容
        f.write(contents) # 把内容写入文件
        
    # 🌟 引入自 langchain_helper.py
    load_pdf_and_save_to_index(file_upload_target_path, index_name)
    
    # 返回 json 格式, 包含文件名跟索引名
    return {"filename": file.filename, "index_name": index_name} 
	
 
# 定义请求的 Schema
class Query(BaseModel): 
    index_name: str
    query: str
    
 
# 🌟🌟 检索向量文件数据的请求 —————————————————————————————————————————————————————————————————————————————————————————————————
@app.post("/query")
def query_index(request: Query):
    index_name = request.index_name
    
    # 🌟首先拿到要获取的数据库的索引 => index_name 从【请求的 JSON 中】获取
    index = load_index(index_name)
    
    # 🌟其次再拿到要查询的问题 => query 从【请求的 JSON 中】获取
    query = request.query
    
    # 调用封装好的 langchain 查询函数
    llmAnswer = query_index_lc(index, query)
    
    # 返回 API 的结果
    return {"answer": llmAnswer, "index_name": index_name, "query": query} # 🔥返回【回答】【向量数据库索引】【用户输入的问题】
    
    
    