## Install
pip3 install -r requirements.txt
- pip3 install langchain
- pip3 install openai
- pip3 install numexpr
- pip3 install pypdf
- pip3 install tiktoken 
  - OpenAI 的向量标记依赖
- pip3 install chromadb 
  - 本地向量存储解决方案
- pip3 install python-dotenv
- pip3 install fastapi
- pip3 install "uvicorn[standard]"
  - 用来启动 fastapi 的 ASGI 服务器
- pip3 install python-multipart
  - fastApi 处理文件上传的必要依赖
- pip3 install unstructured


## Set API token
- export OPENAI_API_KEY="..."
- export PYTHONIOENCODING=utf-8


## Notices
- 🌟 Python 解释器最好是 3.12 版本+
  - 通过命令面板（Ctrl+Shift+P）使用“Python: Select Interpreter”命令，选择解释器


## Git 冲突
- git pull --no-rebase origin main


# 启动 FastApi 服务器
- uvicorn main:app --reload
- uvicorn main:app --host 0.0.0.0 --reload
  - 让局域网内的人也能通过本机 ip 进行访问, 0.0.0.0 表示绑定本机 ip

# 🌟 OpeanAI调用失败的情况
- VPN 开启全局模式
