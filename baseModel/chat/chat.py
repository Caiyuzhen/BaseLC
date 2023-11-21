"""
	聊天模块 Chat module => 可以保留历史上下文等更多能力
		AI Message
  		Human Message
		System Message => 设定 AI 回答的角色设定上下文环境
	调用方法
		predict_messages
"""
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

# 实例化
chat_model = ChatOpenAI()

text = "一个好听的 AI 创业公司的名称"
messages = [HumanMessage(content = text)] # 用对话的形式进行调用
res = chat_model.predict_messages(messages)
print(res) # AIMessage(content='智能未来科技', additional_kwarge={}, example=False)






