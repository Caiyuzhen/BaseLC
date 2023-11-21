from langchain.llms import OpenAI
from langchain.chains import ConversationChain


"""
	聊天模块 => 保留用户输入的上下文
"""
def get_user_input():
   user_input = input("输入对话消息: ") 
   return user_input


def run_llm(user_input: str):
	res = conversation.run(user_input)
	return res
 
if __name__ == "__main__": 
	llm = OpenAI()
	conversation = ConversationChain(llm=llm, verbose=True) # 🔥单实例, 不要每次都建立一个新的 conversation !!
 
	while True:
		user_input = get_user_input()

		# 检查用户是否输入了“结束”，如果是，则退出循环
		if user_input == "结束":
			print("对话结束")
			break

		res = run_llm(user_input)
		print(res)
 