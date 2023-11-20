"""
	WHY?
		因为训练大模型所用的数据是有限的, 因此需要使用一系列的工具来完成复杂任务 (比如访问互联网), 代理人可以使用更多的工具 tools
"""

from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType


llm = OpenAI()


def get_user_input():
    user_input = input("输入数学计算问题:")
    return user_input


def run_llm(user_input: str):
	# 2. 定义代理人使用的工具
	tools = load_tools(['llm-math'], llm=llm)

	# 3. 初始化代理人
	agent = initialize_agent(
		tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
	) # verbose 用来看代理人如何一步步的去处理问题

	res = agent.run(user_input)
	return res


if __name__ == "__main__":
    user_input = get_user_input()
    res = run_llm(user_input)
    print(res)
    
"""
	> Entering new AgentExecutor chain...
	I need to calculate the 3.5th power of 22
	Action: Calculator
	Action Input: 22^3.5
	Observation: Answer: 49943.547010599876
	Thought: I now know the final answer
	Final Answer: 49943.547010599876

	> Finished chain.
	49943.547010599876
"""