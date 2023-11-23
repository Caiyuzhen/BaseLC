"""
	Agent 代理人
		Why?
  			Agent 代表着未来, 有着高复杂性, 扩展了 LLM 的边界, 能够从外部获取数据, 因为 LLM 不擅长更新自己的数据
			Agent 代表着解决方案, 能够跨不同的 AI 工具来解决复杂人物(比如人形检测 + 人脸识别 + LLM + ...)
"""


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
