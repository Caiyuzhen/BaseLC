"""
	WHY?
		用来封装 Langchain 尚未支持的大模型, 自定义返回的数据
"""

from typing import Any, List, Mapping, Optional
from langchain.llm.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class SuperAI(LLM): # 继承 LLM 父类
    @property # 将方法转换为属性, 从类的实例调用 _llm_type 时，它表现得像一个属性而不是一个方法
    def _llm_type(self) -> str:
        return "我是SuperAI"
    
    def _call( # 它重写了父类 LLM 的 _call 方法。它定义了如何处理给定的输入 prompt
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		run_manager: Optional[CallbackManagerForLLMRun] = None,
	) -> str:
        if stop is None:
            raise ValueError("stop kwargs are not permitted.")
        pd = prompt.find("吗") # 检查 prompt 中是否包含“吗”，如果包含，则删除“吗”并返回修改后的句子。否则，返回“哦。” => 比如 “吃了”
        if pd >= 0:
            return prompt[0:pd] + "。 " 
        return "哦。"
    
    
llm=SuperAI()
print(llm("你好吗？")) # 哦。
