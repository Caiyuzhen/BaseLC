"""
	WHY?
		模块的抽象化, 避免牵一发而动全身
			比如:
				定义提示词
				选择 LLM
				运行接口
				...
	WHAT?
		链是不同模块的组合
		链是一个抽象出来的标准,提高了复用性, 但不一定要用链才能完成调用 LLM
	HOW?
		基础链结构
			LLM chain
			Router chain => 多链并行然后选择一条进行处理
			Sequential chain => 多链串行
		应用链结构实例
			Document chains
			Retrieval QA
"""	

# 【🌟 Case-单链结构 single chain - 至少调用一次 llm 】————————————————————————————————————————————————————————————
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from langchain.chains import ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import SimpleSequentialChain

# llm=OpenAI()

prompt_temp = "给我一个小猫咪的昵称"
llm_chain = LLMChain(
	llm=llm,
	prompt=PromptTemplate.from_template(prompt_temp)
)
res1 = llm_chain.predict()
print(res1)



# 【🌟 Case-多链结构 Multiple. chain - 至少调用两次 llm 】————————————————————————————————————————————————————————————
# 比如有文学家、翻译专家、数学家等等不同的专家链: 定义 prompt => 定义llm/embedding => 定义chain(多链并行然后选择一条进行处理) => 运行 predict
# 👇 {input} 在模板字符串中作为占位符使用, 比如用户输入 "这只猫叫什么好？"，那么 {input} 就会被替换为 "这只猫叫什么好？"
llm=OpenAI()
name_temp = """你是一个非常有创意的起名专家. \
    你非常擅长给小猫咪起好听的、富含寓意的名字. \
    用中文给出 3 种回答
    
    这里是问题:
    {input}
"""

tran_temp = """你是一个非常精通英语的翻译专家. \
    你非常擅长将中文翻译成英文. \
	用英文给出你的回答
 
    这里是问题:
    {input}
"""


prompt_infos = [ # 🔥定义 Router chain 需要给到这部份信息
	{
		"name": "naming",
  		"description": "擅长用中文起名字",
    	"prompt_template": name_temp,
	},
 	{
		"name": "english",
  		"description": "擅长将中文翻译成英文",
		"prompt_template": tran_temp,
	}
]

destination_chains = {} # 用来存储不同的单链

# 👇对每条单链进行初始化
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(
		template=prompt_template,
  		input_variables=["input"] # 表示模板中的 {input} 会被替换为用户输入的内容
	)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain # 表示将链存储到 destination_chains 中

default_chain = ConversationChain(llm=llm, output_key="text") # 定义默认的链



# 👇配置 router 的 prompt
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos] # 遍历 prompt_infos
destinations_str = "\n".join(destinations) # 将 destinations 用换行符连接起来

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str) # 将 destinations_str 传入到模板中 => 就是遍历出来的 prompt_infos
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt) # 用 LLMRouterChain.from_llm() 方法创建 router_chain


# 👇将 router_chain 和 destination_chains 传入到 MultiPromptChain 中
chain = MultiPromptChain(
	router_chain=router_chain,
	destination_chains=destination_chains,
 	default_chain=default_chain,
  	verbose=True
)


print(chain.run("这只猫叫什么好？"))
print(chain.run("苹果的英文"))





# 【🌟 Case-多链串行结构 Multiple. chain - 至少调用两次 llm 】————————————————————————————————————————————————————————————
llm=OpenAI()
temp = """你是一个非常有创意的起名专家. \
    你非常擅长给小猫咪起富含寓意的名字. \
    用中文给出 3 个回答
    
    这里是问题:
    {input}
"""

temp = """你是一个非常精通名字的评论家. \
    你非常擅长分析名字的优劣, 给名字做分析. \
	判断哪个名字比较好
 
    名字分析:
    {synopsis}
    对上面的名字进行分析, 判断哪个更好:
"""


# 定义模板结构-A
prompt_template = PromptTemplate(template=temp, input_variables=["input"]) # 模板中的 {input} 会被替换为用户输入的内容
# 定义链结构-A
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# 定义模板结构-B
prompt_template = PromptTemplate(template=temp, input_variables=["synopsis"])# 模板中的 {synopsis} 会被替换为用户输入的内容
# 定义链结构-B
review_chain = LLMChain(llm=llm, prompt=prompt_template)

# 用来存储不同的单链
overall_chain = SimpleSequentialChain(
	chains=[synopsis_chain, review_chain],
	verbose=True
)

review = overall_chain.run("给我一些小猫咪的名称")




# 【🌟 Case-文本链处理 Transformation chain】————————————————————————————————————————————————————————————
# 比如只是把你的 prompt 的第一句话跟最后一句话截取出来发给 llm
# Stuff



# 【🌟 Case-长文本链处理 Document chain type】比如处理向量数据库可以 ————————————————————————————————————————————————————————————
# 比如只是把你的 prompt 的第一句话跟最后一句话截取出来发给 llm
# Refine
# Map reduce
# Map rerank