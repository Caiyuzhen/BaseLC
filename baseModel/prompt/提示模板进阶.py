from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

def get_userInput():
    story_content = input("请输入故事主人: ")
    story_style = input("请输入故事风格: ")
    return story_content, story_style


# 【Case】直接输入两个选项便生成 _______________________________________________________________________________________________________
def run_llm(stories_content: str, stories_style: str) -> str:
    llm = OpenAI()
    
    
    # 👇【提示词模板构建方式一】构建一个包含多个提示词的提示词模板 最终要传给 chain!! ——————————————————————————————
    multiple_input_prompt = PromptTemplate(
        template="给我讲个关于{content_placeholder}的{style_placeholder}风格的故事, 50 字",
		input_variables=["content_placeholder", "style_placeholder"]
	)
    multiple_input_prompt.format(content_placeholder=stories_content, style_placeholder=stories_style) # 把用户的输入进行格式化
    
	# ⚡️ 把格式化后的内容进行实例化
    chain = LLMChain(llm=llm, prompt=multiple_input_prompt) 
    
     
    # 👇【提示词模板构建方式二】, 使用 from_template 方法 —————————————————————————————————————————————————————
    # template = "给我讲个关于{content_placeholder}的{style_placeholder}风格的故事, 50 字"
    # prompt_template = PromptTemplate.from_template(template)
    # prompt_template.format(content_placeholder=stories_content, style_placeholder=stories_style)
    
    # # ⚡️ 把格式化后的内容进行实例化
    # chain = LLMChain(llm=llm, prompt=prompt_template)
    

	# 🌟 以【字典形式】将格式化的提示作为参数传递给 run 方法
    res = chain.run({ # 🌟放入用户输入的问题
        "content_placeholder": stories_content, 
        "style_placeholder": stories_style
	})
    return res




# 【Case】输入一个提示词, 生成一个模板, 生成另一个完整的语句 ______________________________________________________
def get_date_time(): # 获得当前时间 => 某个提示词参数可以通过函数获得
    now_time = datetime.now()
    return now_time.strftime("%m/%d/%Y, %H:%M:%S")


def run_llm2(date_time) -> str:
    llm = OpenAI()
    
    prompt = PromptTemplate(
		template="告诉我{city}在{year}年{date}的平均气温",
		input_variables=["city", "year", "date"]
	)
    partial_prompt = prompt.partial(city="HongKong", year="1998") # 🔥 生成了一个新的提示词模板! 有了 Hongkong 的变量
    partial_prompt.format(date=date_time) # 🔥 输入时间, 最终生成
    chain = LLMChain(llm=llm, prompt=partial_prompt)
    print(f"完整提示词: {partial_prompt}")
    res = chain.run(date_time)
    return res
    

if __name__ == "__main__":
    # Case 1 - 生成故事:
    # story_content, story_style = get_userInput()
    # res = run_llm(story_content, story_style)
    
    # Case 2 - 获得天气:
    now_time = get_date_time()
    res = run_llm2(now_time)
    print(res)