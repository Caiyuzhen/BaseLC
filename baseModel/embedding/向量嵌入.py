from langchain.embeddings import OpenAIEmbeddings

# 创建 OpenAIEmbeddings 实例
embedder = OpenAIEmbeddings()

# 需要嵌入的文本
text = "这是一个示例文本。"

# 生成文本的嵌入向量
embedding = embedder.embed(text)

# 打印嵌入向量
print(embedding)