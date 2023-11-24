# 利用 fastApi 写 web 应用
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return "Home page"
