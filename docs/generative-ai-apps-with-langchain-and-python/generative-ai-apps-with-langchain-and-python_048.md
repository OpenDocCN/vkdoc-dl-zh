# 创建一个生成响应的函数

```
def generate_response(text):
    messages = [HumanMessage(content=text)]
    response = chat_model.invoke(messages)
    return response.content
```

