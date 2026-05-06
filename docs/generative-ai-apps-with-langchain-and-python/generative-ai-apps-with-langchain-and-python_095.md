# 开始对话

```python
chat_history = []

while True:
    question = input("用户: ")
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"助手: {result['answer']}")
```

在此示例中，你初始化了检索器和 LLM，创建了 `ConversationalRetrievalChain`，然后启动了一个对话循环。该链负责利用聊天历史来优化查询并提供上下文相关的响应。

