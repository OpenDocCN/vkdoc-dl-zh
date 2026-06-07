# 现在我们将向 AI 发送请求

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a market research analyst with expertise in consumer goods."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,
    n=1,
    temperature=0.6,
)
```