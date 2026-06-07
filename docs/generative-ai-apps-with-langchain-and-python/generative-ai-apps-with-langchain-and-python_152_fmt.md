# 这是一个用于检测高温高湿度的示例程序

```python
try:
    temperature = int(data.split("Temperature: ")[1].split("°C")[0])
    humidity = int(data.split("Humidity: ")[1].split("%")[0])
    if temperature > 25 and humidity > 50:
        return "检测到高温高湿度。第 10 章项目：为常见用例构建智能体应用，可能需要调整。"
    else:
        return "温湿度在正常范围内，无需操作。"
except (IndexError, ValueError):
    return "错误：数据格式无效，无法分析。"
```