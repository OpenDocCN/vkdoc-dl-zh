# 设置 OpenAI API 客户端

`openai.api_key = api_key`

## 第 2 章 使用 LangChain 集成 LLM API

在这段代码中，你使用 `os.getenv()` 函数从名为 `OPENAI_API_KEY` 的环境变量中获取 API 密钥。然后，在发起 API 请求之前，将获取到的值赋给 `openai.api_key`。

#### 使用配置文件保护 API 密钥

你也可以按照以下方法使用配置文件来存储 API 密钥：

a) 创建一个名为 `config.json` 的文件，内容如下：

```
{ "OPENAI_API_KEY": "your_api_key_here" }
```

b) 将 `config.json` 添加到你的 `.gitignore` 文件中，以防止它被版本控制系统跟踪。

c) 在你的 Python 代码中：

```
import json
import openai

# 读取配置文件
with open('config.json') as f:
    config = json.load(f)

# 从配置中获取 API 密钥
api_key = config['OPENAI_API_KEY']
```



