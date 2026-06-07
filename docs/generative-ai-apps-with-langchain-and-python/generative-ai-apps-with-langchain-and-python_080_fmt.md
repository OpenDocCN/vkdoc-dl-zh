# 为 LLM 创建带指令的提示模板

`prompt = PromptTemplate(template="请按以下格式提供电影信息：\n{format_instructions}\n 问题：{query}", input_variables=["query"], partial_variables={"format_instructions": parser.get_format_instructions()})`

#### 设置 OpenAI API 密钥

你通过`os.environ.get()`从环境变量中获取 OpenAI API 密钥。如果环境变量未设置，则使用默认值（将`"your_api_key_here"`替换为你的实际 API 密钥）：

```
#### 设置 OpenAI API 密钥
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    openai_api_key = "your_api_key_here"  # 替换为你的实际 API 密钥
```

#### 选择语言模型及其设置

你创建一个名为`llm`的`ChatOpenAI`实例，并指定模型名称（`"gpt-3.5-turbo"`）、温度参数（`0`）以及 OpenAI API 密钥：

```
# 选择 LLM 及其设置
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
```