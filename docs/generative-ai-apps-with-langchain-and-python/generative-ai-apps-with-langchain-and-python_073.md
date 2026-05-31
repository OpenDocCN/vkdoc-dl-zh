# 第 5 章 掌握创意内容的提示词

### 使用提示模板调用 LLM

是时候创建 LLM 对象了。以下是使用 LangChain 与 OpenAI 时的示例。你应该注意，初始化和调用 LLM 的过程可能因特定模型、API 或框架而异：

```python
llm = OpenAI(openai_api_key="Your OpenAI key", temperature=0.7)
```

最后，你使用提示字符串作为参数调用 LLM，并打印响应：

```python
# 创建一个非常简单的 LLM 链（注意：会丢失一些输出控制）
chain = LLMChain(llm=llm, prompt=prompt_template)
```

在上述代码中，你创建了`LLMChain`类的一个实例，它代表了语言模型中的一个简单链。`LLMChain`构造函数接受两个参数，即`llm`和`prompt`，你已将之前创建的提示模板赋值给`prompt`。

**注意**：请记住导入必要的库才能使其工作。例如，你需要导入以下内容：

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
```

然后，你通过调用`chain`的`run`方法生成响应，该方法接受两个关键字参数，通过这两个参数传入你之前用于构建提示模板的两个关键字，即算法类型和语言：

