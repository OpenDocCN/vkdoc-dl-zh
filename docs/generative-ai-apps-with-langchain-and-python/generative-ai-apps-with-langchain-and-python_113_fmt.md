# 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

### 文本嵌入代码演练

让我们通过一个实际示例来了解如何使用文本嵌入从产品客户评论列表中提取见解，并获取有价值的客户反馈和情感分析。

以下是完整的端到端可运行代码。

首先，安装所需的包（`langchain` 和 `langchain-openai`）并导入必要的模块：

```
# 安装所需的包
!pip install langchain
!pip install langchain-openai

# 导入必要的模块
from langchain_openai import OpenAIEmbeddings
import os
```

使用环境变量 `OPENAI_API_KEY` 设置 OpenAI API 密钥。请务必将 `"your_api_key_here"` 替换为你实际的 OpenAI API 密钥：

```
#### 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

初始化一个名为 `embeddings_model` 的 `OpenAIEmbeddings` 类实例，该实例将作为我们后续代码中的嵌入模型：

```
# 初始化 OpenAIEmbeddings 类
embeddings_model = OpenAIEmbeddings()
```