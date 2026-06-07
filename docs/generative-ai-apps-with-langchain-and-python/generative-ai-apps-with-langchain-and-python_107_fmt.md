# RecursiveCharacterTextSplitter

`recursive_text_splitter = RecursiveCharacterTextSplitter(separators=[". ", "! ", "? "], chunk_size=30, chunk_overlap=5)`
`recursive_documents = recursive_text_splitter.create_documents([text])`
`print("RecursiveCharacterTextSplitter:")`
`for doc in recursive_documents:`
`print(doc.page_content)`
`print("---")`

以下是输出结果：

`CharacterTextSplitter:`

`This is a sample text.`

`It consists of multiple sentences.`

`Some sentences are longer than others.`

`We will split this text into chunks.`

`RecursiveCharacterTextSplitter:`

`This is a sample text.`

`It consists of multiple sentences.`

`Some sentences are longer than`

`others. We will split this text`

`into chunks.`

如你所见，`CharacterTextSplitter` 根据指定的分隔符（本例中为 `. `）对文本进行分割，并相应地创建文本块。另一方面，`RecursiveCharacterTextSplitter` 会对那些超出 `chunk_size` 限制的文本块进行递归分割，从而得到分布更均匀的文本块。

最终，你需要根据具体需求选择合适的文本分割器。如果你有较长的段落或章节需要分割成更小的文本块，同时又要保留其逻辑结构，那么 `RecursiveCharacterTextSplitter` 可能是更好的选择。否则，对于大多数文本分割任务来说，`CharacterTextSplitter` 是一个更简单直接的选择。

### CodeTextSplitter

在处理源代码文件或代码片段时，你也可以使用 `CodeTextSplitter`。它能够帮助你根据编程语言的结构和语法，将代码分割成有意义的文本块。以下是一些可以使用它的场景：

1. **代码分析与理解**：当你处理大量代码时，`CodeTextSplitter` 可以帮助你将代码分解成易于管理的部分，以便更好地分析和理解代码。

2. **代码搜索与检索**：如果你需要快速找到特定的代码片段或函数，可以使用 `CodeTextSplitter` 创建代码块的索引。然后，你可以根据关键词或条件迅速找到相关代码。

3. **代码摘要与文档生成**：通过 `CodeTextSplitter` 将代码分割成逻辑单元，然后创建有针对性的文档，可以轻松地为代码生成文档。这将帮助你和其他人理解代码库中不同部分的目的和用法。

4. **代码比较与差异分析**：你可以使用 `CodeTextSplitter` 将代码分割成可比较的块，并识别版本之间的变化，从而轻松比较代码的不同版本。

5. **代码格式化与风格检查**：你可以将 `CodeTextSplitter` 用作格式化和风格检查工具的预处理步骤。它可以帮助你将格式化规则和风格指南应用于代码的特定部分，并确保整个代码库的可读性。

以下是一个如何使用 `CodeTextSplitter` 分割 Python 代码片段的示例：

```python
from langchain_text_splitters import CodeTextSplitter

code_snippet = '''

def greet(name):

print(f"Hello, {name}!")

def main():

name = input("Enter your name: ")

greet(name)

if __name__ == "__main__":

main()

'''

code_splitter = CodeTextSplitter(language="python", chunk_size=50, chunk_overlap=0)

code_chunks = code_splitter.create_documents([code_snippet])

for chunk in code_chunks:

print(chunk.page_content)

print("---")
```

以下是输出结果：

```
def greet(name):

print(f"Hello, {name}!")

def main():

name = input("Enter your name: ")

greet(name)

if __name__ == "__main__":

main()
```

如你所见，`CodeTextSplitter` 根据函数和代码块将 Python 代码片段分割成逻辑单元。然后，你可以更轻松地以较小的片段来分析、理解和处理代码。

### 按 Token 分割

在实际应用中，有时你可能需要根据所用大语言模型的 Token 限制来分割文本。假设你需要处理公司知识库中的文章，并将其输入到公司聊天机器人的训练流程中。但是，你需要确保每个文本块都符合语言模型的 Token 限制，以避免训练过程中出现问题。

首先，你必须安装必要的 `text_splitters` 和 `tiktoken` 包：

```bash
pip install --upgrade --quiet langchain-text_splitters tiktoken
```

假设你有一个名为 `returns_policy.txt` 的知识库文章文件。你可以按照如下所示的方式，在关注 Token 限制的同时将其分割成文本块：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载知识库文章

with open("returns_policy.txt") as f:

returns_policy = f.read()

# 创建一个 RecursiveCharacterTextSplitter 实例

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(

model_name="gpt-3.5-turbo", # 指定你正在使用的语言模型

chunk_size=500, # 设置期望的文本块大小（以 Token 为单位）

chunk_overlap=50, # 允许文本块之间有一些重叠以保持上下文

)

# 将文章分割成文本块

chunks = text_splitter.split_text(returns_policy)

# 打印生成的文本块数量

print(f"文本块数量: {len(chunks)}")
```