# 打印加载的文档

`for doc in documents:`

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

`print(f"Content: {doc.page_content}\n")`

`print(f"Metadata: {doc.metadata}\n")`

`print("---")`

以下是我得到的输出结果。

```
Content: this is just a test file
Metadata: {'source': '/media/file1.txt'}
```

是不是很神奇？仅用三行代码，你就成功将 `file1.txt` 的内容加载到了 `Document` 对象列表中。就是这么简单！

现在，我们来分解一下：

1.  你从 `langchain_community.document_loaders` 模块中导入了 `TextLoader` 类，用于处理纯文本文件。

2.  你通过传递文件路径（`"/media/file1.txt"`）作为参数，创建了一个 `TextLoader` 实例。这告诉加载器去哪里寻找你的宝贵数据。

3.  最后，你在加载器实例上调用了 `load()` 方法，该方法会读取文件并返回一个 `Document` 对象列表。每个 `Document` 都包含文本内容及与之关联的元数据。

## 亲自尝试

不要犹豫，大胆尝试不同的加载器。每个加载器都有其独特的功能和特性，找到最适合你需求的即可。

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

### 处理 PDF 文件

现在，让我们来看看最常见的数据格式之一：PDF 文件。

我们的大部分数据都以各种格式存储，如 PDF、CSV 文件、JSON、HTML，甚至是办公文档。如果你能基于这些文档的内容提问并获得答案，那该多好啊？这正是你将要实现的目标！

首先，确保你已经安装了必要的库。在本例中，你将使用 `pypdf` 库来处理 PDF 文件。你可以通过运行下面的代码轻松安装它。`pypdf` 库是一个纯 Python 库，用于解析 PDF 文档，LangChain 的 `PyPDFLoader` 使用它来读取和提取 PDF 文件中的文本：

```
!pip install pypdf langchain langchain_community
```

接下来，导入必要的模块，例如从 `langchain_community.document_loaders` 模块中导入 `PyPDFLoader` 类。`langchain_community` 包是主 `langchain` 库的社区驱动扩展，提供了额外的功能和工具：

```
from langchain_community.document_loaders import PyPDFLoader
```

然后，创建一个 `PyPDFLoader` 类的实例，并将 PDF 文件的路径作为参数传入。在此示例中，PDF 文件位于 `"/media/2022 Annual Report ACME.pdf"`。请确保你提供了正确的 PDF 文件路径：

`loader = PyPDFLoader("/media/2022 Annual Report ACME.pdf")`

接下来，你应该调用 `PyPDFLoader` 实例的 `load_and_split()` 方法来加载 PDF 文档，并自动将其拆分为单独的页面或章节。生成的页面或章节将作为 `Document` 对象列表存储在 `pages` 变量中：

`pages = loader.load_and_split()`

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

`pages` 变量现在包含一个 `Document` 对象列表，每个对象代表 PDF 文档的一个页面或章节。你可以使用 `Document` 对象的属性（如 `page_content` 和 `metadata`）来访问每个页面的文本内容和元数据：

`print(pages[0].page_content)`

这将显示 PDF 文档第一页或第一节的文本内容。

你可以遍历 `pages` 列表，根据需要访问和处理 PDF 文档的每个页面或章节。

但是，如果你想访问某个 `Document` 的元数据呢？没问题！每个 `Document` 对象都有一个 `metadata` 属性，它存储了一个包含元数据信息的字典。例如，要查看第 11 页的元数据：

`print(data[10].metadata)`

如果你想知道加载的 PDF 文件有多少页，可以使用 `len()` 函数轻松获知：

`print(f"{len(pages)} pages in your data")`

如果你想了解特定页面的字符数，可以这样做：

### 处理 CSV 文件

让我们看看如何使用 `langchain_community.document_loaders` 模块中的 `CSVLoader` 类来加载 CSV 文件。

在 CSV 文件中，每一行代表一条记录，每条记录中的值用逗号分隔。这是一种组织和共享数据的简单方式。

你将使用 `CSVLoader` 来加载 CSV 文件，并将每一行转换为一个 `Document` 对象。

首先，请确保你已经安装了 `langchain_community` 包。你可以使用 `pip` 来安装它：

```
! pip install langchain_community
```

安装好包之后，你就可以导入 `CSVLoader` 类并开始加载你的 CSV 文件了。下面是一个示例：

```
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./sample_data/california_housing_test.csv')
data = loader.load()
print(data)
```

在这里，你创建了一个 `CSVLoader` 类的实例，并将 CSV 文件的路径传递给 `file_path` 参数。然后，你调用 `load()` 方法来加载 CSV 数据。加载后的数据存储在 `data` 变量中，它包含一个 `Document` 对象列表，每个对象代表 CSV 文件中的一行。

当你打印 `data` 变量时，你会看到加载的文档，包括它们的内容和元数据。元数据包含诸如源文件路径和行号等信息。

`CSVLoader` 允许你自定义 CSV 文件的解析方式。你可以向 `csv_args` 参数传递额外的参数来控制分隔符、引号字符和字段名称。例如：

```
loader = CSVLoader(
    file_path='./example_data/mlb_teams_2012.csv',
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
    }
)
data = loader.load()
print(data)
```

在这个例子中，我们指定分隔符为逗号，引号字符为双引号，并为各列提供了自定义字段名称。这样，你可以确保 CSV 文件被正确解析，并且数据按预期加载。

以下是你会得到的结果：

```
[Document(page_content='MLB Team: Team\nPayroll in millions: "Payroll (millions)"\nWins: "Wins"', metadata={'source': './sample_data/mlb_teams_2012.csv', 'row': 0}),
Document(page_content='MLB Team: Nationals\nPayroll in millions: 81.34\nWins: 98', metadata={'source': './sample_data/mlb_teams_2012.csv', 'row': 1}),
.....
```

`CSVLoader` 的另一个很酷的功能是指定源列。默认情况下，所有文档都使用文件路径作为源。但是，如果你想使用 CSV 文件中的特定列作为源，可以使用 `source_column` 参数：

```
loader = CSVLoader(
    file_path='./example_data/mlb_teams_2012.csv',
    source_column="Team"
)
data = loader.load()
print(data)
```

在这个例子中，你将 `source_column` 设置为 "Team"，这意味着 "Team" 列的值将被用作每个文档的源。这在处理使用源来回答问题的链时特别有用。

```
[Document(page_content='Team: Nationals\n"Payroll (millions)": 81.34\n"Wins": 98', metadata={'source': 'Nationals', 'row': 0}),
Document(page_content='Team: Reds\n"Payroll (millions)": 82.20\n"Wins": 97', metadata={'source': 'Reds', 'row': 1}),
.....
```

### 处理 JSON 文件

现在，让我们看看如何操作。为了有效处理 JSON 文件，你可以使用 Python 内置的 `json` 模块以及 `pathlib` 来管理文件路径。

首先，确保你已经安装了必要的依赖。然后，你需要使用下面的导入语句从文件中加载 JSON 数据。

在这段代码片段中，你使用 `pathlib` 来处理文件路径，并将 JSON 文件的内容读取为字符串。然后，你使用 `json.loads()` 将该字符串解析为 Python 字典：

```
import json
from pathlib import Path

file_path = './sample_data/products.json'
data = json.loads(Path(file_path).read_text())
```

请查看下载部分资源中的 `products.json` 文件。

在这里，你使用 `json` 模块从文件中加载 JSON 数据。你通过指定文件路径并使用 `Path(file_path).read_text()` 将文件内容读取为字符串来实现这一点。然后，你将那个字符串传递给 `json.loads()` 来将其解析为 Python 字典。

#### 使用 JSONLoader 提取特定数据

```
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path='./sample_data/products.json',
    jq_schema='.messages[].content',
    text_content=False
)
data = loader.load()
```

你会注意到 `JSONLoader` 是一个方便的工具，它允许你使用 `jq` 模式从 JSON 文件中提取特定数据。在这个例子中，你指定了文件路径，并提供了一个 `jq` 模式 (`.messages[].content`) 来提取 JSON 数据中 `messages` 键下 `content` 字段的值。`text_content=False` 参数表示你不想将整个 JSON 文件作为文本内容加载。

以下是输出结果：

让我们回顾一下下面处理 JSON Lines 文件的代码：

```
loader = JSONLoader(
    file_path='./sample_data/products.jsonl',
    jq_schema='.content',
    text_content=False,
    json_lines=True
)
```

如果你正在处理一个 JSON Lines 文件（其中每一行代表一个有效的 JSON 对象），你可以在 `JSONLoader` 构造函数中设置 `json_lines=True`。这告诉加载器将每一行视为一个独立的 JSON 对象。然后，你可以指定 `jq_schema` 来从每个 JSON 对象中提取所需的数据。

以下是输出结果：

## 文本分割器

我们已经讨论过，在处理大型文档时，我们可能需要转换文档以更好地适应应用程序。例如，当你有一个超出模型上下文窗口的长文档时，你需要将其分割成更小、语义上有意义的块。这就是文本分割器的用武之地，它们将相关的文本片段保持在一起，从而确保模型能够理解上下文并提供准确的结果。

你可以根据具体需求从多种文本分割器中进行选择。每个分割器都以自己的方式分割文本，有些甚至会添加元数据来提供关于这些块的额外信息。以下是一些流行的选项：

1.  **RecursiveCharacterTextSplitter**：你可以使用这个分割器，根据你定义的一系列字符递归地切分文本。它有助于将相关的文本片段保持在一起。

2.  **HtmlTextSplitter**：当你处理 HTML 文档时，这个分割器将是你的首选。它根据 HTML 特定的字符分割文本，并添加关于每个块来源的元数据。

3.  **MarkdownTextSplitter**：这与 HTML 分割器类似，但专为 Markdown 文档设计。它根据 Markdown 特定的字符分割文本，并包含关于块来源的元数据。

4.  **TokenTextSplitter**：当你需要根据 token 数量分割文本时，你会用到这个。你可以通过使用不同的 token 度量方式来决定如何对文本进行分块。

### 文本分割的完整工作代码示例

首先，你需要安装 `langchain-text-splitters` 包：

```
pip install langchain-text-splitters
```

安装好包之后，你可以轻松创建一个文本分割器实例，并开始对文本进行分块。以下是使用 `CharacterTextSplitter` 的示例：

```python
# 这是一份可以分割的长文档。
with open("./sample_data/The Art of Money Getting.txt") as f:
    art_of_money_getting = f.read()
```

在上述代码中，你打开了一个名为“The Art of Money Getting.txt”的文本文件，该文件位于 `./sample_data` 目录下。你使用 `with` 语句来确保在读取完文件后，文件能被正确关闭。

你使用 `read()` 方法读取文件的全部内容，并将其存储在 `art_of_money_getting` 变量中。该变量现在保存了文档的完整文本。

```python
from langchain_text_splitters import CharacterTextSplitter
```

在这里，你将从 `langchain_text_splitters` 包中导入 `CharacterTextSplitter` 类。你使用这个类来根据指定的一个或多个字符将文本分割成更小的块。

```python
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
```

在这段代码中，你创建了一个 `CharacterTextSplitter` 类的实例，并使用以下参数对其进行配置：

- `separator="\n\n"`：你将分隔符指定为两个换行符（`\n\n`）。这意味着每当遇到两个连续的换行符时，文本就会被分割。

- `chunk_size=1000`：你将每个块的最大大小设置为 1000 个字符。如果一个块超过此大小，它将被进一步分割。

- `chunk_overlap=200`：你指定了连续块之间 200 个字符的重叠。这种重叠有助于保持块之间的上下文连贯。

- `length_function=len`：你使用内置的 `len` 函数来计算每个块的字符长度。

- `is_separator_regex=False`：你指明分隔符不是正则表达式，而是一个简单的字符串。

```python
documents = text_splitter.create_documents([art_of_money_getting])
```

在这里，你使用 `text_splitter` 实例的 `create_documents` 方法将 `art_of_money_getting` 文本分割成更小的块。该方法接受一个文本列表作为输入（在此例中，你提供了一个文本），并返回一个 `Document` 对象列表。每个 `Document` 对象代表一个文本块。

```python
print(documents[4])
```

最后，你打印 `documents` 列表中的第五个 `Document` 对象（索引为 4）。这将显示第五个文本块的内容和元数据。

输出结果大致如下：

```
page_content='...' metadata={}
```

`page_content` 属性包含该块的实际文本内容，而 `metadata` 属性是一个空字典，因为在此示例中我们没有提供任何元数据。

总的来说，这段代码演示了如何使用 `langchain_text_splitters` 包中的 `CharacterTextSplitter`，根据指定的分隔符（本例中为两个换行符）将长文档分割成更小的块。生成的结果块作为 `Document` 对象存储在 `documents` 列表中，你可以根据需要访问和操作这些对象。

为了测试你的文本分割器的工作效果，你可以使用 Greg Kamradt 创建的 `Chunkviz` 工具，该工具可以让你直观地看到文本是如何被分割的。通过使用它，你可以微调你的分割参数。

请注意，文本分割只是你在将文本输入给大语言模型（LLM）之前，可以对文档进行的转换操作之一。你可以随意从 LangChain 提供的与第三方工具集成的各种文档转换器中进行选择。

## 自己动手试试

要实现成功的文本分割，你必须在块大小和语义连贯性之间找到恰当的平衡。我鼓励你尝试不同的分割器和参数，以找到最适合你特定用例的方案。

### 递归分割

接下来是递归分割。`RecursiveCharacterTextSplitter` 同样根据指定的一个或多个分隔符字符来分割文本。

然而，它采用递归方法将文本分割成块。首先，它根据分隔符对整个文本进行分割，并创建初始块。如果任何生成的块超过了指定的 `chunk_size`，分割器会递归地对这些块应用分割过程。这种递归分割会一直持续，直到所有块都在所需的 `chunk_size` 限制之内。`chunk_overlap` 参数用于保持递归分割的块之间的上下文连贯。

这种递归方法确保了块的分割更加均匀，尤其是在处理超过 `chunk_size` 的长段落或章节时。

`CharacterTextSplitter` 和 `RecursiveCharacterTextSplitter` 之间的区别很微妙，当你希望将文本分割成更小、更易于管理的块，同时保留内容的逻辑结构时，你会发现后者特别有用。

以下是一个示例，用于说明两者的区别：

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = "This is a sample text. It consists of multiple sentences. Some sentences are longer than others. We will split this text into chunks."

# CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator=". ", chunk_size=30, chunk_overlap=5)
documents = text_splitter.create_documents([text])
print("CharacterTextSplitter:")
for doc in documents:
    print(doc.page_content)
    print("---")
```