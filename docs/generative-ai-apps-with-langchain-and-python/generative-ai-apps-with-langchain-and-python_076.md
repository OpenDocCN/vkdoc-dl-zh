# 第 5 章 掌握创意内容的提示词技巧

## 示例

```python
examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivaderci"},
    {"input": "soccer", "output": "calcio"},
]
```

### 实现自定义示例选择器

接下来，你将开发一个`BaseExampleSelector`类的自定义实现。

使用`__init__`方法，用示例列表初始化示例选择器。

通过`add_example`方法，你可以向示例选择器的示例列表中添加新示例。

`select_examples`方法会选择与输入单词长度最接近的示例。它会遍历每个示例，计算输入单词与示例输入之间的长度差，并记录长度差最小的示例。

```python
class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        new_word = input_variables["input"]
        new_word_length = len(new_word)
        best_match = None
        smallest_diff = float("inf")
        for example in self.examples:
            current_diff = abs(len(example["input"]) - new_word_length)
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example
        return [best_match]
```

### 使用自定义示例选择器

这里，你将使用示例列表创建一个`CustomExampleSelector`实例：

```python
example_selector = CustomExampleSelector(examples)
```

然后，你调用`select_examples`方法，传入输入单词`"okay"`，以获取最匹配的示例：

```python
example_selector.select_examples({"input": "okay"})
```

输出结果为`[{'input': 'bye', 'output': 'arrivaderci'}]`。

接着，你使用`add_example`方法添加一个新示例：

```python
example_selector.add_example({"input": "hand", "output": "mano"})
```

你再次用相同的输入单词调用`select_examples`，查看更新后的结果：

```python
example_selector.select_examples({"input": "okay"})
```

现在，修改后的输出为`[{'input': 'hand', 'output': 'mano'}]`。

### 在提示词中使用示例选择器

以下是在提示词中使用示例选择器的代码。这里，你从 LangChain 导入必要的类来创建提示词模板：

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
```

你使用`PromptTemplate.from_template`方法定义一个`example_prompt`，它指定了提示词中每个示例的格式：

```python
example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")
```

你使用自定义示例选择器、示例提示词、后缀、前缀和输入变量创建一个`FewShotPromptTemplate`：

```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italian:",
    input_variables=["input"],
)
```

最后，你使用提示词模板的`format`方法，生成一个包含输入单词`"word"`的提示词：

```python
print(prompt.format(input="word"))
```

现在，你已经了解了如何创建一个自定义示例选择器，该选择器根据示例与输入单词在长度上的相似性进行选择。然后，该示例选择器被用于提示词模板中，以生成用于将英语单词翻译成意大利语的提示词。你应该能够使用相同的方法，为类似的用例创建自定义示例选择器。

### 选择合适的示例选择器

你可以选择不同类型的示例选择器：

#### 1. 相似度示例选择器

相似度示例选择器利用输入与可用示例之间的语义相似性，在提示词中选择相似度得分最高的示例。当你希望向语言模型提供与输入语义相近的示例，以生成更相关的输出时，这非常有用。

#### 2. MMR（最大边际相关性）示例选择器



## 示例选择器

## MMR 示例选择器

`MMR` 示例选择器旨在平衡所选示例的相关性和多样性。它使用了最大边际相关性的概念，同时考虑了输入与示例之间的相似性以及所选示例之间的差异性。这种方法确保所选示例不仅与输入相关，而且具有多样性，从而减少冗余并增加对输入相关不同方面的覆盖。

## 长度示例选择器

`Length` 示例选择器专注于根据指定长度约束内能容纳的示例数量来选择示例。当您的上下文窗口有限或想要控制提示的大小时，此示例选择器非常有用。这有助于在遵守长度约束的同时，为语言模型提供多样化的示例集。

## Ngram 示例选择器

`Ngram` 示例选择器利用输入与示例之间的 ngram 重叠来决定选择哪些示例。ngram 是来自给定文本的 n 个连续项（单词或字符）的序列。通过考虑 ngram 重叠，该选择器倾向于选择与输入共享常见短语或单词序列的示例，从而可能提高模型生成相关输出的能力。

### 选择示例选择器时的考量因素

在选择示例选择器时，请考虑以下因素：

- **相关性**：如果生成高度相关的输出至关重要，则`Similarity`或`MMR`示例选择器可能是合适的选择。
- **多样性**：如果您希望覆盖与输入相关的不同方面，`MMR`示例选择器可以帮助平衡相关性和多样性。
- **长度约束**：如果您的提示有严格的长度限制，`Length`示例选择器可以帮助选择符合这些约束的示例。
- **短语匹配**：如果您希望强调与输入共享常见短语或单词序列的示例，`Ngram`示例选择器会很有效。

## 少样本提示模板

少样本提示模板是一种技术，模型通过极有限的训练数据来学习执行任务。它涉及提供少量示例（样本）来帮助模型理解任务上下文和期望的输出格式。图 5-1 展示了少样本学习的工作原理。

***图 5-1.** 少样本学习的工作原理*

`LangChain` 提供了一个方便的 `FewShotPromptTemplate` 类，使得处理示例更加容易。

### 为问答任务构建少样本提示模板

在本节中，您将学习如何创建一个少样本提示模板，该模板教导语言模型根据一组示例生成自问问题并搜索答案。

#### 第 1 步：准备示例集

首先，创建一个少样本示例列表。每个示例应是一个包含问题及其对应答案的字典：

```python
examples = [
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter is the largest planet in our solar system."
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "The Mona Lisa was painted by Leonardo da Vinci."
    },
    {
        "question": "What is the currency of Japan?",
        "answer": "The currency of Japan is the Japanese yen."
    }
]
```

#### 第 2 步：格式化少样本示例

接下来，使用 `PromptTemplate` 将示例格式化为将呈现给语言模型的字符串。它将问题和答案作为输入变量，并将其格式化为字符串：

```python
from langchain.prompts import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}"
)

# Test the formatting by printing the first example
print(example_prompt.format(**examples[0]))
```

**输出：**

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.
```

#### 第 3 步：创建少样本提示模板

现在，您将创建一个 `FewShotPromptTemplate`，它将作为语言模型从提供的示例中学习的框架。

此代码使用示例和格式化后的示例提示创建了一个 `FewShotPromptTemplate`。它还指定了一个将附加到示例和输入变量之后的后缀：

```python
from langchain.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

# Preview the formatted template
print(prompt.format(input="What is the capital of Australia?"))
```

**输出：**

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.

Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.

Question: What is the currency of Japan?
Answer: The currency of Japan is the Japanese yen.

Question: What is the capital of Australia?
```

#### 第 4 步（可选）：使用示例选择器选择示例

如果您有大量示例，可以使用示例选择器。此代码初始化了一个 `LengthBasedExampleSelector`，它根据示例的长度进行选择。它接受示例、示例提示和最大长度作为参数。您可以选择自己偏好的选择器。

```python
# Initialize the selector with your examples
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50
)

# Find the most relevant example for a new question
selected_examples = example_selector.select_examples(
    {"question": "Who sculpted the Statue of David?"}
)
```

请记得导入以下内容：

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
```

上面这行代码使用 `LengthBasedExampleSelector`，根据问题“Who sculpted the Statue of David?”选择了最相关的示例。然后，所选示例被用于创建一个新的 `FewShotPromptTemplate`，该模板整合了下面的示例选择器。

**让我们讨论一下 `LengthBasedExampleSelector` 的工作原理。**

在初始化 `LengthBasedExampleSelector` 时，您需要提供示例列表（`examples`）、示例提示模板（`example_prompt`）以及所选示例的最大长度（`max_length`）。

调用 `select_examples` 方法时，会传入一个包含问题“Who sculpted the Statue of David?”的输入字典。然而，对于 `LengthBasedExampleSelector` 来说，输入问题并不用于基于相关性选择示例。

`LengthBasedExampleSelector` 会遍历示例列表，并根据示例的长度进行选择。它尝试包含尽可能多的示例，同时确保所选示例的总长度不超过指定的 `max_length`。其目标是在指定的长度限制内最大化包含的示例数量。

#### 第 5 步（可选）：将示例选择器集成到提示模板中

通过将示例选择器集成到提示模板中，所选示例会根据与输入问题的相关性动态地包含在提示中。这种方法允许更有针对性和更高效地使用示例，从而提高生成回复的质量和相关性。

```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)
```



