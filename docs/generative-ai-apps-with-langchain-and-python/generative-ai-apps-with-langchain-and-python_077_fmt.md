# 加载 OpenAI 语言模型（你需要自己的 API 密钥）

`llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key="your openai key", temperature=0.7)`

## 测试提示模板

下面，你将测试带示例选择器和不带示例选择器的提示模板。你根据给定的输入格式化提示，并使用加载的语言模型生成答案：

```python
print("不带示例选择器的提示模板：")
print(prompt.format(input="What is the capital of Australia?"))
print("\n 生成的答案：")
print(llm(prompt.format(input="What is the capital of Australia?")))
```

```python
print("\n 带示例选择器的提示模板：")
print(prompt_with_selector.format(input="Who sculpted the Statue of David?"))
print("\n 生成的答案：")
print(llm(prompt_with_selector.format(input="Who sculpted the Statue of David?")))
```

以下是我得到的答案：

**不带示例选择器的提示模板：**

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.
Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.
Question: What is the currency of Japan?
Answer: The currency of Japan is the Japanese yen.
Question: What is the capital of Australia?
Answer:
```

**生成的答案：**

```
The capital of Australia is Canberra.
```

**带示例选择器的提示模板：**

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.
Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.
Question: Who sculpted the Statue of David?
Answer:
```

**生成的答案：**

```
The Statue of David was sculpted by Michelangelo.
```

## 审查输出结果

上述答案清晰地展示了带示例选择器和不带示例选择器的少样本提示模板的功能。我们来逐一分析每个部分：

### 1. 不带示例选择器的提示模板

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.
Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.
Question: What is the currency of Japan?
Answer: The currency of Japan is the Japanese yen.
Question: What is the capital of Australia?
Answer:
```

在这种情况下，不带示例选择器的提示模板包含了 `examples` 列表中提供的所有示例。这些示例按照 `example_prompt` 模板进行格式化，其中包含每个示例的问题和答案。在示例之后，提示模板使用 `suffix` 参数附加了输入问题“What is the capital of Australia?”。然后由语言模型生成答案。

### 2. 生成的答案（不带示例选择器）

```
The capital of Australia is Canberra.
```

这是语言模型根据不带示例选择器的提示模板，针对问题“What is the capital of Australia?”生成的答案。模型利用提供的示例及其已有的知识生成了正确答案。

### 3. 带示例选择器的提示模板

```
Question: What is the largest planet in our solar system?
Answer: Jupiter is the largest planet in our solar system.
Question: Who painted the Mona Lisa?
Answer: The Mona Lisa was painted by Leonardo da Vinci.
Question: Who sculpted the Statue of David?
Answer:
```

在这种情况下，带示例选择器（`LengthBasedExampleSelector`）的提示模板会根据示例的长度进行选择。选中的示例按照 `example_prompt` 模板进行格式化。输入问题“Who sculpted the Statue of David?”通过 `suffix` 参数附加到选中的示例之后。然后由语言模型生成答案。

### 4. 生成的答案（带示例选择器）

```
The Statue of David was sculpted by Michelangelo.
```

这是语言模型根据带示例选择器的提示模板，针对问题“Who sculpted the Statue of David?”生成的答案。