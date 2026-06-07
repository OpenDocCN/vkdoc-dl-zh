# 提取并打印生成的分析结果

`generated_analysis = response.choices[0].message.content`

`print("Market Analysis:")`

`print(generated_analysis)`

### Codex：你的 AI 编程助手

接下来，你决定探索 OpenAI 的 Codex，它可以帮助你编写代码。

要开始使用 Codex，你需要将其集成到你最喜欢的代码编辑器中。

访问 [`openai.com/index/openai-codex/`](https://openai.com/index/openai-codex/) 并点击“开始使用 Codex”，然后点击“尝试 GPT-4o”。请注意，这些功能在不断演进，当你阅读本书时，这些链接可能已失效，但这些功能仍然存在并且已经进化。

## 第 4 章：探索大型语言模型（LLM）

以下是一个示例 Python 代码，由 Codex 根据提示生成，用于实现两个数相加的函数：

```python
# 让 Codex 生成一个两个数相加的函数
def add_numbers(a, b):
    """
    将两个数相加
    """
    return a + b
```

上面的示例展示了如何使用 Codex 加速你的编码工作流程，并获得即时建议来完成代码。它可以协助你完成以下任务：

-   根据注释或文档字符串完成函数实现

-   为常见编程模式生成样板代码

-   提供优化或重构代码的建议

### DALL-E 2：图像生成大师

接着，你决定使用 OpenAI 的 DALL-E 2 将你的想法生动地呈现为图像。你只需提供一段文字描述，DALL-E 2 就会根据描述生成图像。想象一下各种可能性，例如，你可以仅通过文字描述来创建独特的插图、设计元素或视觉概念。

以下是一个简单示例，展示了如何使用 OpenAI API 根据你的提示生成令人惊叹的图像：

```bash
# 安装 openai 包
!pip install openai==0.28
```

## 第 4 章：探索大型语言模型（LLM）

```python
# 代码从这里开始
import openai

openai.api_key = "YOUR_API_KEY"

def generate_image(prompt, num_images=1, size="1024x1024"):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=num_images,
            size=size
        )
        image_urls = [data['url'] for data in response['data']]
        return image_urls
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
```