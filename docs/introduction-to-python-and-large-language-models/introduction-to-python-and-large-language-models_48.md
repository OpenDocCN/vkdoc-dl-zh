# Cohere

Cohere 成立于 2019 年，总部横跨多伦多和旧金山，并在帕洛阿尔托和伦敦设有办事处，是一家专注于为企业提供人工智能解决方案的全球性科技企业，尤其致力于开发先进的大型语言模型。公司的创始人是 `Aidan Gomez`、`Ivan Zhang` 和 `Nick Frosst`，他们都拥有多伦多大学的学术背景。

Cohere 的根源可以追溯到 2017 年人工智能研究的一个变革性时刻。当时，Gomez 作为 Google Brain 团队的成员，合著了开创性论文《注意力就是一切》，该论文引入了彻底改变自然语言处理任务的 Transformer 模型。这项开创性的工作为 Gomez、Frosst 和 Zhang 创立 Cohere 奠定了基础，他们此前曾在 `FOR.ai` 共事。

自成立以来，Gomez 一直担任 Cohere 的首席执行官。公司在 2022 年底任命 YouTube 前首席财务官 Martin Kon 为总裁兼首席运营官，这标志着一个重要的里程碑。Cohere 的技术进步获得了大力支持，包括 2021 年与 Google Cloud 建立战略合作伙伴关系，以利用 Google 的基础设施和 TPU 进行产品开发。

Cohere 对创新的承诺在 2022 年 6 月成立的 Cohere For AI 中得到了进一步体现。这是一项非营利性研究计划，旨在在前 `Google Brain` 成员 Sara Hooker 的领导下推进开源机器学习研究。该公司还在语言处理方面取得了进展，开发了一个能够理解超过 100 种语言的多语言模型，这是使非英语语言处理更易普及的一项重大成就。

2023 年，与 Oracle、McKinsey 和 LivePerson 的合作扩大了 Cohere 的业务范围，为企业提供生成式 AI 服务和定制语言模型，提高了各行业的自动化和运营效率。此外，Cohere 积极践行合乎道德的 AI 实践，遵守白宫和加拿大为 AI 开发和管理制定的自愿措施。

作为 OpenAI 的竞争对手，Cohere 的技术套件服务于寻求将 AI 应用于从聊天机器人和搜索引擎到内容审核和数据分析等各种场景的企业。该平台的 API 便于与主要云服务集成，确保了其多功能性和广泛的应用。

## Cohere 模型

Cohere 提供多种多样的模型，以满足广泛的需求。对于寻求更定制化解决方案的用户，可以选择自定义训练模型，使其精确符合特定要求。

### Command

Command 模型是 Cohere 的主要生成工具，旨在解释和执行用户的文本命令或提示。该模型不仅能根据指令生成文本，还具备对话能力，非常适合驱动基于聊天的应用程序。

### Embed

Embed 模型提供生成文本嵌入或根据一组标准对文本进行分类的功能。这些嵌入可用于多种任务，例如衡量句子之间的语义相似度、选择最可能接续另一个句子的句子，或将用户反馈分类。

此外，Embed 模型中的 Classify 功能支持各种分类或分析任务。Representation 模型通过额外的支持功能（包括输入的语言检测）增强了这些能力。

### Rerank

最后，Rerank 模型旨在通过根据特定标准重新排序现有模型的结果，来优化和改进其输出。此功能对于提高搜索算法的效率特别有用。

#### 情感分析示例应用

在 Cohere 网站上注册并获取您的 API 密钥，然后在您的笔记本或终端中使用以下命令安装 Cohere SDK：`pip install cohere`，然后使用以下代码：

```python
import cohere
from cohere.responses.classify import Example
co = cohere.Client('GHviIR5p9NC7kNzRf383ykOxU2Y9LQVbSAvAdNSj')
examples=[
Example("Dermatologists don't like her!", "Spam"),
Example("'Hello, open to this?'", "Spam"),
Example("I need help please wire me $1000 right now", "Spam"),
Example("Nice to know you ;)", "Spam"),
Example("Please help me?", "Spam"),
Example("Your parcel will be delivered today", "Not spam"),
Example("Review changes to our Terms and Conditions", "Not spam"),
Example("Weekly sync notes", "Not spam"),
Example("'Re: Follow up from today's meeting'", "Not spam"),
Example("Pre-read for tomorrow", "Not spam"),
]
user_input = input()
inputs=[
user_input
]
response = co.classify(
model = 'large',
inputs=inputs,
examples=examples,
)
print(response.classifications)
```

**输出：**

```
Looking forward to your email
[Classification]
```

这段 Python 代码使用 `cohere` 库执行情感分析。以下是代码功能的分解说明：

1.  **导入必要的模块：**
    -   `cohere`：用于与 Cohere API 交互的主库。
    -   来自 `cohere.responses.classify` 的 `Example`：用于定义带有相应标签的示例输入，以训练模型。

2.  **创建一个 `cohere.Client` 对象：**
    -   创建一个客户端对象以与 Cohere API 交互。它需要一个 API 密钥进行身份验证。

3.  **定义一组带有相应标签的示例输入：**
    -   每个示例都是来自 `cohere.responses.classify` 模块的 `Example` 类的一个实例。
    -   这些示例用于训练分类模型。它们被标记为 `Spam` 或 `Not spam`。

4.  **提示用户输入：**
    -   `input()` 函数提示用户输入一些文本，这些文本将被分类为 `Spam` 或 `Not spam`。

5.  **创建一个包含用户输入的列表：**
    -   用户的输入存储在一个名为 `inputs` 的列表中。

6.  **使用 `co.classify()` 方法对用户输入进行分类：**
    -   `co.classify()` 方法接受几个参数：
    -   `model`：指定用于分类的模型。此处设置为 `large`。
    -   `inputs`：要分类的输入列表。在这种情况下，它只包含用户的输入。
    -   `examples`：用于训练模型的示例输入及其标签列表。
    -   此方法返回一个响应对象。

7.  **打印分类结果：**
    -   `response.classifications` 属性包含输入文本的分类结果。
    -   这将打印出分类结果，指示输入被归类为 `Spam` 还是 `Not spam`。

