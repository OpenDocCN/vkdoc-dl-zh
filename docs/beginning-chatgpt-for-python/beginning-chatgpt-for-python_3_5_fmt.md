# 8. 为 Discord 机器人注入智能，第二部分：使用聊天与审核模型进行内容审核

在本章中，我们将采取必要步骤，使我们的内容审核 Discord 机器人具备人工智能。让我们概述一下将要进行的更改：

*   创建一个新脚本 `moderation_client.py`，用于调用 OpenAI API 的 `OpenAI.moderations.create()` 方法。审核模型使我们能够识别任何文本内容是否属于以下类别：
    *   仇恨言论
    *   仇恨/威胁性言论
    *   骚扰
    *   骚扰/威胁性言论
    *   自残
    *   自残/意图
    *   自残/指导
    *   色情内容
    *   色情内容/涉及未成年人
    *   暴力内容
    *   暴力内容/血腥画面

*   复用上一章中的 `chatgpt_client_for_qa_and_moderation.py` 脚本。在第 7 章中，该脚本用于调用 OpenAI 的 Chat 类，以回答用户提出的问题。在本章中，它将再次用于调用 Chat 类，但这次是为了审核目的。这就是为什么该脚本被恰当地命名为 `chatgpt_client_for_qa_and_moderation`，因为它既用于第 7 章的问答，也用于本章的审核。

*   修改我们的 `content_moderator_bot.py` 脚本（原名为 `content_moderator_bot_dumb.py`），使其能够同时调用 `moderation_client.py` 和 `chatgpt_client_for_qa_and_moderation.py`。如果任一脚本指示 Discord 频道中输入的内容存在异议，它将从该 Discord 频道中删除该消息。请记住，此机器人会监控 Discord 服务器中所有频道的所有内容！

现在，你可能会问自己，既然审核类已经知道如何标记任何有害内容，为什么我们还需要使用聊天类呢？问得好。

是的，审核类能让我们了解有害内容，但它**不会**告知我们其他类型的不当内容（针对我们的用例而言），例如当不怀好意的人试图引诱我们的用户陷入骗局时。请记住，这是一个银行应用的 Discord 服务器，因此诈骗者肯定会非常乐意针对该服务器的所有成员，因为这里聚集了大量银行用户！

因此，我们将使用 `moderation_client.py` 调用审核类，以了解 Discord 服务器中是否存在任何有害内容；同时，我们将复用上一章的 `chatgpt_client_for_qa_and_moderation.py` 来调用聊天方法，以便了解 Discord 服务器中是否发布了任何其他不良内容，例如诈骗企图。

## 使用 `OpenAI.moderations.create()` 审核内容

通过使用审核模型，开发者可以提交一段文本，并随后了解该文本是否包含暴力、仇恨和/或威胁性内容，或任何形式的骚扰。

表 8-1 描述了调用 `OpenAI.moderations.create()` 方法所需参数的格式。该服务使用起来非常简单，因为只需一个参数即可正确调用该服务。

### 检查方法参数

**表 8-1** 审核模型的请求体

| 字段 | 类型 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `input` | 字符串或列表 | 必需 | 需要分类的文本。 |
| `model` | 字符串默认值：`omni-moderation-latest` | 可选 | 有多个内容审核模型可供使用，例如：`omni-moderation-latest`、`text-moderation-stable`、`text-moderation-latest`。默认情况下，此参数设置为 `omni-moderation-latest`。它会随时间自动升级，确保你始终使用最准确的模型。如果你决定使用任何基于文本的审核模型，则只能提交文本进行评估。然而，全能审核模型能够评估文本和图像内容。因此，请选择最适合你用例的模型。 |

### 处理响应

成功调用审核模型后，该方法将返回一个响应，其结构如表 8-2 所示。

### 审核（字典）

**表 8-2** 审核响应的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `id` | String | 审核请求的唯一标识符。 |
| `model` | String | 用于执行审核请求的模型。 |
| `results` | List | 审核对象列表。 |
| ↳ `flagged` | Boolean | 标记内容是否违反 OpenAI 的使用政策。 |
| ↳ `categories` | List | 类别列表及其是否被标记。 |
|  ↳↳ `hate` | Boolean | 表示给定文本是否表达、煽动或宣扬基于种族、性别、宗教、民族、国籍、残疾状况、性取向或种姓的仇恨。 |
|  ↳↳ `hate/threatening` | Boolean | 表示给定文本是否包含仇恨内容，并同时基于上述偏见对目标群体威胁实施暴力或严重伤害。 |
|  ↳↳ `harassment` | Boolean | 表示给定文本是否包含表达、煽动或宣扬针对任何目标的骚扰性语言。 |
|  ↳↳ `harassment/threatening` | Boolean | 表示给定文本是否包含骚扰内容，并同时威胁对任何目标实施暴力或严重伤害。 |
|  ↳↳ `self-harm` | Boolean | 表示给定文本是否包含宣扬、鼓励或描述自残行为的内容，例如自杀、割伤和饮食失调。 |
|  ↳↳ `self-harm/intent` | Boolean | 表示给定文本是否包含说话者表示正在或意图进行自残行为的内容，例如自杀、割伤和饮食失调。 |
|  ↳↳ `self-harm/instructions` | Boolean | 表示给定文本是否包含鼓励进行自残行为的内容，例如自杀、割伤和饮食失调。这包括提供如何实施此类行为的指导或建议的内容。 |
|  ↳↳ `sexual` | Boolean | 表示给定文本是否包含旨在引起性兴奋的内容，例如对性行为的描述。这包括推广性服务的内容；但**不包括**性教育和健康等主题。 |
|  ↳↳ `sexual/minors` | Boolean | 表示给定文本是否包含涉及未满 18 岁个人的内容。 |
|  ↳↳ `violence` | Boolean | 表示给定文本是否包含描述死亡、暴力或身体伤害的内容。 |
|  ↳↳ `violence/graphic` | Boolean | 表示给定文本是否包含详细描述死亡、暴力或身体伤害的内容。 |
| ↳ `category_scores` | List | 类别列表及其对应的模型评分。 |
|  ↳↳ `hate` | Number | “hate”类别的评分。 |
|  ↳↳ `hate/threatening` | Number | “hate/threatening”类别的评分。 |
|  ↳↳ `harassment` | Number | “harassment”类别的评分。 |
|  ↳↳ `harassment/threatening` | Number | “harassment/threatening”类别的评分。 |
|  ↳↳ `self-harm` | Number | “self-harm”类别的评分。 |
|  ↳↳ `self-harm/intent` | Number | “self-harm/intent”类别的评分。 |
|  ↳↳ `self-harm/instructions` | Number | “self-harm/instructions”类别的评分。 |
|  ↳↳ `sexual` | Number | “sexual”类别的评分。 |
|  ↳↳ `violence` | Number | “violence”类别的评分。 |
|  ↳↳ `violence/graphic` | Number | “violence/graphic”类别的评分。 |

以下列表是调用审核模型后返回的审核响应示例。表 8-2 看起来有点复杂，但正如你所见，如果任何类别被标记为“true”，那么 `results.flagged` 节点也会被标记为“true”。

请查看清单 8-1，了解审核响应的实际示例。

```json
{
"id": "modr-XXXXX",
"model": "text-moderation-005",
"results": [
{
"flagged": true,
"categories": {
"sexual": false,
"hate": false,
"harassment": false,
"self-harm": false,
"sexual/minors": false,
"hate/threatening": false,
"violence/graphic": false,
"self-harm/intent": false,
"self-harm/instructions": false,
"harassment/threatening": true,
"violence": true,
},
"category_scores": {
"sexual": 1.2282071e-06,
"hate": 0.010696256,
"harassment": 0.29842457,
"self-harm": 1.5236925e-08,
"sexual/minors": 5.7246268e-08,
"hate/threatening": 0.0060676364,
"violence/graphic": 4.435014e-06,
"self-harm/intent": 8.098441e-10,
"self-harm/instructions": 2.8498655e-11,
"harassment/threatening": 0.63055265,
"violence": 0.99011886,
}
}
]
}
```

**清单 8-1** 审核响应

## 为审核模型创建客户端：`moderation_client.py`

清单 8-2 是我们用于调用审核模型的客户端。请先查看它，然后我们将讨论其中的重要部分。

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
class ModerationResponse:
def __init__(self):
load_dotenv()
self.client = OpenAI()
def moderate_text(self, text):
moderation = self.client.moderations.create(input=text)
return moderation
```

**清单 8-2** `moderation_client.py`

在本书的前几章中，我们为 OpenAI API 类的各种方法创建了客户端脚本。因此，上面 `moderation_client.py` 中的 `ModerationResponse` 类应该看起来相当熟悉。归根结底，我们有一个简单的函数 `moderate_text()`，它允许我们传递要评估的文本并返回响应。

## 让 `content_moderator_bot.py` 更智能

现在我们有了用于调用审核模型的 `moderation_client.py`，让我们看看更新后的 `content_moderator_bot.py`（之前名为 `content_moderator_bot_dumb.py`），它将使用 `moderation_client.py` 检查有害内容，并使用 `chatgpt_client_for_qa_and_moderation.py`（与前一章相比未修改）检查潜在的诈骗。

清单 8-3 是我们智能 Discord 审核机器人 `content_moderator_bot.py` 的完整源代码。

```python
"""
content_moderation_bot.py
一个集成了 ChatGPT 和审核服务的 Discord 机器人，用于 Discord 服务器的自动内容审核。
该脚本使用`discord.Client()`类初始化一个 Discord 机器人，并监听消息发送等事件。当收到消息时，它会同时调用 ChatGPT API 和审核服务来分析消息内容是否违反规则。如果任一服务标记了该消息，机器人将删除该消息，并向用户发送通知，说明其为何被视为不当内容。

依赖项：
- discord (https://pypi.org/project/discord.py/)
- chatgpt_client_for_qa_and_moderation (假设为自定义模块，提供与 ChatGPT 的交互)
- moderation_client (假设为自定义模块，提供与审核服务的交互)

使用方法：
1. 将`DISCORD_TOKEN`变量替换为从 Discord 开发者门户获取的机器人令牌。
2. 确保`chatgpt_client_for_qa_and_moderation`和`moderation_client`模块已正确实现并可访问。

注意：此脚本假设存在用于与 ChatGPT 和审核服务交互的模块。
"""
import discord
from chatgpt_client_for_qa_and_moderation import ChatGPTClient
from moderation_client import ModerationResponse

### 用于身份验证的机器人令牌
DISCORD_TOKEN = ''

### 初始化 Discord 客户端
discord_client = discord.Client()

### 为 ChatGPT 创建系统消息
system_message_to_chatgpt = """
你是 Discord 服务器的自动审核助手。
请检查每条消息是否存在以下违规行为：
1. 敏感信息
2. 辱骂
3. 不当评论
4. 垃圾信息，例如：全大写字母的消息、重复多次的相同短语或单词、超过 3 个感叹号或问号。
5. 广告
6. 外部链接
7. 政治消息或辩论
8. 宗教消息或辩论
如果检测到任何违规行为，请回复"FLAG"（大写，不含引号）。如果消息符合规则，请回复"SAFE"（大写，不含引号）。
"""

initial_instructions_to_chatgpt = "分析以下消息是否存在违规行为："

### 初始化 ChatGPT 客户端
chatgpt_client_for_qa_and_moderation = ChatGPTClient(system_message_to_chatgpt, initial_instructions_to_chatgpt)

### 初始化审核客户端
moderation_client = ModerationResponse()

### 机器人就绪时的事件处理程序
@discord_client.event
async def on_ready():
    """
    当机器人成功登录并准备好接收事件时触发的事件处理程序。
    """
    print('已登录为', discord_client.user)
    print('------')

### 收到消息时的事件处理程序
@discord_client.event
async def on_message(message):
    """
    收到消息时触发的事件处理程序。
    参数：
    message (discord.Message)：机器人收到的消息。
    返回：
    None
    """
    # 忽略机器人自己发送的消息，防止自我响应
    if message.author == discord_client.user:
        return

    # 调用审核方法检查有害内容
    moderation_response = moderation_client.moderate_text(message.content)

    # 调用 ChatGPT 根据收到的消息生成响应
    response_from_chatgpt = chatgpt_client_for_qa_and_moderation.send_message_from_discord(message.content)

    # 检查 ChatGPT 的响应是否为"FLAG"，或审核响应是否标记了输入内容
    if response_from_chatgpt == "FLAG" or moderation_response.results[0].flagged:
        # 删除消息
        await message.delete()
        # 提及发送不当消息的用户
        author_mention = message.author.mention
        # 发送一条消息，提及用户并解释为何不当
        await message.channel.send(f"{author_mention} 此评论被视为不适合本频道。" +
                                   "如果您认为这是误判，请联系人工服务器管理员。")

### 使用提供的令牌运行机器人
discord_client.run(DISCORD_TOKEN)
```

**清单 8-3** `content_moderator_bot.py`

### 对 `on_message(message)` 函数的更新

在 Discord 服务器的任何频道中收到消息后，`on_message(message)` 函数会被异步调用。以下是需要注意的最重要的更改：

```markdown
#### 调用审核模型检查有害内容
`moderation_response = moderation_client.moderate_text(message.content)`
#### 调用 ChatGPT 根据收到的消息生成响应
`response_from_chatgpt = chatgpt_client_for_qa_and_moderation.send_message_from_discord(message.content)`
#### 检查 ChatGPT 的响应是否为"FLAG"，或审核响应是否标记了输入内容
```python
if response_from_chatgpt == "FLAG" or moderation_response.results[0].flagged:
    # 删除消息
    await message.delete()
    # 提及发送不当消息的用户
    author_mention = message.author.mention
    # 发送一条消息，提及用户并解释为何不当
    await message.channel.send(f"{author_mention} 此评论被视为不适合本频道。" +
                               "如果您认为这是误判，请联系人工服务器管理员。")
```

在这里，我们获取在 Discord 服务器中发布的每条消息，并使用`Moderations`类和`Chat`方法进行检查。如果任一方法返回信息告知我们消息被标记，则删除频道中的消息，并通知用户其消息违反了规则。

现在我们的内容审核 Discord 机器人已经智能化，让我们试试看吧！

## 运行我们的智能内容审核机器人：`content_moderator_bot.py`

现在让我们运行全新改进的内容审核 Python Discord 机器人`content_moderator_bot.py`。执行应用程序后，请务必返回 Discord 服务器并开始提问。图 8-1 展示了机器人的运行情况。

![](img/615442_1_En_8_Fig1_HTML.jpg)

**图 8-1** 与我们的智能 Discord 内容审核机器人进行讨论：`content_moderator_bot.py`

清单 8-4 展示了我们与 Discord 机器人之间的对话，用于测试并查看其功能。

```
我：大家好，我喜欢 Crooks Bank 应用！
我：这个应用太棒了！
我：来我的网站！http://www.google.com
内容审核机器人：@PythonChatGPT 此评论被视为不适合本频道。如果您认为这是误判，请联系人工服务器管理员。
我：很抱歉违反了规则。我现在已经改过自新了
我：但我有个坏消息要告诉你们
我：我想☠所有人
内容审核机器人：@PythonChatGPT 此评论被视为不适合本频道。如果您认为这是误判，请联系人工服务器管理员。
```

**清单 8-4** 我们与智能审核 Discord 机器人的攻击性对话

在两种情况下，当不当内容发布到 Discord 服务器的任何频道时，不仅违规用户被点名，而且不良消息也被删除了。好机器人！

你注意到`Moderation`和`Chat`方法也能读取表情符号了吗？

## 结论

在本章中，我们为整个 Discord 服务器创建了一个功能完整的内容审核机器人！我们利用 OpenAI 的`Moderations`和`Chat`方法创建了一个自定义内容审核机器人，它不仅能够标记仇恨和威胁消息等不安全内容，还能防止 Discord 服务器的用户受到不必要的骚扰。

## 留给读者的练习

尽管我们在本章（以及本书中！）已经完成了许多工作，但仍有一项改进代码的工作可以做。例如：

- 我们创建的各个 Discord 机器人已经知道不响应自己发送的消息。然而，这些机器人还不知道它们不应该响应由**其他机器人**发送的消息。换句话说，如果你同时运行两个机器人，并且有人在“问答”频道中发布了不良内容，内容审核机器人当然会删除该消息并通知所有人消息已被删除。但是，由于技术支持机器人不知道它不应该响应其他机器人，它会尝试创建回复。当然，机器人之间不应该互相交谈。

## 索引

### A

- `AccuWeather API`
- `accuweather_forecaster.py`
- `带注释的数据`
- `API 密钥`
- `OpenAI API`
  - 清理响应
  - 模型列表
  - 系统级环境变量
  - 创建 `.env` 文件
  - 创建 `.gitignore` 文件
  - 硬编码
  - Linux
  - Mac OS
  - Windows
- `API 请求流程` (`SlackApiError`)
- `人工智能工具`
- `ASR` (参见 `自动语音识别 (ASR)`)
- `助手消息 (字典)`
- `audio_splitter.py` 应用
- `自动语音识别 (ASR)`

### B

- `机器人`
  - `Discord`
    - `ID 令牌`
    - 基本信息
    - `消息内容意图`
    - `OAuth2` 参数
    - 权限
    - 角色

### C

- `channel_reader_slack_bot.py`
- `ChatCompletion`
- `ChatCompletion 对象响应`
- `ChatGPT`
  - 用于避免阅读文档的 API
  - `聊天游乐场`
  - 下载的代码
  - 设计模式
  - 高效的结对编程伙伴
  - 用于检查 linting 错误的语言模型
  - `model_lister.py`
  - 结对编程伙伴 (参见 `结对编程伙伴`)
  - 使用 `Python` 设计模式的 `Python 代码`
  - `Python` 开发者
  - `工厂方法模式`
  - `观察者模式`
  - `单例模式` *对比* `正则表达式`
  - 响应
  - 软件开发
  - 远程医疗
  - `分词器`
- `ChatGPT API`
  - 带注释的数据
  - `DALL⋅E`
  - 数据模型定义
  - 嵌入模型
  - `GPT-3.5`
  - `GPT-4`
  - 遗留/弃用模型
  - 模型
  - 审核模型
  - 预训练模型
  - 预训练神经网络
  - 温度
  - 令牌
  - `TTS`
  - `Whisper` 模型
- `chatgpt_client_for_qa_and_moderation.py` 脚本
- `Python` 中的 `ChatGPT` 客户端
  - 创建 `chatgpt_client.py`
  - 运行脚本
- `聊天游乐场`
  - “/命令”
- `内容审核 AI`
  - `content_moderator_bot.py`
  - `moderation_client.py`
  - `OpenAI.moderations.create()` 方法
- `内容审核机器人`
  - `content_moderator_bot_dumb.py`
  - `content_moderator_bot.py`
  - 智能 `Discord` 审核机器人
  - `on_message(message)` 方法
- `上下文窗口`
- `协调世界时 (UTC)`
- `Core Weather 有限试用版`
- `Crook 银行`
- `cURL 命令`
- `客户支持`

### D

- `dalle_client.py`
- `DALL⋅E` 模型
  - 描述性
  - `OpenAI.images.generate()` 方法
  - 创建图像生成器
  - `dalle_client.py`
  - `DALL⋅E 生成的图像`
  - 方法参数
  - 请求体
  - 响应对象，结构
- `提示工程`
  - `GPT-4`
  - 使用 `GPT-4` 创建提示
  - 图像类型
- `数据模型`
- `Discord`
  - 频道
  - 社区平台
  - 确认、权限和能力
  - 创建频道
  - 生成的 URL
  - 注册新的 `Discord 机器人` 应用
  - 流媒体视频
  - 语音聊天
- `Discord 机器人`
  - “/命令”
  - `content_moderator_bot_dumb.py`
  - 带有 `Discord` 的 `content_moderator_bot_dumb.py`
  - `Discord ID 令牌`
  - 基本信息
  - `OAuth2` 参数
  - `onMessageReceived()` 方法
- `Discord 开发者` 网站
- `Discord ID 令牌`

### E

- `嵌入模型`
- 创建 `.env` 文件

### F

- `工厂方法模式`
- `FAQ.txt` 文件
- `FFmpeg`

### G, H

- `地理编码`
- 创建 `.gitignore` 文件
- `Google Maps API`
  - 应用仪表盘
  - `JavaScript API`
  - 库
  - 平台账户设置
  - `API 和服务`
  - 文档主页
  - 密钥和凭据选项卡
  - 路线
- `GPT-3.5`
- `GPT-4`

### I, J, K

- `IDE`
- `__init__` 方法

### L

- `语言模型`
- `遗留/弃用模型`

### M

- `审核机器人`
- `model_lister.py`
- `moderation_client.py`
- `审核模型`
- `多模态 AI`
  - 格式
  - `播客可视化工具`

### N

- `自然语言处理 (NLP)`
- `自然语言理解 (NLU)`
- `神经网络`
- `NLP` (参见 `自然语言处理 (NLP)`)
- `NLU` (参见 `自然语言理解 (NLU)`)

### O

- `观察者模式`
- `on_message(message)` 方法
- `on_message()` 方法
- `onMessageReceived()` 方法
- `OpenAI`
  - 添加消息
  - `API 密钥`
  - 助手字段
  - `聊天游乐场`
  - 库，安装/更新
  - 模型
  - 系统字段
  - 温度
  - 令牌
  - 用户字段
  - 用户消息
  - 查看代码
- `OpenAI.audio.transcriptions.create()` 方法
- `OpenAI.chat.completions.create()` 方法
  - 定义
  - 参数格式
- `OpenAI.images.generate()` 方法
- `OpenAI.models.list()`
  - 处理响应
  - `SyncPage` 结构
- `OpenAI.moderations.create()` 方法
  - 审核响应参数
  - `results.flagged` 节点
  - 审核响应的结构

### P

- `结对编程伙伴`
  - 到达时间和距离
  - `cURL 命令`
  - `departure-time` 参数
  - `Google Maps API`
  - *大量*文档
  - 天气应用
  - `AccuWeather API`
  - 生产力
  - 提示工程
  - 提示
- `播客可视化工具`
- `提示工程`
  - `提示 #1：“tl;dr”`
  - `提示 #2：“用三句话以内解释”`
  - `提示 #3：“我是经理。向我解释发生了什么”`
  - `提示 #4：“给我下一步行动的建议”`
- `Python`
  - `OpenAI APIs`
- `问答机器人应用` (参见 `问答机器人应用, Python`)

### Q

- `问答机器人应用, Python`
  - `Discord`
  - 特性
  - `on_message(message)` 方法
  - `tech_support_bot_dumb.py`

### R

- `正则表达式` *对比* `ChatGPT`
  - `NLP`
  - `NLU`
  - 正则表达式
  - 情感分析
  - 分离类型
- `正则表达式`

### S

- `send_message_from_discord()` 方法
- `情感分析`
- `单例模式`
- `Slack`
  - 客户支持
  - 查找频道 ID
  - 读取消息
  - `Slack API 令牌` (`SLACK_BOT_TOKEN`)
- `Slack 机器人应用`
  - 添加特性和功能
  - `API` 网站
  - 创建
  - 获取访问令牌
  - 安装，工作区
  - 邀请到频道
  - `Python`
  - `Slack` 实例
  - 指定 `OAuth` 范围
  - 查看 `OAuth 和权限` 页面
  - `Python Slack` 库，`slack_sdk`
- `系统消息 (字典)`

### T

- `tech_support_bot_dumb.py`
- `tech_support_bot.py` 脚本
  - 所做的更改
  - `ChatGPTClient` 对象实例化
  - `discord.Client()` 类
  - 外部文本文件 `FAQ.txt` 文件
  - 里程碑式的成就
  - `on_message()` 系统消息更新
- `tech_support_smart_bot.py`
- `远程医疗`
- `令牌`
- `工具消息 (字典)`
- `TTS`
- `消息类型`
  - 助手消息
  - 系统消息
  - 工具消息
  - 用户消息

### U, V

- `用户消息 (字典)`

### W

- `天气追踪器`
- `Whisper` 模型
  - `audio_splitter.py` 应用
  - 特性
  - 局限性
  - `OpenAI.audio.transcriptions.create()` 方法
  - `OpenAI` 模型
  - 播客
  - 请求体
  - 语音识别
  - 转录音频，`OpenAI.audio.transcriptions.create()` 方法
  - `whisper_transcriber.py`

### X, Y, Z

- `XP (极限编程)`
```