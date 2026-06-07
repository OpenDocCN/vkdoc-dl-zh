# 5. 多模态 AI：使用 Whisper 和 DALL·E 3 创建播客可视化工具

在本章中，我们将看到如何结合多个模型来创造令人惊叹的结果。作为一名狂热的播客听众，我在收听沉浸式音频故事时，常常会好奇其中的场景、画面、角色、主题或背景究竟是什么样的。

让我们引入一个新术语：**多模态 AI**。简单来说，生成式 AI 模型可以创建以下四种格式的内容：

- 文本
- 音频
- 图像
- 视频

每种格式都是一种**模态**。多模态 AI 是指同时使用多个 AI 模型来生成（或理解）内容的过程，其中输入是一种模态，而输出是另一种不同的模态。

以 OpenAI 的 Whisper 模型为例。如果你提供音频，它能够将所说的所有内容转录为文本。同样，DALL·E 模型也是如此。如果你提供文本提示，它就能根据你的描述生成图像。

因此，我们将使用 OpenAI 的多个模型来创建一个播客可视化工具。虽然涉及几个步骤，但最终效果令人惊艳。在收听一个关于某人用豆腐烹饪美味佳肴的播客时（别急着否定，先试试看），播客可视化工具生成了图 5-1 中的图像。

![](img/625846_1_En_5_Fig1_HTML.jpg)

一个人戴着眼镜，穿着浅蓝色衬衫，手里端着一个白色碗，碗里放着一大块豆腐。背景模糊，暗示是室内环境。

**图 5-1** AI 生成的图像。使用 GPT-4、Whisper 和 DALL·E 模型可视化关于豆腐的播客的结果

为了让播客可视化工具的代码易于理解，我们将分以下三个步骤进行：

- **步骤 1：** 选取一集播客，使用 Whisper 模型获取转录文本。
- **步骤 2：** 获取生成的转录文本，使用 GPT-4 模型描述该集播客中讨论内容的视觉方面。
- **步骤 3：** 获取生成的描述，使用 DALL·E 模型生成图像。

本章提供的代码具有大量实际用途，例如：

- 如果你只是好奇播客一集中的内容可能是什么样子（我经常如此），你可以获得一个简单的代表性视觉图像，与你正在收听的内容相关联。
- 对于有听力障碍的人，你可以轻松地将播客或广播节目转换为幻灯片式的图像展示。这大大增强了内容的可访问性。
- 对于播客创作者，你现在有了一种简单的方法，可以为每一集添加视觉/主图。这很有用，因为 Apple Podcasts 和 Spotify 等播客播放器允许播客创作者显示单张图像来关联单集节目，这有助于提高听众的参与度。

## 介绍 OpenAI 的 Whisper 模型

现在让我们引入另一个新术语：**自动语音识别**（ASR）。普通日常消费者对这种技术非常熟悉，因为它已集成到手机（例如 iPhone 的 Siri）和智能音箱（例如任何 Alexa 设备）中。其核心是，ASR 技术将口语转换为文本。

Whisper 是 OpenAI 的语音识别模型，其准确性高得惊人。下面的列表是广受欢迎的 DuoLingo 西班牙语播客一集的转录文本，该播客通过将英语和西班牙语交织成一个叙事故事，使英语听众能够轻松理解西班牙语。该转录文本是使用 Whisper 模型生成的。

```
...我是 Martina Castro。每集节目，我们都会为您带来引人入胜的真实故事，帮助您提高西班牙语听力，并获得看待世界的新视角。讲故事的人将使用中级西班牙语，而我会用英语补充上下文。如果您错过了什么，可以随时回退重听。我们还在 podcast.duolingo.com 提供完整转录文本。
在成长过程中，Linda 对她的祖母 Erlinda 非常着迷。Erlinda 是一位治疗师或 curandera，即那些为精神、情感、身体或灵性疾病提供疗法的人。
在危地马拉，这是一种通过同一家族世代口头传承的实践。Mal de ojo，即邪恶之眼，被许多危地马拉人视为一种疾病，他们认为人类有能力将负能量传递给他人。邻居们会在怀疑婴儿能量失衡时，把他们带到 Linda 的祖母那里。Su madre lo llevaba a nuestra casa para curarlo...
```

**列表 5-1** Whisper 模型执行语音识别，将音频转换为文本

如果你以前使用过语音识别系统（即使是像 Siri 和 Alexa 这样复杂的技术），你会知道它存在一些问题，例如：

- **语音识别在标点符号方面存在问题。**
    - 你有没有注意到，没有人说话时会带标点符号？对于英语，我们通过改变语调或音量来提问或表达感叹。我们也用短停顿和长停顿来表示逗号和句号。
- **语音识别在外来词和口音方面存在问题。**
    - 根据你问的人不同，英语中至少有 17 万个单词。然而，在英语会话中，我们经常使用外来词，例如：
        - Tsunami（源自日语）：通常由地震引起的大海啸
        - Hors d’oeuvre（源自法语）：开胃菜
        - Lingerie（源自法语）：女性内衣或睡衣
        - Aficionado（源自西班牙语）：对特定活动或主题非常热衷的人
        - Piñata（源自西班牙语）：一个色彩鲜艳的糖果盒，供孩子们尽情敲打
- **语音识别在名称方面存在问题。**
    - 某些人名、企业名和网站名通常难以拼写和理解。
- **语音识别在同音词方面存在问题。**
    - 你还记得那些发音相同但拼写和含义不同的词吗？本书那位出色的编辑全都知道！
        - Would / Wood
        - Flour / Flower
        - Two / Too / To
        - They’re / There / Their
        - Pair / Pare / Pear
        - Break / Brake
        - Allowed / Aloud

正如你从列表 5-1 中看到的，Whisper 能够理解音频中的所有标点符号，识别所有外来词（其中有多个），理解名称，以及 URL 中的公司名称（`Duolingo`）！当然，如果你注意到了，它还能区分`wood`和`would`。

## Whisper 模型的功能与限制

Whisper 模型能够将以下语言的语音音频转换为文本：

- 南非荷兰语
- 阿拉伯语
- 亚美尼亚语
- 阿塞拜疆语
- 白俄罗斯语
- 波斯尼亚语
- 保加利亚语
- 加泰罗尼亚语
- 中文
- 克罗地亚语
- 捷克语
- 丹麦语
- 荷兰语
- 英语（当然！）
- 爱沙尼亚语
- 芬兰语
- 法语
- 加利西亚语
- 德语
- 希腊语
- 希伯来语
- 印地语
- 匈牙利语
- 冰岛语
- 印度尼西亚语
- 意大利语
- 日语
- 卡纳达语
- 哈萨克语
- 韩语
- 拉脱维亚语
- 立陶宛语
- 马其顿语
- 马来语
- 马拉地语
- 毛利语
- 尼泊尔语
- 挪威语
- 波斯语
- 波兰语
- 葡萄牙语
- 罗马尼亚语
- 俄语
- 塞尔维亚语
- 斯洛伐克语
- 斯洛文尼亚语
- 西班牙语
- 斯瓦希里语
- 瑞典语
- 他加禄语
- 泰米尔语
- 泰语
- 土耳其语
- 乌克兰语
- 乌尔都语
- 越南语
- 威尔士语

因此，归根结底，它能够理解您自己说的音频，以及您的朋友和同事可能说的任何语言的音频。

开发者每分钟最多只能向 API 发送 50 个请求，因此如果您要转录大量音频，需要考虑此限制。

Whisper 支持 `flac`、`mp3`、`mp4`、`mpeg`、`mpga`、`m4a`、`ogg`、`wav` 或 `webm` 格式的音频。无论您使用哪种格式，发送到 API 的最大文件大小为 25MB。

如果您没有大量处理过音频文件，请注意，某些格式会生成**非常大**的文件（例如 `wav` 格式），而其他格式会生成非常小的文件（例如 `m4a` 格式）。因此，将文件转换为其他格式可以帮助您应对 25MB 的限制。不过，在本章后面，我们将看到一个工具的代码，该工具可以获取单个大音频文件并将其分割成多个较小的文件。

## 使用 `OpenAI.audio.transcriptions.create()` 转录音频

`OpenAI.audio.transcriptions.create()` 方法将音频转换为文本，并且仅与 Whisper 模型兼容。让我们看看该方法需要哪些参数才能成功调用 API。

### 检查方法参数

**表 5-1. Whisper 的请求体**

| 字段 | 类型 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `file` | 文件 | 必需 | 您想要转录的整个音频文件。接受的格式为：`flac`、`mp3`、`mp4`、`mpeg`、`mpga`、`m4a`、`ogg`、`wav`、`webm` |
| `model` | 字符串 | 必需 | 您想要用于转录的模型 ID。兼容的模型包括：`whisper-1` |
| `prompt` | 字符串 | 可选 | 可以提供任何文本来改变模型的转录风格，或为模型提供来自先前音频片段的更多上下文。为确保最佳效果，请确保提示与音频使用相同的语言。此外，此字段可用于更改 Whisper 不熟悉的任何单词的拼写或大小写。 |
| `response_format` | 字符串（默认值：`json`） | 可选 | 这是转录输出的格式。接受的格式为：`json`、`text`、`srt`、`verbose_json`、`vtt` |
| `temperature` | 数字（默认值：`0`） | 可选 | 这是采样温度，范围从 0 到 1。较高的值会增加输出的随机性，而较低的值则确保输出更具确定性。 |
| `language` | 字符串 | 可选 | 这是输入音频的语言。它是可选的，但提供该值可以提高转录的准确性和延迟 |

## 创建用于分割音频文件的实用程序

那么，我们几乎已经能够使用转录端点以编程方式调用 Whisper 模型了。但是，Whisper 模型对每个文件有 25MB 的限制。

现在，如果您收听的是例如德克萨斯大学奥斯汀分校的 StarDate 播客，这不成问题。这个播客大约有 2 分钟的音频，让您很好地了解夜空中应该寻找什么。但是，对于其他往往持续一小时（甚至更长时间）的音频节目来说，情况就不同了。在这种情况下，您肯定会超过 25MB 的文件限制。

因此，让我们与 ChatGPT 结对编程，利用我们的人类智慧创建我们自己的实用程序，该程序将获取单个音频文件并将其分割成多个较小的文件。

**注意：** 在本节中，我将介绍如何将大音频文件分割成较小片段的众多可能性之一。例如，您可以使用流行的音频编辑应用程序（例如开源工具 Audacity 或授权工具 Adobe Audition）手动将大文件切割成较小的文件。

`FFmpeg` 是以编程方式操作媒体文件最可靠的工具之一，并且由于它是开源的，因此使用它来满足我们的需求是合理的。

清单 5-2 是总结我们需要完成的任务的提示。

```
系统：你是一名 JavaScript 开发者
用户：编写一个应用程序，该应用程序接收单个 MP3 文件作为输入，并使用 ffmpeg 库将该文件分割成不超过 10 分钟的连续片段。
```

**清单 5-2. 提示。使用 ChatGPT 创建 AudioSplitter 应用程序**

经过一番来回交流，我能够创建（如清单 5-3 所示）`audiosplitter` 应用程序，其中包含我对 ChatGPT 生成内容的编辑。

```javascript
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("ffmpeg-static");
const path = require("path");
const fs = require("fs");

// 配置 ffmpeg 使用静态二进制文件
ffmpeg.setFfmpegPath(ffmpegPath);

// 分割 MP3 文件的函数
function splitMP3(inputFile, duration) {
    ffmpeg.ffprobe(inputFile, (err, metadata) => {
        if (err) {
            console.error(`发生错误：${err.message}`);
            return;
        }
        const totalDuration = metadata.format.duration;
        const numberOfChunks = Math.ceil(totalDuration / duration);
        const outputDir = path.join(__dirname, "output");

        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir);
        }

        for (let i = 0; i < numberOfChunks; i++) {
            const startTime = i * duration;
            const outputFile = path.join(outputDir, `output_${i + 1}.mp3`);

            ffmpeg(inputFile)
                .setStartTime(startTime)
                .setDuration(duration)
                .output(outputFile)
                .on("end", () => {
                    console.log(`已创建 ${outputFile}`);
                })
                .on("error", (err) => {
                    console.error(`发生错误：${err.message}`);
                })
                .run();
        }
    });
}

// 使用示例
const inputFilePath = path.join(__dirname, "input.mp3"); // 输入 MP3 文件的路径
const maxDuration = 600; // 10 分钟（以秒为单位）
splitMP3(inputFilePath, maxDuration);
```

**清单 5-3. 响应。Audio-splitter.js**

我们的目标很简单：使用 JavaScript 和 FFmpeg 将一个 MP3 文件分割成指定时长的片段。每个片段不应超过 10 分钟（即 600 秒）。整个过程使用 `fluent-ffmpeg` 库进行管理，该库充当 FFmpeg 的 Node.js 封装器，使我们能够高效地处理音频和视频处理。

当然，首要步骤之一是确保您已安装并正确配置了 FFmpeg。我们的脚本使用 `ffmpeg-static` 来确保 FFmpeg 二进制文件与代码打包在一起。这避免了特定于系统的依赖性问题。您可以使用以下命令安装依赖项：

```
npm install fluent-ffmpeg ffmpeg-static
```

主要逻辑位于 `splitMP3()` 函数中，该函数接受两个参数：输入 MP3 文件的路径和所需的片段时长（以秒为单位）。

首先，我们使用 `ffmpeg.ffprobe()` 函数从输入 MP3 文件中提取元数据，特别是其总时长。这是必要的，因为它让我们确定需要创建多少个片段。

如果输出目录不存在，我们使用 Node.js 的 `fs` 模块创建它。这确保了我们分割后的音频文件有一个专用的目标文件夹。

运行脚本后，输出文件夹将填充 MP3 文件，每个文件的时长不超过 10 分钟。如果输入文件少于 10 分钟，则只会生成一个片段。

## 使用 Whisper 创建音频转录器

现在，让我们构建下一个 JavaScript 应用，它将使用 Whisper 模型来创建音频转录。同样，我们将与 ChatGPT 进行结对编程，以获得一个可用的基础。

清单 5-4 是在聊天游乐场中输入的提示，以启动整个过程。请务必注意，我要求设置 60 秒的 HTTP 请求超时，因为 Whisper 生成转录可能需要一些时间。

```
System: You are a JavaScript developer.
User: Using JavaScript, write a script that iterates over all of the mp3 files in a single folder on my local computer and send all the files in the folder to the webservice provided by OpenAI's Whisper model, using the OpenAI's node.js library.
Model: gpt-4
Temperature: 1
Maximum Length: 1150
```

**清单 5-4** 提示：要求 ChatGPT 使用 OpenAI 的 JavaScript 库并将 MP3 文件发送到 Whisper 的 API

经过一番来回交流，ChatGPT 给出了一个可行的回复（清单 5-5）：

# 播客可视化工具

## 小试牛刀：用播客进行测试

好了，让我们用目前已有的代码来运行一个测试。《美国生活》是一档每周播出的公共广播节目（也是一个播客），由艾拉·格拉斯主持，并与芝加哥 WBEZ 电台合作制作。

每一集都会围绕一个特定的主题或话题，编织一系列故事。有些故事是调查性新闻报道，另一些则只是对拥有引人入胜故事的普通人的采访。第 811 集名为“我唯一不能去的地方”，文件大小为 56MB，格式为 MP3。由于我们已经知道 56MB 的文件太大，无法发送给 Whisper 进行转录，因此让我们运行工具来拆分音频文件，并转录各个片段。

清单 5-6 展示了该集完整转录文本的一个摘录。

```
"...我的表妹卡米尔其实并不喜欢狗，但有一只狗她非常喜爱。它叫福克西，因为它看起来就像一只狐狸，只不过它是黑色的。它是邻居家的狗，但卡米尔和它似乎有一种真正的亲密关系，也许是因为它们俩都离地面不远。那时卡米尔大约四五岁，说话有点咬舌，所以福克西被她叫成了福兹。我觉得这是我听过最可爱的事情之一。
卡米尔对福克西的记忆，几乎像电影一样。她的回忆感觉像是无尽的夏天，朦胧而完美，就像在噼啪作响的胶片上拍摄的场景。我只记得那种兴奋地跑去看福克西的感觉。我脑海里有一个画面，就是来到房子前，我能看到福克西在外面。我能透过通往花园的门看到福克西。有一个关于卡米尔和福克西的故事，我经常想起。我和姐姐谈论了好几年，但从未和卡米尔谈过。故事是这样的：有一次，当它们在玩耍的时候..."
```

清单 5-6 《美国生活》第 811 集的转录文本节选

为简洁起见，我们只展示了转录文本的一个摘录。完整的转录文本超过 8000 个单词，因为该集节目时长接近 1 小时。

## 元操作：提示工程——让 `gpt-4o-mini` 为 DALL·E 编写提示

由于我们想要可视化的播客剧集的完整文本转录有数千个单词，我们将使用 `gpt-4o-mini` 来自动创建 DALL·E 模型所需的提示。DALL·E 能够根据提示中的文本描述创建图像，但最好让提示尽可能简短。清单 5-7 是让 `gpt-4o-mini` 为 DALL·E 生成提示的输入。

```
System: You are a service that helps to visualize podcasts.
User: Read the following transcript from a podcast. Describe for a visually impaired person the background and subject that best represents the overall theme of the episode. Start with any of the following phrases:
- "A photo of"
- "A painting of"
"A macro 35mm photo of"
"Digital art of "
User: Support for This American Life comes from Squarespace...
Model: gpt-4o-mini
Temperature: 1.47
Maximum length: 150
Top P: 0
Frequency penalty: 0.33
Presence penalty: 0
```

清单 5-7 让 GPT-4 为 DALL·E 创建提示的提示

正如你在提示中所见，使用的模型是 `gpt-4o-mini`，它允许我们处理长达 128k 个 token 的极长文本转录。DALL·E 需要知道要生成的图像类型，因此我们需要指定图像应该是照片、绘画、数字艺术等。我们需要确保模型生成的文本简短，因此我们希望最大长度为 150 个 token。此外，为了防止 ChatGPT 多次重复某些短语，我们引入了 0.33 的频率惩罚。

清单 5-8 展示了 ChatGPT 在阅读《美国生活》第 811 集转录文本后给出的结果。

```
Digital art of a young girl sitting in a garden with a black dog that looks like a fox. The girl is smiling and the dog is wagging its tail. The image has a hazy, dream-like quality, with crackly film effects to evoke nostalgia.
```

清单 5-8 由 GPT-4o-mini 为 DALL·E 创建的提示

## 使用 `OpenAI.openai.images.generate()` 创建图像

`OpenAI.images.generate()` 方法允许你使用 DALL·E 模型，根据文本提示动态创建图像。

表 5-2 描述了创建图像所需的 JSON 对象格式。显而易见，`prompt` 是成功调用服务的唯一必需参数。

**表 5-2** 创建图像端点的请求体

| 字段 | 类型 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `prompt` | `String` | 必需 | 在此描述你想要创建的图像。对于 `dall-e-2`，最大长度为 1000 个字符；对于 `dall-e-3`，最大长度为 4000 个字符。 |
| `model` | `String` 默认值：`"dall-e-2"` | 可选 | 用于生成图像的模型名称。兼容的模型包括：`"dall-e-2"`、`"dall-e-3"`。 |
| `n` | `integer` 或 `null` 默认值：`1` | 可选 | 请求创建的图像数量。必须在 1 到 10 之间。注意：由于 `dall-e-3` 的复杂性，OpenAI 可能会将你的请求限制为单张图像。 |
| `quality` | `String` 默认值：`"standard"` | 可选 | 允许你指定生成图像的质量。此参数仅对 `dall-e-3` 有效。可接受的值有：`"standard"`、`"hd"`。 |
| `size` | `String` 或 `null` 默认值：`"1024x1024"` | 可选 | 生成图像的尺寸。`dall-e-2` 可用的图像尺寸有：`"256x256"`、`"512x512"`、`"1024x1024"`。`dall-e-3` 可用的图像尺寸有：`"1024x1024"`、`"1792x1024"`（横向）、`"1024x1792"`（纵向）。 |
| `style` | `String` 默认值：`"vivid"` | 可选 | 允许你指定生成图像的自然程度。此参数仅对 `dall-e-3` 有效。可接受的值有：`"natural"`（适合照片）、`"vivid"`（适合艺术风格）。 |
| `response_format` | `String` 或 `null` 默认值：`"url"` | 可选 | 生成图像的格式。可接受的值有：`"url"`、`"b64_json"`。 |
| `user` | `String` | 可选 | 代表最终用户的唯一标识符，可帮助 OpenAI 监控和检测滥用行为。 |

### 创建图像（JSON）

### 处理响应

成功调用该方法后，API 将返回一个 `Image` JSON 对象。以下是 `Image` 对象的详细说明，它只有一个参数（表 5-3）。

**表 5-3** `Image` 对象的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `url`（或）`b64_json` | `String` | 如果请求中的 `response_format` 为 `"url"`，则这是生成图像的 URL。（或）如果请求中的 `response_format` 为 `"b64_json"`，则这是 base64 编码的 JSON 图像。 |

### 图像

## 使用 DALL·E 模型创建图像

从表 5-2 和表 5-3 可以看出，使用 DALL·E 模型创建图像是一个非常直接的过程。因此，我们在清单 5-9 中的代码展示了如何在 JavaScript 中以编程方式创建图像。

```
const { OpenAI } = require("openai");
require("dotenv").config();
async function main() {
  // 使用你的 API 密钥设置 OpenAI 客户端
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const response = await openai.images.generate({
    model: "dall-e-3",
    n: 1,
    // 图像的提示词
    prompt:
      "a 35mm macro photo of 3 cute Rottweiler puppies with no collars laying down in a field",
    size: "1024x1024",
  });
  image_url = response.data[0].url;
  console.log(image_url);
}
main();
```

清单 5-9 使用 DALL·E 模型通过 JavaScript 创建图像

好了，我们来分解一下这个代码示例中发生的事情。我们使用 OpenAI API 中的 `openai.images.generate()` 函数，根据我们的提示词生成图像。

- 定义了一个 `prompt`。在本例中，提示词描述了所需的图像：“a 35mm macro photo of 3 cute Rottweiler puppies with no collars laying down in a field.”

- 指定了生成图像的所需 `size` 为 `"1024x1024"`。

- 使用提供的 `prompt`、`size` 和 `model` 参数调用 `openai.images.generate()` 函数。该函数使用指定的模型（`"dall-e-3"`）根据输入的提示词生成图像。

- 响应对象（称为 `ImageResponse`）包含有关生成图像的信息，包括其 URL。

- 最后，代码打印出生成图像的 URL。

总之，这段代码使用 DALL·E 模型，根据提供的提示词生成了一张三只可爱的罗威纳幼犬躺在田野里的图像，然后打印出可以访问该生成图像的 URL。

## 可视化播客

现在我们有了使用 DALL·E 模型创建图像所需的代码，图 5-2 展示了根据前面清单 5-8 中的文本提示词生成的图像。

![img/625846_1_En_5_Fig2_HTML.jpg](img/625846_1_En_5_Fig2_HTML.jpg)

一个长着卷曲长发的小女孩赤脚坐在郁郁葱葱的绿色花园的草地上。她穿着一件粉色毛衣和一条绿色裙子。她身边有一只毛茸茸的黑色狗，吐着舌头，看起来很满足。背景是鲜艳的绿色植物和紫色的花朵，营造出一种宁静而迷人的氛围。

**图 5-2** AI 生成的图像。DALL·E 根据“This American Life”播客第 811 集生成的女孩和她的狗的图像

## DALL·E 提示词工程与最佳实践

现在，使用 DALL·E 创建图像需要提示词工程，才能获得一致、理想的结果。最好尝试不同的提示词来练习，看看哪些对你和你的用例有效。也许你更喜欢绘画而不是 3D 效果的图像？也许你需要照片而不是数字艺术？也许你希望图像是特写镜头而不是肖像？有很多可能性需要考虑。

无论你的用例是什么，这里有两条黄金法则，可以帮助你充分利用 DALL·E 提示词。

### DALL·E 黄金法则 #1：熟悉 DALL·E 可以生成的图像类型

首先也是最重要的，DALL·E 需要理解的最重要的事情之一是需要生成的图像类型。以下是 DALL·E 能够创建的几种最常见的图像类型列表：

- 3D 渲染
- 绘画
- 抽象画
- 表现主义油画
- 油画（任何已故艺术家的风格）
- 油画棒
- 数字艺术
- 照片
- 照片级真实感
- 超写实
- 霓虹照片
- 35 毫米微距照片
- 高质量照片
- 剪影
- 蒸汽波
- 卡通
- 毛绒玩具
- 大理石雕塑
- 手绘草图
- 海报
- 铅笔与水彩
- 合成波
- 漫画风格
- 手绘

### DALL·E 黄金法则 #2：详细描述你希望出现在前景和背景中的内容

我再怎么强调也不为过：为了获得一致、理想的结果，你需要对 DALL·E 进行详细描述。这听起来可能有点奇怪，但向 DALL·E 描述图像的最佳方式，就像你向另一个人描述一个梦一样。

所以，作为我们之间的一个思维练习，试着描述你最近做的一个梦。当你描述梦中的人物、地点和事物时，你脑海中会浮现出你记得的最重要的事情，以及你所感受到的体验。当你向另一个人描述事情时，微小的细节开始浮现，例如：

- 有多少人在场（如果有的话）？
- 人或动物处于什么姿势？站着、坐着还是躺着？
- 场景和背景中有什么东西？
- 什么东西让你印象深刻？声音？气味？颜色？
- 你感觉如何？快乐、诡异、兴奋？
- 感觉是什么时间？早晨、中午、夜晚？

如果你能向另一个人描述一个梦，那么你应该毫无问题地向 DALL·E 描述你想要的东西。

## 结论

在本章中，我们收获颇丰！仅用几个脚本，我们就创建了一个播客可视化工具。

- 我们创建并使用了 `audio-splitter` 脚本，它作为我们的一个实用工具。如果你有一个音频文件，其大小超过了 Whisper 模型的限制，这个脚本会为你提供一个包含多个较小音频文件的文件夹，以便发送给 Whisper。
- 我们创建了一个脚本，用于调用 `audio-splitter` 并将文件夹中的文件发送给 Whisper 进行转录。你唯一的限制是能够发送给 Whisper 模型的请求数量。
- 我们进行了一些提示工程，以便根据转录内容获取播客中图像的描述性提示。
- 最后，我们创建并使用了 `DALLE-model`，它接收调用 `gpt-4-o-mini` 模型生成的提示，并生成一张能够直观代表该播客剧集的图像。

## 留给读者的练习

显然，这里还有一些额外的工作可以做，这些步骤将留给你（读者）去完成，例如：

- `audio-splitter` 脚本是我们与 FFmpeg 交互的 JavaScript 接口。FFmpeg 不仅能分割音频文件，还能对媒体文件做更多处理，例如格式转换和重新编码。请尝试找出 Whisper 支持的媒体格式中，哪种格式的音频文件最小。提示：肯定不是 WAV 格式。

如果你计划创建一个能根据最终用户的文本提示自动生成图像的应用或服务，那么你肯定需要更新 DALL-E 脚本，以确保在请求中跟踪并提供 `user` 参数。这是因为你的最终用户有可能通过你的 API 密钥生成有害图像。请记住，你拥有 OpenAI 的 API 账户，而他们没有！因此，你需要留意，是否要终止与那些通过你的服务违反 OpenAI 内容规则的用户之间的业务关系。