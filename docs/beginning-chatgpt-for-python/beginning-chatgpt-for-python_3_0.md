# 2. 使用 ChatGPT 作为你的 Python 结对编程伙伴

我是 XP（极限编程）某些实践的忠实粉丝，尤其是结对编程。无论你喜欢哪种风格的结对编程，它都涉及两名工程师坐在同一屏幕前，共同解决同一个问题。你获得的最大好处之一就是为问题带来新的视角，当然，现在有两名工程师“接触”过代码库，而不是只有一名。有时，你可以让一名工程师编写代码，另一名编写测试和注释。无论你如何划分，这都是好事。



## 概述

本章将引导您获取并测试您的 API 密钥，让您熟悉如何调用 OpenAI Python API 来使用 ChatGPT，并介绍如何将其应用于其他模型。此外，我们将把 ChatGPT 作为结对编程伙伴，开发一个应用程序，该程序能够接收城市名称和您希望到达工作地点的时间，然后根据交通状况为您提供天气和预计到达时间！听起来很激动人心吧？那么，让我们马上开始吧。

## 使用 `pip` 安装（或更新）OpenAI 库

要使用 Python 的 OpenAI 库，您的机器上至少需要安装 Python 3.7.1。要检查您安装的 Python 版本，只需打开一个终端窗口，然后执行：

```
python3 -v
```

如果已安装 Python 3，终端将显示您拥有的版本。当然，如果终端提示您使用的是不兼容的版本（即低于 3.7.1），那么您应该更新机器上的 Python。此外，如果您根本没有安装 Python 3，那么您应该在继续之前安装它。

现在您已经满足了所有先决条件，是时候安装 OpenAI Python 库本身了。回到您的终端窗口，执行以下命令：

```
pip install --upgrade openai
```

上面的命令将为您安装 OpenAI 库（如果尚未安装），如果已安装，则会将其更新到最新版本。

## 设置 API 密钥的三种方法

在设置 OpenAI API 的 API 密钥时，有多种方法可供选择，每种方法都针对特定的项目需求和安全考虑而设计。

### 选项 #1：设置系统级环境变量

环境变量方法会建立一个系统级的环境变量来存储 API 密钥。这为密钥管理提供了一个集中点，简化了跨不同项目的部署。

让我们看看具体的操作步骤。

#### 对于 Mac OS

首先，打开终端。您可以在“应用程序”文件夹中找到它，或使用聚焦搜索（Command + Space）进行搜索。

接下来，编辑您的 bash 配置文件。对于较旧的 MacOS 版本，您需要使用命令 `nano ~/.bash_profile`。较新 MacOS 版本的用户需要使用 `nano ~/.zshrc`。这将在文本编辑器中打开配置文件。

现在，让我们添加您的环境变量。在编辑器中，添加下面这一行，将 `'your-api-key-here'` 替换为您的实际 API 密钥（不要包含单引号）。

```
export OPENAI_API_KEY='your-api-key-here'
```

让我们保存并退出，按 `Ctrl+O` 写入更改，然后按 `Ctrl+X` 关闭编辑器。

现在，您需要加载您的配置文件，对于较旧的 Mac OS 版本使用 `source ~/.bash_profile`，对于较新的 Mac OS 版本使用 `source ~/.zshrc`。这将加载更新后的配置文件。

最后，我们将验证一切是否正确完成。在终端中，输入 `echo $OPENAI_API_KEY`。如果一切顺利，它将显示您的 API 密钥，确认设置成功。

```
echo $OPENAI_API_KEY
```

#### 对于 Windows

首先打开命令提示符。您可以在开始菜单中搜索 "cmd" 来找到它。

现在，我们将使用以下命令在当前会话中设置环境变量，将 `'your-api-key-here'` 替换为您的实际 API 密钥。此命令为当前会话设置 `OPENAI_API_KEY`。

```
setx OPENAI_API_KEY 'your-api-key-here'
```

您可以通过系统属性添加变量来使设置永久生效：

*   右键单击“此电脑”或“我的电脑”，然后选择“属性”。
*   点击“高级系统设置”。
*   点击“环境变量”按钮。
*   在“系统变量”部分，点击“新建...”，然后输入 `OPENAI_API_KEY` 作为变量名，输入您的 API 密钥作为变量值。

为确保一切正常工作，请重新打开命令提示符并输入以下命令来验证设置。它应该会显示您的 API 密钥。

```
echo %OPENAI_API_KEY%
```

#### 对于 Linux

要为当前会话设置环境变量，请打开一个终端窗口并使用 `export` 命令。将 `'your-api-key-here'` 替换为您的实际 API 密钥。

```
export OPENAI_API_KEY='your-api-key-here'
```

要使环境变量在会话之间持久化，您可以将其添加到 shell 的配置文件中，例如 Bash 的 `~/.bashrc`。操作方法如下：

在文本编辑器中打开配置文件。例如：

```
nano ~/.bashrc
```

在文件末尾添加以下行：

```
export OPENAI_API_KEY='your-api-key-here'
```

保存文件并退出文本编辑器。

要立即应用更改，您可以关闭并重新打开终端，或者运行：

```
source ~/.bashrc
```

要验证环境变量是否正确设置，您可以在终端中 `echo` 其值。此命令应显示您的 API 密钥：

```
echo $OPENAI_API_KEY
```

### 选项 #2：创建 `.env` 文件

使用系统级环境变量非常适合让机器上的任何应用程序或脚本都能访问 API 密钥。但是，如果您的用例更简单一些，我们可以简单地创建一个仅在特定程序或脚本范围内可访问的局部变量。这在不同项目需要不同密钥的情况下也很有用，因此可以防止密钥使用冲突。让我们开始吧！

我们将首先创建一个本地的 `.env` 文件。此文件将存放您的 API 密钥，确保它仅被指定的项目使用。导航到您打算创建 `.env` 文件的项目文件夹。

**注意：** 为防止您的 `.env` 文件通过版本控制被意外共享，请在项目的根目录中创建一个 `.gitignore` 文件。添加一行 `.env` 以确保您的 API 密钥和其他敏感信息的机密性。

接下来，使用终端或 IDE 创建 `.gitignore` 和 `.env` 文件。复制您的 API 密钥，并将 `'your-api-key-here'` 替换为您的实际 API 密钥（不要包含单引号）。

此时，您的 `.env` 文件应如下所示：

```
OPENAI_API_KEY='your-api-key-here'
```

最后，您可以使用以下代码片段将 API 密钥导入到您的 Python 代码中。

```
import os
from dotenv import load_dotenv
from openai import OpenAI
### 从 .env 加载环境变量
load_dotenv()
### 使用环境变量中的 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
清单 2-1
将您的 .env 文件导入到 Python 应用程序中
```



### 选项 #3：在应用程序中直接硬编码 API 密钥（请谨慎使用）

出于安全原因，不建议长期使用最后这种方法。但是，为了了解其工作原理，我们将介绍如何将 API 密钥硬编码到应用程序中，以便快速测试你的 API 密钥是否正常工作。

首先，你需要在 Python 代码中将 API 密钥赋值给一个变量。将 `"YOUR_API_KEY"` 替换为你从 OpenAI 收到的实际 API 密钥。请确保此 API 密钥安全存放，不要公开分享。

接下来，你需要在 Python 脚本中初始化 OpenAI 客户端。这通过实例化 `OpenAI` 类并将 `api_key` 参数设置为该 API 密钥来完成。通过在初始化时提供 API 密钥，你可以让 OpenAI 客户端访问 OpenAI API 提供的服务。此步骤确保你的 Python 脚本能够使用指定的 API 密钥与 OpenAI API 进行通信。

```python
from openai import OpenAI
#### API 密钥
API_KEY = "your-api-key-here"
#### 使用硬编码的 API 密钥初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY)
清单 2-2
将你的 API 密钥直接编码到应用程序中
```

现在，让我们使用 OpenAI API 创建第一个应用程序，并通过获取 OpenAI API 可用模型列表来同时测试该密钥。

> **注意**  
> 从现在开始，代码示例将通过本地的 `.env` 文件访问我们的 API 密钥。

## 创建你的第一个 Python ChatGPT 应用程序：`model_lister.py`

实际上，我们将在这里同时完成两项任务。我们将使用 OpenAI API 创建一个基本的 Python 应用程序，并在此过程中验证我们是否已正确获取 API 密钥。因此，不言而喻，如果你还没有这样做，请按照第 1 章的说明创建你的 OpenAI 开发者帐户并获取 API 密钥。接下来，本书中的所有代码示例都需要一个有效的 API 密钥。

### 使用 `OpenAI.models.list()` 获取可用模型列表

我们可以调用的最基本（但也至关重要）的功能之一位于 `Model` 类中。你可能会问，为什么？`Model` 类允许你获取当前开发者可通过 Python API 使用的所有 AI 模型的列表。

成功获取 `OpenAI` 和 `Model` 对象的实例后，API 将提供 OpenAI API 中所谓的 `SyncPage`，它本质上是一个字典，其结构如表 2-1 所示。

### 处理响应

**表 2-1** Model 对象的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `object` | 字符串 | 始终返回字面量 `"list"`。 |
| `data` | 字典 | OpenAI 提供的 AI 模型字典。 |
| `↳ id` | 字符串 | AI 模型的唯一 ID，本质上是模型的完整名称。 |
| `↳ object` | 字符串 | 始终返回字面量 `"model"`。 |
| `↳ created` | 整数 | 模型的创建日期。 |
| `↳ owned_by` | 字符串 | 拥有该模型的组织名称。 |

> **注意**  
> 由于对象可以包含字典（在表格中可能难以表示），我们使用以下符号 `↳` 来表示字典的元素。从表 2-1 中可以看出，`id`、`object`、`created` 和 `owned_by` 都是响应中 `data` 字典的元素。

现在我们已经了解了 `Model` 对象的详细信息，接下来讨论如何测试我们在第一章中获取的 API 密钥。实际上有几种方法可以做到这一点。

### 使用你的 API 密钥通过 OpenAI API 获取可用模型列表

将 API 密钥设置在本地 `.env` 文件中后，我们将使用以下代码获取 OpenAI API 中可用的模型列表，然后将其打印到终端。

```python
"""用于与 OpenAI API 交互以列出可用模型的模块"""
import os
from dotenv import load_dotenv
from openai import OpenAI
### 从 .env 加载环境变量
load_dotenv()
### 使用环境变量中的 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
models_list = client.models.list()
print(models_list)
清单 2-3
通过调用 models.list() 方法并使用 model_lister.py 获取 OpenAI API 可用模型列表
```

运行清单 2-3 中的代码后，清单 2-4 是你应该会看到的截断响应。

```
SyncPageModel, Model(id='dall-e-2', created=1698798177, object='model', owned_by='system'), Model(id='gpt-3.5-turbo-0125', created=1706048358, object='model', owned_by='system'), Model(id='text-embedding-ada-002', created=1671217299, object='model', owned_by='openai-internal'), Model(id='tts-1-hd-1106', created=1699053533, object='model', owned_by='system'), Model(id='tts-1-hd', created=1699046015, object='model', owned_by='system'). . .
清单 2-4
运行我们的 model_lister.py 应用程序后得到的难看响应
```

### 获取更美观的模型列表

清单 2-4 只是一个部分列表，因为可供开发者使用的模型数量非常庞大！不过，好消息是，完整的响应以表格形式提供在附录 1 中。

所以，首先，这很棒。我们知道我们的 API 密钥有效！但是，你可能没有注意到，我们得到的响应相当难看。为简洁起见，清单 2-4 已被缩短，但很明显，当我们只想知道 OpenAI API 中的所有模型时，没有人愿意阅读那一大段文本。

如果我们能在运行代码后得到更清晰的响应，那就太好了。那么，让我们添加一些代码来格式化响应。

```python
"""用于与 OpenAI API 交互以列出可用模型的模块"""
import os
from dotenv import load_dotenv
from openai import OpenAI
### 从 .env 加载环境变量
load_dotenv()
### 使用环境变量中的 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
models_list = client.models.list()
#### 遍历列表并打印模型信息
for model in models_list.data:
print(f"模型 ID: {model.id}")
print(f"创建时间: {model.created}")
print(f"对象: {model.object}")
print(f"拥有者: {model.owned_by}")
print("\n============================\n")  # 分隔符，提高可读性
清单 2-5
使用 model_lister_pretty.py 清理从 OpenAI API 获得的响应
```

清单 2-6 是运行 `model_lister_pretty.py` 后的响应。

```
模型 ID: text-embedding-ada-002
创建时间: 1671217299
对象: model
拥有者: openai-internal
==================================================
模型 ID: gpt-3.5-turbo-instruct
创建时间: 1692901427
对象: model
拥有者: system
==================================================
模型 ID: dall-e-3
创建时间: 1698785189
对象: model
拥有者: system
清单 2-6
使用 model_lister_pretty.py 从 OpenAI 获得的清理后响应
```



## 等等，我的提示词里有多少个 Token？

在某个时刻，你开始思考计划发送给 ChatGPT 的提示词，并会仔细考虑所用模型的 Token 限制（以及成本）。以防你忘记，请务必参考表 1-1，其中列出了模型及其 Token 价格。此外，OpenAI 还创建了一个简单易用的网站，可以让你查看提示词中包含多少个 Token，如下图所示。

*ChatGPT Token 计数器*

[`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer)

![](img/615442_1_En_2_Fig1_HTML.jpg)

**图 2-1** ChatGPT Token 计数器可以快速统计你提示词中的 Token 数量

## 使用 ChatGPT 作为结对编程伙伴构建天气应用

现在我们已经确认 API 密钥可以正常工作，是时候开始用 ChatGPT 编写我们的第一个项目了。这是激动人心的部分！我们将使用 ChatGPT 帮助我们创建一个 Python 应用程序，它能提供天气信息以及从家到工作地点的通勤时间。听起来很厉害，对吧？

### 设置提示词

让我们问问 ChatGPT 应该如何开始。现在是练习提示词工程的好时机。我们将向 ChatGPT 说明我们的需求，看看它会给出什么方案。这是我们使用的提示词。

```
如何用 Python 制作一个简单的应用程序，让我输入城市名称，然后使用 AccuWeather 的 API 获取天气信息？
清单 2-7
提示 ChatGPT 帮助我们制作一个用于提高生产力的天气应用
```

这是我们从回复中得到的代码。

```
要创建一个简单的 Python 应用程序，让你输入城市名称并通过 AccuWeather 的 API 获取天气信息，可以按照以下步骤操作：
- 在 AccuWeather 开发者门户注册并获取 API 密钥：https://developer.accuweather.com/。
- 如果尚未安装 requests 库，请先安装。可以通过 pip 安装：
pip install requests
- 编写一个 Python 脚本来与 AccuWeather API 交互。以下是一个基本示例：
import requests
def get_weather(accuweather_api_key, city):
base_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
params = {
"apikey": accuweather_api_key,
"q": city,
}
response = requests.get(base_url, params=params)
location_data = response.json()
if location_data:
location_key = location_data[0]["Key"]
weather_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
params = {
"apikey": accuweather_api_key,
}
response = requests.get(weather_url, params=params)
weather_data = response.json()
if weather_data:
weather_text = weather_data[0]["WeatherText"]
temperature = weather_data[0]["Temperature"]["Metric"]["Value"]
print(f"当前{city}的天气是{weather_text}，温度为{temperature}°C。")
else:
print("未找到天气数据。")
else:
print("未找到位置。")
if __name__ == "__main__":
accuweather_api_key = "api-key"
city = input("请输入你的城市：")
get_weather(accuweather_api_key, city)
清单 2-8
从 ChatGPT 获取回复以构建我们的天气应用 accuweather_forecaster.py
```

### 使用 AccuWeather API 创建 `accuweather_forecaster.py`

那么，让我们按照从 ChatGPT 获得的步骤操作。首先，我们需要在 AccuWeather 上注册一个账户。访问 [`developer.accuweather.com/`](https://developer.accuweather.com/) 并点击注册。

![](img/615442_1_En_2_Fig2_HTML.jpg)

**图 2-2** AccuWeather 开发者主页

登录后，你会注意到菜单发生了变化，现在可以导航到 **My Apps**。

![](img/615442_1_En_2_Fig3_HTML.jpg)

**图 2-3** 导航到 AccuWeather 开发者门户的 My Apps 选项卡

进入 **My Apps** 选项卡后，你需要 **Add a new App** 以获取一个 API 密钥，用于我们的应用程序。

![](img/615442_1_En_2_Fig4_HTML.jpg)

**图 2-4** 在 AccuWeather 上创建新应用

![](img/615442_1_En_2_Fig5_HTML.jpg)

**图 2-5** 为我们的 AccuWeather 应用添加规格说明

在设置过程中，你需要为应用命名，并回答一些简单的问题，例如 API 将在何处使用以及你计划用 API 创建什么。如图 2-5 所示，我们将 AccuWeather 应用命名为“Weather Tracker”；但这并不是我们实际 Python 应用 `accuweather_forecaster.py` 的名称。

名称不必一致。你可以在 AccuWeather 中随意命名你的应用。这里需要启用的最重要配置是，系统会要求你指定打算使用的产品。**请务必启用 Core Weather Limited Trial。**

选择 Core Weather 和 MinuteCast 的区别在于，一个应用能正常工作，而另一个则不能，并且需要花很长时间才能找出原因（你能看出这是一次令人难忘的经历吗？）。

你的应用可能需要一些时间才能获得批准，但通常这个过程非常快。完成后，你会在 **My Apps** 页面上看到你的新应用，其中包含你的 API 密钥！任务完成。

![](img/615442_1_En_2_Fig6_HTML.jpg)

**图 2-6** 在 AccuWeather 开发者门户上查看你新创建的应用



#### 通过提示工程优化我们的提示

回到 ChatGPT 提供的代码示例，我们可以在代码中需要的位置提供 API 密钥。为清晰起见，我们将 API 密钥变量重命名为`accuweather_api_key`。如果需要回顾，以下是我们需要输入 AccuWeather API 密钥的代码行。

```
if __name__ == "__main__":
    accuweather_api_key = "your-accuweather-api-key-here"
    city = input("Enter your city: ")
    get_weather(accuweather_api_key, city)
```

现在，所有内容整合在一起后，运行代码时，您应该会看到一个提示，要求输入城市名称。

```
Enter your city:
```

一旦您提供有效的城市名称，一切将顺利运行，您将得到类似如下的响应（显然，预报会根据您输入的城市以及运行当天的天气情况而变化）。

```
The current weather in New York City is Sunny with a temperature of 5.6°C.
```

通过运行代码，我们得到了纽约市天气的基本描述，但仍有几个方面可以改进。

AccuWeather 为各种天气状况提供了图标。目前，我们可以看到纽约的天气是晴天，但我们也可以获取更长的描述，说明当天剩余时间的天气情况。美国人可能更倾向于以华氏度查看温度（作为美国作者，我们也选择这样做）。因此，让我们请 ChatGPT 帮助我们进行这些修改。

为了让 ChatGPT 保留上下文，我们将在同一个聊天会话中向 ChatGPT 提出以下提示。

```
This code works, however, I'd like to make the following changes:
- get the temperature in Fahrenheit
- make an API call to get an icon number and an icon url to print in the response
- make an API call to get a weather description in the response
```

这是我们从 ChatGPT 得到的回答。

```
Sure, here's the modified version of the code to accommodate your requirements:
```python
import requests

def get_weather(accuweather_api_key, city):
    base_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
    params = {
        "apikey": accuweather_api_key,
        "q": city,
    }
    response = requests.get(base_url, params=params)
    location_data = response.json()
    if location_data:
        location_key = location_data[0]["Key"]
        weather_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
        params = {
            "apikey": accuweather_api_key,
        }
        response = requests.get(weather_url, params=params)
        weather_data = response.json()
        if weather_data:
            weather_text = weather_data[0]["WeatherText"]
            temperature_imperial = weather_data[0]["Temperature"]["Imperial"]["Value"]

            # Fetching icon number and icon URL
            icon_number = weather_data[0]["WeatherIcon"]
            icon_url = f"http://developer.accuweather.com/sites/default/files/{icon_number:02d}-s.png"
```



##### 调用 API 获取天气头条

```python
headline_url = f"http://dataservice.accuweather.com/forecasts/v1/daily/1day/{location_key}"
response = requests.get(headline_url, params=params)
headline_data = response.json()
if headline_data and "Headline" in headline_data:
    headline_text = headline_data["Headline"]["Text"]
    print(f"The current weather in {city} is {weather_text} with a temperature of {temperature_imperial}°F.")
    print(f"Weather Headline: {headline_text}")
    print(f"Weather Icon: {icon_url}")
else:
    print("Weather headline not found.")
else:
    print("Weather data not found.")
else:
    print("Location not found.")
if __name__ == "__main__":
    accuweather_api_key = "your-accuweather-api-key-here"
    city = input("Enter your city: ")
    get_weather(accuweather_api_key, city)
```

此修改后的代码版本：
- 将温度从摄氏度转换为华氏度。
- 获取天气图标编号和描述。
- 打印天气图标编号并构建图标 URL。

请务必将 `"your-accuweather-api-key-here"` 替换为你自己的 AccuWeather API 密钥。

**清单 2-13** ChatGPT 对我们 `accuweather_forecaster.py` 应用程序中代码修改的响应

当我们运行这段代码时，得到了我们想要的结果。我们再次输入一个有效的城市名称（本例中为纽约市），得到了如下响应。

```
Weather in New York City:
Temperature: 45.0°F
Weather Text: Sunny
Weather Headline: Expect rainy weather late tomorrow night through late Thursday night
Icon Number: 1
Icon URL: http://developer.accuweather.com/sites/default/files/01-s.png
```

**清单 2-14** 在 `accuweather_forecaster` API 密钥中运行修改后的代码后，AccuWeather API 的响应

现在，我们的代码完全按照我们的意愿运行，但仍有改进空间。如果你使用的 IDE 带有任何类型的 linting 工具，你可能会看到当前运行的代码存在不少 linting 错误。以下是当前代码可能显示的一些 linting 错误。

```
Redefining name 'city' from outer scope (line 45)
Redefining name 'city' from outer scope (line 45)
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing module docstring
Missing function or method docstring
Line too long (102/100)
Line too long (103/100)
Line too long (121/100)
Constant name "accuweather_api_key" doesn't conform to UPPER_CASE naming style
```

**清单 2-15** IDE 指出的 Linting 错误

> **注意：** 那么，什么是 linting？Linting 是分析代码错误的过程，但主要针对风格不一致的问题。它有助于维护代码质量，强制执行编码标准，并通过使用通用约定使代码更具可读性。

为了清理这些问题，我们再次在同一个聊天对话中向 ChatGPT 提交以下内容。

```
My IDE is making me aware of the following linting errors. Help me modify my code to comply to the styling standards.
###
Redefining name 'city' from outer scope (line 45)
Redefining name 'city' from outer scope (line 45)
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
Missing module docstring
Missing function or method docstring
Line too long (102/100)
Line too long (103/100)
Line too long (121/100)
Constant name "accuweather_api_key" doesn't conform to UPPER_CASE naming style
```

**清单 2-16** 提示 ChatGPT 帮助我们解决 Linting 错误

这是我们从 ChatGPT 得到的响应。

```
To address the linting errors and improve the code's compliance with styling standards, you can make the following modifications:
- Add module docstring to describe the purpose of the script.
- Add function docstring to describe the purpose of the `get_weather` function.
- Rename the `accuweather_api_key` variable to `ACCUWEATHER_API_KEY` to comply with the UPPER_CASE naming convention for constants.
- Remove redefining `city` from the outer scope by renaming the input parameter of the `get_weather` function.
- Add a timeout argument to the `requests.get` calls to prevent hanging indefinitely.
- Split long lines to comply with the maximum line length.
Here's the modified code:
```

```python
"""
Script to fetch current weather information using AccuWeather API.
"""
import requests

def get_weather(accuweather_api_key, city):
    """
    Get current weather information for a given city using AccuWeather API.

    Args:
        accuweather_api_key (str): AccuWeather API key.
        city (str): Name of the city for which weather information is requested.

    Returns:
        None
    """
    base_url = "http://dataservice.accuweather.com/locations/v1/cities/search"
    params = {
        "apikey": accuweather_api_key,
        "q": city,
    }
    response = requests.get(base_url, params=params, timeout=10)
    location_data = response.json()
    if location_data:
        location_key = location_data[0]["Key"]
        weather_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
        params = {
            "apikey": accuweather_api_key,
        }
        response = requests.get(weather_url, params=params, timeout=10)
        weather_data = response.json()
        if weather_data:
            weather_text = weather_data[0]["WeatherText"]
            temperature_imperial = weather_data[0]["Temperature"]["Imperial"]["Value"]

            # Fetching icon number and icon URL
            icon_number = weather_data[0]["WeatherIcon"]
            icon_url = (
                f"http://developer.accuweather.com/sites/default/files/"
                f"{icon_number:02d}-s.png"
            )

            # Making API call to get weather headline
            headline_url = (
                f"http://dataservice.accuweather.com/forecasts/v1/daily/1day/"
                f"{location_key}"
            )
            response = requests.get(headline_url, params=params, timeout=10)
            headline_data = response.json()
            if headline_data and "Headline" in headline_data:
                headline_text = headline_data["Headline"]["Text"]
                print(f"The current weather in {city} is {weather_text} "
                      f"with a temperature of {temperature_imperial}°F.")
                print(f"Weather Headline: {headline_text}")
                print(f"Weather Icon: {icon_url}")
            else:
                print("Weather headline not found.")
        else:
            print("Weather data not found.")
    else:
        print("Location not found.")

if __name__ == "__main__":
    ACCUWEATHER_API_KEY = "your-accuweather-api-key"
    user_city = input("Enter your city: ")
    get_weather(ACCUWEATHER_API_KEY, user_city)
```

**清单 2-17** 从 ChatGPT 获得响应，解决了 `accuweather_forecaster_cleaned_up.py` 中的所有 Linting 错误

这些修改应该能够解决 linting 错误，并提高代码的可读性和对样式标准的符合度。请务必将 `"your-accuweather-api-key-here"` 替换为你实际的 API 密钥。

## 使用 ChatGPT 作为结对编程伙伴，构建一个估算距离和到达时间的应用程序

接下来我们要做的是，开始构建一个能够提供从一个地点到另一个地点（例如，从家到公司）的到达时间和距离的应用程序。我们将使用一个经典方案，深入研究 Google Maps API。



### 使用 Google Maps Platform API 创建项目

大多数人已经拥有 Gmail 账户，但万一你没有，请务必在继续之前创建一个。

Google 为各种应用提供了大量的 API，因此我们需要直接访问 Google Maps 的 API，具体地址是 [`https://developers.google.com/maps/documentation`](https://developers.google.com/maps/documentation)。

![](img/615442_1_En_2_Fig7_HTML.jpg)

图 2-7 Google Maps Platform 主页

在 Google Maps Platform 页面上，点击 **开始使用** 来设置你的账户以使用这些 API。按照 Google 提供的步骤操作后，你将进入图 2-8 所示的页面，在这里你可以看到 Google Maps Platform 提供的不同 API。但首先可能引起你注意的是，你仍然需要 **完成账户设置**。

![](img/615442_1_En_2_Fig8_HTML.jpg)

图 2-8 完成你的 Google Maps Platform 账户设置

完成账户设置需要输入信用卡信息，以便开始免费试用，这将为你提供价值 200 美元的额度，这完全足够我们测试之用。

![](img/615442_1_En_2_Fig9_HTML.jpg)

图 2-9 导航到 Google Maps Platform 上的 API 和服务选项卡

正确设置账户后，你会看到一个欢迎页面。在左侧，你会找到一个菜单图标，点击它可以显示你可以访问的服务列表。你需要导航到 **应用和服务**，然后点击 **库**。

![](img/615442_1_En_2_Fig10_HTML.jpg)

图 2-10 API 库页面

你需要点击 **Maps JavaScript API**，然后 **启用** 它。

![](img/615442_1_En_2_Fig11_HTML.jpg)

图 2-11 启用 JavaScript API

启用 JavaScript API 后，返回两次并再次打开菜单，然后点击 **Google Maps Platform** 以查看 Google Maps 的仪表盘。

从这里，我们将查看另一个侧边菜单，它看起来与之前的类似，但这次我们要再次点击 **应用和服务** 以查看一个不同的页面。在这里，你可以点击 **路线** 来启用它。

![](img/615442_1_En_2_Fig12_HTML.jpg)

图 2-12 启用 Routes API

一旦启用了我们需要的 API，请导航到 **密钥和凭据** 选项卡，并 **创建一个新的 API 密钥**。

![](img/615442_1_En_2_Fig13_HTML.jpg)

图 2-13 Google Maps Platform 上的密钥和凭据页面

现在我们有了 Google Maps API 密钥，就可以利用我们的结对编程伙伴了。实际上，有两种方法可以使用 ChatGPT 来简化代码运行，从而成功检索我们想要的信息：目的地有多远，以及到达那里需要多长时间（这里有一个有趣的事实——Google Maps Routes API 在预估行程时间中考虑了交通趋势，这使得我们的应用程序非常适合提高生产力！）。

### 使用 ChatGPT 将 cURL 命令转换为可在 Python 中使用的代码：灵活程序员的方法

对一些人来说，使用 cURL 可能有点令人生畏，这就是为什么我们提供了两种使用 ChatGPT 帮助我们编写代码来利用 Google Maps Platform API 的方法；一种方法是将 cURL 代码“翻译”成我们可以在 Python 中使用的东西，另一种方法则完全不需要查看任何令人生畏的 cURL 代码。

在第一种方法中，我们实际上不会去查阅 Google Maps Platform 文档，而是直接切入正题，为你提供所需的 cURL 代码，并向你展示如何使用 ChatGPT 将其转换为 Python 程序员可用的代码。不用谢。

以下是来自 Google Maps Platform 文档的 cURL 代码。

```
curl -X POST -d '{
"origin":{
"location":{
"latLng":{
"latitude": 37.419734,
"longitude": -122.0827784
}
}
},
"destination":{
"location":{
"latLng":{
"latitude": 37.417670,
"longitude": -122.079595
}
}
},
"travelMode": "DRIVE",
"routingPreference": "TRAFFIC_AWARE",
"departureTime": "2023-10-15T15:01:23.045123456Z",
"computeAlternativeRoutes": false,
"routeModifiers": {
"avoidTolls": false,
"avoidHighways": false,
"avoidFerries": false
},
"languageCode": "en-US",
"units": "IMPERIAL"
}' \
-H 'Content-Type: application/json' -H 'X-Goog-Api-Key: YOUR_API_KEY' \
-H 'X-Goog-FieldMask: routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline' \
'https://routes.googleapis.com/directions/v2:computeRoutes'
```

清单 2-18 使用 Google Maps Routes API 的 cURL 代码

请记住，在实践提示工程时，最佳实践是使用三个井号（`###`）将指令与内容分开，以获得最佳结果。为此，以下是我们给 ChatGPT 的提示，以便我们可以使用这段 cURL 代码并产生我们想要的结果，我们使用了三个井号来分隔指令和内容，内容即清单 2-18 中展示的 cURL 代码：

```
我正在用 Python 编码，试图构建一个应用程序，该应用程序向 Google Maps API 发出请求，并获取从一个地点到另一个地点所需的时间。这是可以实现该目标的 cURL 代码。请帮助我将其转换为可以在 Python 中使用的东西。
###
[来自清单 2-18 的 cURL 代码]
```

清单 2-19 提示 ChatGPT 以 cURL 命令为上下文提供 Python 代码

这是我们从 ChatGPT 的回复中得到的代码。

```
import requests
import json
#### 定义负载
payload = {
"origin": {
"location": {
"latLng": {
"latitude": 37.419734,
"longitude": -122.0827784
}
}
},
"destination": {
"location": {
"latLng": {
"latitude": 37.417670,
"longitude": -122.079595
}
}
},
"travelMode": "DRIVE",
"routingPreference": "TRAFFIC_AWARE",
"departureTime": "2025-02-17T17:00:00Z",
"computeAlternativeRoutes": False,
"routeModifiers": {
"avoidTolls": False,
"avoidHighways": False,
"avoidFerries": False
},
"languageCode": "en-US",
"units": "IMPERIAL"
}
#### 定义请求头
headers = {
"Content-Type": "application/json",
"X-Goog-Api-Key": "YOUR_API_KEY",
"X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
}
#### 发起请求
response = requests.post("https://routes.googleapis.com/directions/v2:computeRoutes", json=payload, headers=headers)
#### 打印响应
print(response.json())
```

清单 2-20 从 ChatGPT 获得包含可用 Python 代码的回复

**注意**

为了使此代码运行，你必须在 `DEPARTURE-TIME` 参数中设置一个晚于你运行代码时间的日期和时间。换句话说，出发时间不能设置为过去的时间。在此示例中，日期设置为 2025 年的某个特定时间，但如果你在 2025 年之后运行此代码，则需要调整 `DEPARTURE-TIME` 参数中的代码，以便代码按预期运行。

运行此代码后，你应该会得到如下响应。

```
{'routes': [{'distanceMeters': 773, 'duration': '158s', 'polyline': {'encodedPolyline': 'ipkcFjichVzQ@d@gU{E?'}}]}
```

清单 2-21 运行 ChatGPT 代码后的成功响应



### 使用 ChatGPT 避免阅读大量文档和解析任何 cURL 代码

如前所述，Google Maps 平台有*大量*文档需要阅读。如果你粗略浏览一下我们为了到达 Routes API 而经过的其他 API，特别是，你会看到用于提取各种数据的各种 API：空气质量、花粉、鸟瞰图（就像你使用 Google Maps 时看到的那样），以及你可能认为理所当然的 Google Maps 的许多其他功能，它们都有各自的 API。你知道什么是地理编码吗？好吧，如果你自己通读文档，你可能会费一番功夫才能弄明白。

通读文档很烦人。尤其是当我们只想要某个 API 或服务中的几个功能时。在一个完美的世界里，我们甚至不*需要*知道什么是地理编码，就能让我们的代码在不同于 Google 提供的示例代码的各种场景下工作。

我们可能生活在一个不完美的世界，但我们*不必*知道什么是地理编码，至少现在不用了。我们可以简单地使用 ChatGPT 来跳过我们不想阅读的文档。

以下是我们向 ChatGPT 提出的问题，这样你就不必阅读任何 Google Maps Route API 文档就能获得可用的代码。

```
我需要使用 Google Maps 平台的 Routes API 来获取从一个目的地到另一个目的地所需的时间以及两点之间的距离信息。我如何在 Python 中发出 HTTP 请求来获取我正在寻找的信息？
清单 2-22
提示 ChatGPT 以避免阅读文档
```

这是我们从 ChatGPT 的回复中得到的代码。

```
import requests
def get_route_info(google_maps_api_key, origin, destination):
url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={google_maps_api_key}"
response = requests.get(url)
if response.status_code == 200:
route_data = response.json()
if route_data["status"] == "OK":
route_info = route_data["routes"][0]["legs"][0]
distance = route_info["distance"]["text"]
duration = route_info["duration"]["text"]
print(f"Distance: {distance}, Duration: {duration}")
else:
print("Error:", route_data["status"])
else:
print("Error:", response.status_code)
#### 将 'YOUR_GOOGLE_MAPS_API_KEY' 替换为你的实际 Google Maps API 密钥
google_maps_api_key = 'YOUR_GOOGLE_MAPS_API_KEY'
origin = 'New York, NY'  # 示例起点
destination = 'Los Angeles, CA'  # 示例终点
get_route_info(google_maps_api_key, origin, destination)
清单 2-23
从 ChatGPT 获得回复以避免阅读文档
```

如果你运行这段代码，你应该会得到这个非常清晰、非常简单的响应：

```
Distance: 2,789 mi, Duration: 1 day 17 hours
```

## 结论

让我们澄清一个常见的误解并说明真相：ChatGPT 并不是什么读心术的巫师！作为一种 AI 工具，它并不是要取代开发人员和架构师的专业知识——而是为了补充它，并节省大量时间和麻烦，解决普通程序员日常工作中经常遇到的挫折。当然，它可以帮助将你的自然语言请求翻译成代码，但我们不要忘记人类判断的关键作用。最终，由开发者来决定是使用、调整还是丢弃生成的代码。

