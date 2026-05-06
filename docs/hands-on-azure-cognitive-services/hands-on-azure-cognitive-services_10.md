# Azure Bot 服务

人类：我们想要什么？

机器：上下文感知的 NLP 机器人

人类：我们什么时候想要它？

机器：我们想要什么？

自然语言处理 (NLP) 是机器人服务的核心元素。与我们的玩笑不同，我们希望在我们的机器人中开发人工智能 (AI)，以理解对话的上下文。机器人是对话式用户界面，允许你通过不同的模式（例如文本和语音）与机器进行通信。Azure Bot 服务是一个基于云的、托管的、集成的环境，用于支持机器人开发、托管和管理。Azure Bot 服务允许你专注于重要的事情，即对话界面的核心，同时它处理基础设施和相关的工具集。由全面的 Bot Framework 提供支持，Bot 服务附带预构建的模板供你入门，并提供一个基于浏览器的界面，使机器人开发尽可能简单。

在本章中，我们将通过创建 COVID-19 QnA 机器人来深入了解 Bot 服务。你将学习以下主题：

1.  了解 Azure Bot 服务
2.  使用 Azure Bot 服务创建 COVID-19 机器人
3.  使用 Azure Bot Builder SDK^(³²)



## Azure Bot 服务生态系统

Microsoft Bot 生态系统包含几个重要的产品，包括 Microsoft Bot Framework、Azure Bot 服务、语言理解 (LUIS) 和 Azure Health Bot（面向医疗保健的对话式 AI）。

Bot Framework 经历了多次迭代。它提供了一个 SDK 和一套全面的工具来构建对话式 AI。它包含 Bot Framework Composer、Bot Framework Solutions、Botkit、Bot Framework Emulator、Bot Framework Web Chat、Bot Framework Tools、Language Understanding、QnA Maker、Dispatch、Speech Services、Adaptive Cards 和 Analytics。这些主题相当庞大，超出了本书的范围。在本章中，我们将重点介绍如何使用 Azure Bot 服务构建机器人。

## 构建 Azure Service COVID-19 机器人^(³³)

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig1_HTML.jpg](img/499686_1_En_9_Fig1_HTML.jpg)

图 9-1

Azure 门户 – 创建机器人服务实例

1. 从 Azure 门户 (`portal.azure.com`) 开始，搜索 `bot services`，如图 9-1 所示。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig2_HTML.jpg](img/499686_1_En_9_Fig2_HTML.jpg)

图 9-2

Azure 门户 – 创建 Web App Bot

1. 选择 `Bot Services`，您将看到 Microsoft 及市场中提供的不同机器人服务列表。选择 `Web App Bot`，如图 9-2 所示。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig3_HTML.jpg](img/499686_1_En_9_Fig3_HTML.jpg)

图 9-3

Azure 门户 – 创建 Web App Bot

1. 点击 `Create` 创建一个 Web App Bot，它将使用 Azure Bot 服务来帮助构建、部署和管理底层活动。参见图 9-3。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig4_HTML.jpg](img/499686_1_En_9_Fig4_HTML.jpg)

图 9-4

Azure 门户 – 创建 Web App Bot

1. 填写创建机器人的详细信息，包括机器人句柄名称、订阅、资源组、位置和定价层，并选择一个机器人模板。参见图 9-4。请记住，您始终可以使用自己现有的应用服务计划，而无需创建新的计划。

有几个机器人入门套件可供选择，这些套件提供 C# 和 Node.js 版本。Echo Bot 会回显消息，而 basic bot 则演示了 LUIS 技能。参见图 9-5。我们还有虚拟助手、LUIS Bot 和 QnA Bot。在本例中，由于我们专注于创建一个 COVID-19 机器人，因此我们将使用 QnA 模板。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig5_HTML.jpg](img/499686_1_En_9_Fig5_HTML.jpg)

图 9-5

Azure 门户 – 机器人模板

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig6_HTML.jpg](img/499686_1_En_9_Fig6_HTML.jpg)

图 9-6

QnA Maker – 创建知识库

1. 为了使 QnA Bot 正常工作，我们需要创建一个问答知识库 (KB)。可以在此处创建此知识库。访问 [`www.qnamaker.ai/Create`](http://www.qnamaker.ai/Create) 上的 QnA Maker，并使用您的 Azure 帐户登录，即可看到如图 9-6 所示的屏幕截图。点击 `Create a QnA service`。

1. 然后您将看到 Cognitive Services 创建屏幕。QnA Maker 是一种基于云的服务，用于构建对话式 UI。使用您的订阅、资源组和应用服务详细信息填写以下表单（将打开一个新门户窗口以创建 QnA Maker）。点击 `Review + create` 继续。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig7_HTML.jpg](img/499686_1_En_9_Fig7_HTML.jpg)

图 9-7

Azure 门户 – 创建认知服务

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig8_HTML.jpg](img/499686_1_En_9_Fig8_HTML.jpg)

图 9-8

Azure 门户 – 认知服务部署完成

1. 服务创建并部署完成后，您将看到如图 9-8 所示的屏幕。然后进入 QnA Maker 的下一步。

![../images/499686_1_En_9_Chapter/499686_1_En_9_Fig9_HTML.jpg](img/499686_1_En_9_Fig9_HTML.jpg)

图 9-9

QnA Maker – 将 QnA 服务连接到知识库



