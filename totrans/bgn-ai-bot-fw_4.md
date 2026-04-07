# 4. IBM Watson 聊天机器人

Watson 是一个专门为问答类型能力构建的计算机系统。其核心实现使用自然语言处理、信息检索、知识表示、自动推理和开放域问答领域的机器学习技术。它用于快速基于 AI 的解决方案。

在本章中，您将学习如何实现 IBM 的机器人服务，即 Watson，以供您使用（见图 4-1）。为了访问 Watson，您需要在 IBM 云（以前称为 Bluemix）上创建一个账户。然后您将使用 Watson 助手（以前称为对话服务）来创建两个聊天机器人：一个常见问题解答（FAQ）机器人和一个咖啡机器人。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig1_HTML.jpg](img/457478_1_En_4_Fig1_HTML.jpg)

图 4-1

IBM Watson 网页

## 实现 Watson

在本节中，您将学习如何访问 Watson，并了解其各种服务。您通过 IBM 云网站访问 Watson，因此您首先需要通过 IBM 云为 Watson 服务设置一个账户。我们将设置新账户。

### IBM 云

首先，通过打开网页到[`https://www.ibm.com/cloud-computing/in-en/`](https://www.ibm.com/cloud-computing/in-en/) 访问 IBM 云。图 4-2 显示了 IBM 云网站。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig2_HTML.jpg](img/457478_1_En_4_Fig2_HTML.jpg)

图 4-2

IBM 云主页

使用如图 4-3 所示的 IBM ID 在此处登录。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig3_HTML.jpg](img/457478_1_En_4_Fig3_HTML.jpg)

图 4-3

IBM 云登录窗口

登录后，您将看到图 4-4 所示的网页。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig4_HTML.jpg](img/457478_1_En_4_Fig4_HTML.jpg)

图 4-4

IBM 云主控制台窗口

在此页面上，点击右上角的创建资源按钮，如图 4-5 所示，以创建资源并添加新服务。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig5_HTML.jpg](img/457478_1_En_4_Fig5_HTML.jpg)

图 4-5

创建新资源。当我们创建资源时，我们就可以创建会话机器人。

### Watson 助手服务

在本节中，您将了解如何设置 Watson 助手服务。

当我们创建资源时，有很多选项可供选择。您必须在其中添加 Watson 助手服务。如图 4-6 所示，Watson 有很多服务可供选择。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig6_HTML.jpg](img/457478_1_En_4_Fig6_HTML.jpg)

图 4-6

可用的各种 Watson 服务

在本例中，您将选择 Watson 助手，如图 4-7 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig7_HTML.jpg](img/457478_1_En_4_Fig7_HTML.jpg)

图 4-7

选择 Watson Assistant 服务

接下来，点击右下角的创建按钮以创建 Watson Assistant 的服务计划，这显示了 Watson Assistant 服务的不同计划以及免费层级的获取内容，如图 4-8 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig8_HTML.jpg](img/457478_1_En_4_Fig8_HTML.jpg)

图 4-8

Watson Assistant 服务计划

设置很简单。我们需要点击创建，设置过程将向完成方向移动。设置完成后，点击服务的启动工具以开始。

## 创建 FAQ 机器人

在本节中，您将使用 Watson Assistant 创建一个使用 FAQ 进行回答的聊天机器人。点击前一步中的启动工具已打开 IBM Watson Assistant 屏幕，如图 4-9 所示。点击屏幕底部的创建工作区按钮以开始。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig9_HTML.jpg](img/457478_1_En_4_Fig9_HTML.jpg)

图 4-9

开始使用 IBM Watson Assistant

现在，您将创建工作区。点击“工作区”标签，然后在“创建新工作区”部分点击创建按钮，如图 4-10 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig10_HTML.jpg](img/457478_1_En_4_Fig10_HTML.jpg)

图 4-10

创建工作区

在下一屏，输入工作区名称；在这个例子中，输入 *FAQBot* 并输入一个描述，如图 4-11 所示。然后点击创建按钮。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig11_HTML.jpg](img/457478_1_En_4_Fig11_HTML.jpg)

图 4-11

命名工作区

### 为机器人创建意图

下一步是根据您的需求为机器人创建意图。我们需要点击添加意图。此时，Watson 控制台看起来如图 4-12 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig12_HTML.jpg](img/457478_1_En_4_Fig12_HTML.jpg)

图 4-12

创建新的意图

我们的 FAQBot 将基于 IBM 云业务流程。将第一个意图命名为 *Capabilities*，如图 4-13 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig13_HTML.jpg](img/457478_1_En_4_Fig13_HTML.jpg)

图 4-13

我们现在在意图名称选项中将意图的名称添加为“功能”。命名意图。

接下来，您需要添加意图的描述，如图 4-14 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig14_HTML.jpg](img/457478_1_En_4_Fig14_HTML.jpg)

图 4-14

为意图添加描述

然后在“功能”意图中添加示例，如图 4-15 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig15_HTML.jpg](img/457478_1_En_4_Fig15_HTML.jpg)

图 4-15

为能力意图添加用户示例

现在您将创建第二个意图。将此意图命名为 *迁移*，如图 4-16 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig16_HTML.jpg](img/457478_1_En_4_Fig16_HTML.jpg)

图 4-16

创建名为迁移的新意图

创建第三个意图并将其命名为 *User*，如图 4-17 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig17_HTML.jpg](img/457478_1_En_4_Fig17_HTML.jpg)

图 4-17

创建名为 User 的第三个意图

现在创建一个名为 *SSO* 的第四个意图，代表单点登录，如图 4-18 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig18_HTML.jpg](img/457478_1_En_4_Fig18_HTML.jpg)

图 4-18

创建 SSO 意图

现在您已经列出了所有意图，如图 4-19 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig19_HTML.jpg](img/457478_1_En_4_Fig19_HTML.jpg)

图 4-19

所列出的所有意图

### 应用程序的对话流程

在本节中，您将为 FAQBot 创建对话。点击“对话”选项卡，然后点击“创建”按钮，如图 4-20 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig20_HTML.jpg](img/457478_1_En_4_Fig20_HTML.jpg)

图 4-20

对话创建空间。在这里，我们将创建机器人的对话流程，我们需要点击“创建”。

在下一屏幕上，点击“欢迎”以便我们可以在这里开始对话，如图 4-21 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig21_HTML.jpg](img/457478_1_En_4_Fig21_HTML.jpg)

图 4-21

对话工作流程

### 理解流程的意义

现在您已经创建了各种对话，通过创建对话，我们将连接机器人的流程以使其高效运行，您现在可以通过链接整个流程来创建本节中的交互。您将从添加您想要执行的操作的意图开始。

xxxxxxxxxxxxxxx，如图 4-22 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig22_HTML.jpg](img/457478_1_En_4_Fig22_HTML.jpg)

图 4-22

创建一个新节点，我们希望触及每个需要触及的信息，以及完美的流程。

接下来，为迁移意图添加其他意图，如图 4-23 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig23_HTML.jpg](img/457478_1_En_4_Fig23_HTML.jpg)

图 4-23

为迁移意图添加新节点

为用户意图添加一个意图，如图 4-24 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig24_HTML.jpg](img/457478_1_En_4_Fig24_HTML.jpg)

图 4-24

添加新的意图用户

现在添加单点登录，如图 4-25 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig25_HTML.jpg](img/457478_1_En_4_Fig25_HTML.jpg)

图 4-25

添加 SSO 节点

### 测试机器人

在我们第一个机器人的最后一个部分，你将看到如何与机器人交互。点击右上角的“尝试”按钮，访问如图 4-26 所示的“尝试一下”链接。现在让我们尝试一下这个机器人。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig26_HTML.jpg](img/457478_1_En_4_Fig26_HTML.jpg)

图 4-26

测试机器人。这里我们只是在测试机器人如何工作以及流程逻辑是否完美。

在下一节中，你将创建另一个机器人。这个机器人将帮助用户订购咖啡。

## 创建咖啡机器人

在本节中，你将使用 Watson Assistant 创建一个 CoffeeBot。这个机器人将帮助用户订购一杯咖啡。首先，你将创建一个工作区；然后创建新的意图并添加实体。接下来，你将添加对话。最后，你将为意图创建一个嵌套结构。

### 创建工作区

要开始，你将创建一个新的工作区。工作区是从 Watson Assistant 窗口中创建的，并且对于新机器人的创建，这一步是必须的，如图 4-27 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig27_HTML.jpg](img/457478_1_En_4_Fig27_HTML.jpg)

图 4-27

为咖啡机器人创建一个新的工作区

在打开的屏幕上，将聊天机器人的名称命名为 *CoffeeBot*，如图 4-28 所示。然后点击“创建”按钮。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig28_HTML.jpg](img/457478_1_En_4_Fig28_HTML.jpg)

图 4-28

命名工作区

### 与意图一起工作

在本节中，你将开始为机器人的使用创建意图。点击“意图”选项卡，然后点击“添加意图”按钮，如图 4-29 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig29_HTML.png](img/457478_1_En_4_Fig29_HTML.png)

图 4-29

创建 CoffeeBot 的意图

让我们从创建一个名为 Greetings 的意图开始，如图 4-30 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig30_HTML.jpg](img/457478_1_En_4_Fig30_HTML.jpg)

图 4-30

将意图命名为 Greetings

机器人问候用户的方式有很多种。你将使用这个屏幕来选择选项，如图 4-31 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig31_HTML.jpg](img/457478_1_En_4_Fig31_HTML.jpg)

图 4-31

添加用户示例

接下来，你将添加一个名为 BuyCoffee 的意图，如图 4-32 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig32_HTML.jpg](img/457478_1_En_4_Fig32_HTML.jpg)

图 4-32

创建一个名为 BuyCoffee 的新意图

如图 4-33 所示，为此添加一些示例。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig33_HTML.jpg](img/457478_1_En_4_Fig33_HTML.jpg)

图 4-33

添加用户示例

创建另一个意图并命名为 Suggestion，如图 4-34 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig34_HTML.jpg](img/457478_1_En_4_Fig34_HTML.jpg)

图 4-34

创建名为 Suggestion 的新意图

如图 4-35 所示，为此新意图添加一些示例。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig35_HTML.jpg](img/457478_1_En_4_Fig35_HTML.jpg)

图 4-35

为 Suggestion 意图添加用户示例

接下来，添加另一个名为 Yes 的意图，如图 4-36 所示。像之前为其他意图所做的那样，添加用户示例。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig36_HTML.jpg](img/457478_1_En_4_Fig36_HTML.jpg)

图 4-36

为 Yes 意图添加用户示例

如图 4-37 所示，ThankYou 将是下一个意图。为此意图命名并添加示例。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig37_HTML.jpg](img/457478_1_En_4_Fig37_HTML.jpg)

图 4-37

ThankYou 意图和用户示例

下一个意图将是 Cancel，如图 4-38 所示。遵循相同的过程来命名它并添加示例。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig38_HTML.jpg](img/457478_1_En_4_Fig38_HTML.jpg)

图 4-38

添加 Cancel 意图

### 与实体一起工作

在本节中，你将使用实体来使机器人更加结构化。点击实体选项卡，然后点击我的实体以访问如图 4-39 所示的屏幕。点击添加实体按钮。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig39_HTML.jpg](img/457478_1_En_4_Fig39_HTML.jpg)

图 4-39

添加实体

在打开的屏幕上，通过键入 *CoffeeSize* 来命名实体，如图 4-40 所示。然后点击创建实体按钮。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig40_HTML.jpg](img/457478_1_En_4_Fig40_HTML.jpg)

图 4-40

将实体命名为 CoffeSize

在下一屏幕上，选择 CoffeeSize 的值，如图 4-41 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig41_HTML.jpg](img/457478_1_En_4_Fig41_HTML.jpg)

图 4-41

为 CoffeeSize 实体添加值

现在添加一个名为 CoffeeOptions 的实体，如图 4-42 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig42_HTML.jpg](img/457478_1_En_4_Fig42_HTML.jpg)

图 4-42

为 CoffeeOptions 添加同义词

### 与对话框一起工作

在本节中，你将使用对话框来创建机器人与用户之间无缝交互的流程。

从 XXXX 屏幕上，按照图 4-43 所示，执行 XXXXX 操作以添加新的对话框。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig43_HTML.jpg](img/457478_1_En_4_Fig43_HTML.jpg)

图 4-43

添加新的对话框

创建对话框后，工作区看起来如图 4-44 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig44_HTML.jpg](img/457478_1_En_4_Fig44_HTML.jpg)

图 4-44

为机器人添加对话框流程

您必须为机器人添加额外的对话框，以便将事物整合在一起。在这个地方本身我们添加不同的逻辑，以便机器人能够正常工作。您已经创建了订购咖啡的意图以及实体，因此现在您可以使用它们来创建工作流程。

让我们先添加一个节点，其中我们请求购买咖啡，如图 4-45 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig45_HTML.jpg](img/457478_1_En_4_Fig45_HTML.jpg)

图 4-45

为 BuyCoffee 添加新的节点

机器人的内容和响应如图 4-46 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig46_HTML.jpg](img/457478_1_En_4_Fig46_HTML.jpg)

图 4-46

为“购买咖啡”添加响应

接下来，为 CoffeeBot 添加一个新的建议节点，如图 4-47 所示。在这个节点中，您将创建子节点来表示不同咖啡类型的偏好。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig47_HTML.jpg](img/457478_1_En_4_Fig47_HTML.jpg)

图 4-47

添加新的建议节点

### 嵌套意图

本节介绍如何使用嵌套意图，以便您可以匹配流程，即机器人将如何根据我们的意图工作。

在每个节点名称的右侧，您可以看到三个垂直点，如图 4-48 所示。这些点用于创建嵌套子工作流程。XXXXXXXXX

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig48_HTML.jpg](img/457478_1_En_4_Fig48_HTML.jpg)

图 4-48

添加新的子节点

您现在可以看到您为咖啡选项创建的子节点，如图 4-49 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig49_HTML.jpg](img/457478_1_En_4_Fig49_HTML.jpg)

图 4-49

添加嵌套意图

，如图 4-50 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig50_HTML.jpg](img/457478_1_En_4_Fig50_HTML.jpg)

图 4-50

嵌套意图传达了我们在购买或订购咖啡时，针对小、中、大杯的所有选项所采取的行动。

现在下一个节点将用于取消订单。XXXXXXXXX，如图 4-51 所示。

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig51_HTML.jpg](img/457478_1_En_4_Fig51_HTML.jpg)

图 4-51

通过与对话框的对话取消订单状态在这里完成

你将有两个选项：是和否，因此需要进一步嵌套流程。当你把所有节点加起来时，你得到图 4-52 所示的分类结构！

![../images/457478_1_En_4_Chapter/457478_1_En_4_Fig52_HTML.jpg](img/457478_1_En_4_Fig52_HTML.jpg)

图 4-52

CoffeeBot 的完整工作流程

## 结论

在本章中，你学习了如何开始使用 IBM Cloud；请注意，IBM 提供了该服务的免费一个月试用期。

本章还介绍了 IBM Watson。你学习了如何使用 Watson Assistant 创建两个聊天机器人。

在上一章中，你学习关于聊天机器人的旅程将继续。你将创建 TensorFlow 聊天机器人并了解它们的用途。
