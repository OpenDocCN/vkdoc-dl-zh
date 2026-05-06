# 语言 – 理解非结构化文本和模型

一个智能应用程序能够理解用户的输入。用户可以通过键盘、麦克风或使用任何其他外部设备提供输入。这些系统非常智能，可以与用户进行交互。

本章的目标是开始使用自然语言处理（NLP），并创建一个能够与其用户交互的智能应用程序。有很多方法可以训练和准备系统，使其能够与用户交互。

在本章中，我们将涵盖以下主题：

*   创建和理解语言模型
*   使用认知服务进行训练和增强
*   Muppet 模型 – 用于 NLP 的 Transformer
*   使用微调后的 BERT 进行命名实体识别
*   语言 API 总结



## 创建与理解语言模型

大多数企业数据都以非结构化形式存在，这并不令人意外。能够理解文档中词语含义的算法需求量巨大。自 Transformer 网络问世以来，自然语言处理领域在理解、问答、摘要、主题建模、机器翻译、情感分析、文本生成、信息检索和关系抽取等任务上取得了显著进步。这些自然语言处理任务的多样性，正如同我们在企业中处理的文档类型一样丰富。

训练企业级语言模型需要理解多种模态，以及众多文档类型所关联的细微差别。大多数非结构化数据来自电子邮件、维基（SharePoint、Confluence 等）以及其他官方文档，例如客户协议、供应商协议、保密协议、员工合同、贷款和租赁文件、投资与合规文件、修订案、采购订单、工作说明书等。这些文档在命名法、行话和术语上各不相同，在结构和形式上也有所差异。语言模型还必须处理法律文件以及具有治理和审计需求的文档，例如条款与终止、保密性、责任限制、赔偿、管辖法律、转让、通知，以及诸如禁止招揽和竞业禁止等条款。如果在通用环境中应用，提取此类法律条款可能是一项具有挑战性的任务。一个自然语言处理流程包含多个步骤，如图 4-1 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig1_HTML.jpg](img/499686_1_En_4_Fig1_HTML.jpg)

图 4-1

端到端自然语言处理流程——图片由[`https://docs.microsoft.com/en-us/azure/architecture/data-guide/technology-choices/natural-language-processing`](https://docs.microsoft.com/en-us/azure/architecture/data-guide/technology-choices/natural-language-processing)提供

一个典型的语言处理流程处理多模态数据格式和存储，以及 NLP 引擎，后者是整个端到端生命周期的一部分。例如，此类语言模型的一个重要用例可能需要实体识别。这可能超越典型的主语、宾语和动词对，而是涉及业务和合同实体。这些实体包括文档标题、当事方名称和地址、生效日期、条款和法律条款、费用、共付额等。一旦识别并提取出来，你就可以分析这些实体，将其呈现在业务报告中并转化为可执行项。例如，如果你想了解某个特定有限责任公司何时到期（以便向客户发送警报），你可以使用提取的备案日期，并在到期前 30、60 和 90 天创建警报，以通知客户。

随着行业日趋成熟，与 AI 执行的零散任务相比，企业现在寻求的是以结果为导向的端到端解决方案和完全自动化的“作业”（更大的任务组）。在下一个示例中，我们将通过一个演示向你展示这种非结构化文本富化是如何工作的。

## 进行中的富化——JFK 文件演示

为了展示认知服务和 Azure 机器学习的强大功能，微软获取了 JFK 文件的复杂数据集，并整合了一个令人印象深刻的演示（参见[`www.microsoft.com/ai/ai-lab-jfk-files`](http://www.microsoft.com/ai/ai-lab-jfk-files)）。在此示例中，我们将向你展示该演示的工作原理，以及如何为你自己重现它。图 4-2 展示了端到端的富化模型，从存储复杂文件（如照片、手写内容和未分类文档）的 Blob 存储开始。然后，使用认知搜索来提取信息。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig2_HTML.jpg](img/499686_1_En_4_Fig2_HTML.jpg)

图 4-2

用于 JFK 文件的 Azure 搜索——图片由[`https://github.com/microsoft/AzureSearch_JFK_Files`](https://github.com/microsoft/AzureSearch_JFK_Files)提供

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig3_HTML.jpg](img/499686_1_En_4_Fig3_HTML.jpg)

图 4-3

用于 JFK 文件的 Azure 搜索的 ARM 部署——图片由[`https://github.com/microsoft/AzureSearch_JFK_Files`](https://github.com/microsoft/AzureSearch_JFK_Files)提供

1. 为了为 Azure 解决方案实现基础设施即代码，该示例包含了 Azure 资源管理器模板（通常称为 ARM 模板）。在这种情况下，你将部署 ARM 模板以在 Azure 上设置项目，同时执行其余的配置步骤。你可以在[`https://github.com/microsoft/AzureSearch_JFK_Files`](https://github.com/microsoft/AzureSearch_JFK_Files)找到部署链接，如图 4-3 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig4_HTML.jpg](img/499686_1_En_4_Fig4_HTML.jpg)

图 4-4

Azure 门户自定义部署

1. 部署自定义模板的一个步骤是为其创建设置。点击**部署到 Azure**按钮。将打开“自定义部署”屏幕，如图 4-4 所示。然后，你选择并设置区域、资源前缀、托管计划、搜索服务等，如图 4-4 所示。

点击**新建**并选择一个资源组，该资源组是容纳所有相关资源的容器。见图 4-5。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig5_HTML.jpg](img/499686_1_En_4_Fig5_HTML.jpg)

图 4-5

创建新资源

点击**查看 + 创建**以继续，如图 4-6 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig6_HTML.jpg](img/499686_1_En_4_Fig6_HTML.jpg)

图 4-6

点击查看 + 创建

查看订阅信息，然后点击**创建**，如图 4-7 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig7_HTML.jpg](img/499686_1_En_4_Fig7_HTML.jpg)

图 4-7

创建资源的最后一步

一旦部署创建完成，你将看到正在提交部署的通知，如图 4-8 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig8_HTML.jpg](img/499686_1_En_4_Fig8_HTML.jpg)

图 4-8

部署进行中的通知

部署完成后，你将看到部署成功的通知，如图 4-9 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig9_HTML.jpg](img/499686_1_En_4_Fig9_HTML.jpg)

图 4-9

部署成功通知

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig10_HTML.jpg](img/499686_1_En_4_Fig10_HTML.jpg)

图 4-10

打开 Visual Studio



1.  打开 Visual Studio（如图 4-10 所示），然后从 GitHub 上的 [`https://github.com/microsoft/AzureSearch_JFK_Files`](https://github.com/microsoft/AzureSearch_JFK_Files) 克隆存储库。

点击**文件**选项卡，点击**打开**，然后点击**项目/解决方案**（如图 4-11 所示）。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig11_HTML.jpg](img/499686_1_En_4_Fig11_HTML.jpg)

图 4-11

打开项目

克隆存储库后，从 **JfkWebApiSkills** Visual Studio 解决方案文件中打开项目目录（如图 4-12 所示）。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig12_HTML.jpg](img/499686_1_En_4_Fig12_HTML.jpg)

图 4-12

打开 JfkWebApiSkills 解决方案文件

接下来，在解决方案资源管理器中，编辑 **App.config** 文件（如图 4-13 所示）。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig13_HTML.jpg](img/499686_1_En_4_Fig13_HTML.jpg)

图 4-13

选择 App.config 文件

现在，点击 Azure 控制台中的**输出**选项卡，如图 4-14 所示。该选项卡在模板的部署屏幕中可用。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig14_HTML.jpg](img/499686_1_En_4_Fig14_HTML.jpg)

图 4-14

输出选项卡

然后您将看到模板信息，其中包含所有密钥和字符串（如图 4-15 所示）。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig15_HTML.jpg](img/499686_1_En_4_Fig15_HTML.jpg)

图 4-15

模板输出屏幕

将除 `SearchServiceQueryKey` 之外的值复制并粘贴到配置密钥设置中。参见图 4-16。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig16_HTML.jpg](img/499686_1_En_4_Fig16_HTML.jpg)

图 4-16

在应用配置文件中粘贴值

您需要从已部署的部分获取 `SearchServiceQueryKey` 值。对于部署，点击**概述**选项卡，如图 4-17 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig17_HTML.jpg](img/499686_1_En_4_Fig17_HTML.jpg)

图 4-17

获取 SearchServiceQueryKey 值

选择已部署的服务（如图 4-18 所示），然后点击右侧的**操作详细信息**。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig18_HTML.jpg](img/499686_1_En_4_Fig18_HTML.jpg)

图 4-18

选择已部署的服务

在详细信息屏幕中，点击左侧菜单中“设置”下的**密钥**（如图 4-19 所示）。复制 `SearchServiceQueryKey` 值，然后将其粘贴到应用程序配置文件中。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig19_HTML.jpg](img/499686_1_En_4_Fig19_HTML.jpg)

图 4-19

密钥选项

现在构建 API 解决方案，这将创建应用程序密钥（如图 4-20 所示）。在您接下来运行 `npm` 步骤后，您将回到此窗口。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig20_HTML.jpg](img/499686_1_En_4_Fig20_HTML.jpg)

图 4-20

在 API 解决方案中设置密钥

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig21_HTML.jpg](img/499686_1_En_4_Fig21_HTML.jpg)

图 4-21

将目录更改为下载的文件夹

1.  下一步是构建 Web 应用程序，该应用程序使用 Node.js 构建。打开 Node.js 命令提示符，并导航到前端文件夹。参见图 4-21。

运行 `npm install` 命令来构建应用程序（如图 4-22 所示）。它将获取所有依赖项。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig22_HTML.jpg](img/499686_1_En_4_Fig22_HTML.jpg)

图 4-22

运行命令 npm install

`npm install` 完成后，运行 `npm run build:prod` 命令来运行 Web 应用程序，该命令将构建并部署 Web 应用程序。参见图 4-23。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig23_HTML.jpg](img/499686_1_En_4_Fig23_HTML.jpg)

图 4-23

运行命令 npm run build:prod

接下来，返回到步骤 3（概述）中的上一个屏幕，然后按任意键完成构建（如图 4-24 所示）。然后导航到控制台中的 URL，以打开 Web 应用程序。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig24_HTML.jpg](img/499686_1_En_4_Fig24_HTML.jpg)

图 4-24

返回上一个屏幕以完成构建

构建和部署现已完成，可以访问关联的 JFK Files 应用程序，如图 4-25 所示。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig25_HTML.jpg](img/499686_1_En_4_Fig25_HTML.jpg)

图 4-25

在 JFK Files 应用中搜索

在这里，您可以搜索一个词条，例如“security”，并查看来自手写笔记图像以及键入的 PDF 文件的相关结果。它会在左侧窗格中显示关联的实体，在主窗口中显示包含该词条的文档（如图 4-26 所示）。

![../images/499686_1_En_4_Chapter/499686_1_En_4_Fig26_HTML.jpg](img/499686_1_En_4_Fig26_HTML.jpg)

图 4-26

JFK Files 应用搜索结果

至此，JFK Files 应用程序的安装、配置和运行就完成了。令人印象深刻的是，如何使用 Azure 认知服务、Azure 搜索、计算机视觉 API（使用文本识别 API 进行 OCR）、文本分析 API 和自定义技能来对与 JFK 遇刺相关的 34,000 页文件进行编目和解释。这些功能被用来构建一个复杂的认知搜索解决方案。它甚至可以关联和搜索 CIA 的密语，例如 GPFLoor 到 Oswald^(⁶)。

JFK Files 并非徒劳无功的假设性练习，而是一个针对企业数据的现实解决方案，这些数据大多是非结构化的，以多种格式存在，并来自各种类型的数据源。通过应用和修改 JFK 解决方案中规定的自然语言处理管道，您可以轻松地将其用于各种满足类似条件的企业解决方案。在下一个示例中，我们将探讨 Transformer 语言模型。



