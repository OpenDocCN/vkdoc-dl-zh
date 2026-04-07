Pramod Singh 和 Avinash Manure

# 学习 TensorFlow 2.0

## 使用 Python 实现机器学习和深度学习模型

![../images/489297_1_En_BookFrontmatter_Figa_HTML.png](img/489297_1_En_BookFrontmatter_Figa_HTML.png)

ISBN 978-1-4842-5560-5e-ISBN 978-1-4842-5558-2[`doi.org/10.1007/978-1-4842-5558-2`](https://doi.org/10.1007/978-1-4842-5558-2)© Pramod Singh, Avinash Manure 2020Apress StandardTrademarked names, logos, and images may appear in this book. Rather than use a trademark symbol with every occurrence of a trademarked name, logo, or image, we use the names, logos, and images only in an editorial fashion and to the benefit of the trademark owner, with no intention of infringement of the trademark. The use in this publication of trade names, trademarks, service marks, and similar terms, even if they are not identified as such, is not to be taken as an expression of opinion as to whether or not they are subject to proprietary rights.While the advice and information in this book are believed to be true and accurate at the date of publication, neither the authors nor the editors nor the publisher can accept any legal responsibility for any errors or omissions that may be made. The publisher makes no warranty, express or implied, with respect to the material contained herein.Distributed to the book trade worldwide by Springer Science+Business Media New York, 233 Spring Street, 6th Floor, New York, NY 10013\. Phone 1-800-SPRINGER, fax (201) 348-4505, e-mail orders-ny@springer-sbm.com, or visit www.springeronline.com. Apress Media, LLC is a California LLC, and the sole member (owner) is Springer Science+Business Media Finance Inc (SSBM Finance Inc). SSBM Finance Inc is a Delaware corporation.

*我将此书献给我的妻子，Neha，我的儿子，Ziaan，以及我的父母。没有你们，这本书*就不会*完成。你们完善了我的世界，是我力量的源泉*。

*——Pramod Singh*

*我将此书献给我的妻子，Jaya，因为她始终鼓励我在我所从事的任何事情上都做到最好，也献给我的母亲和父亲，感谢他们无条件的爱和支持，这使我成为今天的我。最后但同样重要的是，我要感谢 Pramod，因为他信任我，并给了我共同撰写这本书的机会。*

*——Avinash Manure*

引言

Google 在引入突破性的技术和产品方面一直处于先锋地位。TensorFlow 也不例外，在效率和规模方面，尽管还存在一些采用挑战，这促使 Google 的 TensorFlow 团队实施变更以简化使用。因此，写这本书的初衷仅仅是向读者介绍 TensorFlow 核心团队所做的这些重要变更。本书侧重于 TensorFlow 的不同方面，从机器学习的角度出发，并深入探讨最近方法变更的内部细节。本书是那些寻求迁移到 TensorFlow 以进行机器学习的人的好参考。

本书分为三个部分。第一部分介绍了使用 TensorFlow 2.0 进行数据处理。第二部分讨论了使用 TensorFlow 2.0 构建机器学习和深度学习模型，还包括使用 TensorFlow 2.0 进行神经语言编程（NLP）。第三部分涵盖了在生产环境中保存和部署 TensorFlow 2.0 模型。这本书对数据分析师和数据工程师也很有用，因为它涵盖了使用 TensorFlow 2.0 进行大数据处理的步骤。想要过渡到数据科学和机器学习领域的读者也会发现，这本书提供了一个实用的入门介绍，可以引导他们进入更复杂的领域。书中提供的案例研究和示例使相关基本概念易于理解和跟随。此外，关于 TensorFlow 2.0 的书籍非常少，这本书无疑会提高读者的知识水平。本书的强大之处在于其简洁性以及将机器学习应用于有意义的数据集。

我们尽力将我们所有的经验和知识融入到这本书中，并觉得它特别适合企业寻求解决实际挑战的需求。我们希望你能从中获得一些有用的收获。

致谢

这是我和 Apress 合作的第三本书，写作过程中投入了大量的思考。主要目标是向 IT 社区介绍 TensorFlow 新版本中引入的关键变化。我希望读者会认为它有用，但首先，我想感谢一些在旅途中帮助过我的人。首先，我必须感谢我生命中最重要的那个人，我深爱的妻子，Neha，她无私地支持我，牺牲了很多以确保我完成这本书。

我必须也要感谢我的合著者，Avinash Manure，他付出了巨大的努力以确保项目按时完成。此外，我要感谢 Celestin Suresh John，他信任我并给了我这次为 Apress 写另一本书的机会。Aditee Mirashi 是印度最好的编辑之一。这是我和她合作的第三本书，再次合作非常令人兴奋。她一如既往地非常支持我，总是随时准备满足我的要求。我要感谢 James Markham，他耐心地审查每一行代码并检查每个示例的适当性，感谢你的反馈和鼓励。这对我以及这本书都产生了很大的影响。我还想感谢那些不断鼓励我追逐梦想的导师们。感谢 Sebastian Keupers，Vijay Agneeswaran 博士，Sreenivas Venkatraman，Shoaib Ahmed 和 Abhishek Kumar。

最后，我对我的儿子 Ziaan 和我的父母无限感激，他们无论在什么情况下都给予了我无尽的爱和支持。你们让我的世界变得如此美好。

> ——Pramod Singh

这是我写的第一本书，确实非常特别。正如普拉莫德所说，这本书的目的是向读者介绍 TensorFlow 2.0，并解释这个平台是如何在过去的几年里发展演变，成为目前最流行和用户友好的机器学习源代码库之一。我要感谢普拉莫德对我的信任，并给予我共同撰写这本书的宝贵机会。由于这是我写的第一本书，普拉莫德一直在指导和帮助我完成它。

我要感谢我的妻子，贾亚，她确保我在家中拥有合适的环境，以便集中精力并及时完成这本书。我还要感谢出版团队——阿迪提·米拉希、马修·穆迪和詹姆斯·马克汉姆，他们在我确保这本书以最佳状态达到读者手中方面给予了我巨大的帮助。我还要感谢我的导师们，他们通过始终支持我的梦想并引导我朝着梦想前进，确保我在专业和个人方面不断成长。感谢特里斯坦·比什普、埃林·阿姆德森、迪帕克·贾因、Vijay Agneeswaran 博士和 Abhishek Kumar，感谢你们给予我的所有支持。最后但同样重要的是，我要感谢我的父母、朋友和同事，在我困难的时候一直在我身边，并激励我追求我的梦想。

> ——阿维纳什·马努尔

### 目录

第一章：TensorFlow 2.0 简介 1 Tensor + Flow = TensorFlow 2 组件和基向量 3 张量 6 秩 7 形状 7 流 7 TensorFlow 1.0 与 TensorFlow 2.0 9 与可用性相关的更改 10 与性能相关的更改 16 TensorFlow 2.0 的安装和基本操作 17 Anaconda 17 Colab 17 Databricks 19 结论 24 第二章：使用 TensorFlow 进行监督学习 25 什么是监督机器学习？ 25 使用 TensorFlow 2.0 进行线性回归 28 使用 TensorFlow 和 Keras 实现线性回归模型 29 使用 TensorFlow 2.0 进行逻辑回归 37 使用 TensorFlow 2.0 的增强树 47 集成技术 47 梯度提升 49 结论 52 第三章：使用 TensorFlow 进行神经网络和深度学习 53 什么是神经网络？ 53 神经元 54 人工神经网络 (ANNs) 55 简单的神经网络架构 57 前向和反向传播 58 使用 TensorFlow 2.0 构建神经网络 61 关于数据集 61 深度神经网络 (DNNs) 67 使用 TensorFlow 2.0 构建 DNNs 68 使用 Keras 模型构建估计器 71 结论 74 第四章：使用 TensorFlow 处理图像 75 图像处理 76 卷积神经网络 77 卷积层 77 池化层 80 全连接层 81 使用 TensorFlow 2.0 构建卷积神经网络 82 高级卷积神经网络架构 89 迁移学习 93 迁移学习与机器学习 95 使用 TensorFlow 2.0 的变分自动编码器 97 自动编码器 97 自动编码器的应用 98 变分自动编码器 98 使用 TensorFlow 2.0 实现变分自动编码器 99 结论 106 第五章：使用 TensorFlow 2.0 进行自然语言处理 107 NLP 概述 107 文本预处理 109 分词 110 词嵌入 112 使用 TensorFlow 进行文本分类 113 文本处理 115 深度学习模型 119 嵌入 120 TensorFlow Projector 123 结论 129 第六章：TensorFlow 模型在生产中的应用 131 模型部署 132 隔离 133 协作 133 模型更新 134 模型性能 134 负载均衡器 134 基于 Python 的模型部署 135 保存和恢复机器学习模型 135 部署机器学习模型为 REST 服务 138 模板 142 使用 Flask 的挑战 145 构建基于 Keras TensorFlow 的模型 146 TF ind 部署 151 结论 159 索引 161

### 关于作者和关于技术审稿人

### 关于作者

### 关于技术审稿人
