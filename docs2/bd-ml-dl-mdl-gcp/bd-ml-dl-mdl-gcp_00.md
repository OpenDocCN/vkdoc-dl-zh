Ekaba Bisong

# 在 Google Cloud Platform 上构建机器学习和深度学习模型

## 《初学者全面指南》

![img/463852_1_En_BookFrontmatter_Figa_HTML.png](img/463852_1_En_BookFrontmatter_Figa_HTML.png)

ISBN 978-1-4842-4469-2e-ISBN 978-1-4842-4470-8[`doi.org/10.1007/978-1-4842-4470-8`](https://doi.org/10.1007/978-1-4842-4470-8)© Ekaba Bisong 2019 本作品受版权保护。无论整个作品还是部分内容，所有权利均由出版社保留，具体包括翻译权、重印权、插图的使用权、朗诵权、广播权、在胶片或任何其他物理方式上的复制权，以及传输或信息存储和检索、电子改编、计算机软件或通过现在已知或将来开发的方法相似或不同的方法。商标名称、标志和图像可能出现在本书中。我们不会在每个商标名称、标志或图像出现时使用商标符号，我们仅以编辑方式使用名称、标志和图像，以商标所有者的利益为重，无意侵犯商标权。在本出版物中使用贸易名称、商标、服务标志和类似术语，即使它们没有被标识为这样的，也不应被视为对它们是否受专有权利约束的意见表达。虽然本书中的建议和信息在出版日期被认为是真实和准确的，但作者、编辑或出版社不能对可能出现的任何错误或遗漏承担任何法律责任。出版社对本书中包含的材料不提供任何明示或暗示的保证。本书由 Springer Science+Business Media New York 通过全球图书贸易发行，地址：纽约市 Spring Street 233 号，第 6 层，纽约，NY 10013。电话：1-800-SPRINGER，传真：(201) 348-4505，电子邮件：orders-ny@springer-sbm.com，或访问 www.springeronline.com。Apress Media, LLC 是一家加利福尼亚有限责任公司，唯一成员（所有者）是 Springer Science + Business Media Finance Inc（SSBM Finance Inc）。SSBM Finance Inc 是一家特拉华州公司。

*本书献给创造天地和所有智慧的至高无上的神圣三位一体上帝。献给我的父母，弗朗西斯教授和（夫人）诺索·比松，我的导师约翰·奥蒙恩教授和已故的皮厄斯·阿代桑米教授，以及我的最好朋友和伴侣拉辛。*

引言

机器学习和深度学习技术以深刻的方式影响了世界，从我们与技术产品以及彼此互动的方式。这些技术正在改变我们相互关系、工作方式以及我们一般的生活方式。今天，以及可预见的未来，智能机器越来越多地成为社会文化和社会经济关系的基础。我们确实已经进入了“智能时代”。

## 什么是机器学习和深度学习？

机器学习可以被描述为一系列工具和技术，用于根据特定数据集中变量（也称为特征或属性）之间的一系列交互来预测或分类未来事件。另一方面，深度学习扩展了名为神经网络的机器学习算法，用于学习对计算机来说极其困难的复杂任务。这些任务的例子可能包括识别面部和在不同语境中理解语言。

## 大数据的作用

机器学习和深度学习崛起和未来性能提升的关键因素是数据。自 21 世纪初以来，生成和存储的数据量一直在稳步指数级增长。巨量数据的兴起部分是由于互联网的出现和处理器小型化的推动，这催生了“物联网（IoT）”技术。这些庞大的数据量使得训练计算机学习复杂任务成为可能，在这些任务中，显式指令集是不切实际的。

## 计算挑战

可用于训练学习模型的可用数据的增加提出了另一种类型的问题，那就是计算或处理能力的可用性。从经验上看，随着数据的增加，学习模型的性能也会提高。然而，由于今天数据集的规模越来越大，在普通机器上训练复杂、最先进的学习模型是不切实际的。

## 云计算拯救了世界

云是一个术语，用于描述由称为数据中心的一组计算机组成的庞大集合。这些数据中心通常分布在多个地理位置。像谷歌、微软、亚马逊和 IBM 这样的大公司拥有庞大的数据中心，在那里他们管理着提供给公众（即企业和个人用户）使用的计算基础设施，成本非常合理。

云技术/基础设施正在使个人能够利用大企业的计算资源进行机器学习/深度学习实验、设计和开发。例如，通过利用谷歌云平台（GCP）、亚马逊网络服务（AWS）或微软 Azure 等云资源，我们可以以本地机器所需时间的几分之一来运行一系列算法和多个测试网格。

## 进入谷歌云平台（GCP）

云计算领域的主要竞争对手之一是谷歌，他们提供的云资源服务被称为“谷歌云平台”，通常简称为 GCP。谷歌也是互联网空间中顶尖的技术领导者之一，拥有包括 Gmail、YouTube 和谷歌地图在内的多种领先网络产品。这些产品每天从全球互联网用户那里生成、存储和处理数以吨计的太字节数据。

为了处理这些大量数据，多年来谷歌在处理和存储基础设施上投入了大量资金。截至目前，谷歌拥有世界上一些最令人印象深刻的数据中心设计和技术，以支持其计算需求和计算服务。通过 Google Cloud Platform，公众可以利用这些强大的计算资源来设计和开发前沿的机器学习和深度学习模型。

## 本书的目标

本书的目标是从零开始，为读者提供构建学习模型所需的基本原理和工具。机器学习和深度学习正在快速发展，对于初学者来说，进入这个领域往往感到令人不知所措和困惑。许多人不知道从何开始。本书是一个一站式商店，带领初学者了解机器学习和深度学习技术的理论基础和实践步骤，以解决感兴趣的问题。

## 书籍组织

本书分为八个部分。它们的结构如下：

+   第一部分：Google Cloud Platform 入门

+   第二部分：数据科学编程基础

+   第三部分：介绍机器学习

+   第四部分：实践中的机器学习

+   第五部分：介绍深度学习

+   第六部分：实践中的深度学习

+   第七部分：高级分析/在 Google Cloud Platform 上的机器学习

+   第八部分：在 GCP 上实现机器学习解决方案

按顺序通读整本书是最好的。然而，每个部分及其包含的章节都是这样编写的，可以挑选感兴趣的章节进行阅读。本书的代码库可在[`https://github.com/Apress/building-ml-and-dl-models-on-gcp`](https://github.com/Apress/building-ml-and-dl-models-on-gcp)找到。读者可以通过克隆存储库到 Google Colab 或 GCP 深度学习虚拟机来跟随本书中的示例。

致谢

我想利用这个机会感谢卡尔顿大学计算机科学学院的员工和教师；他们在本书的撰写过程中营造了一个友好而富有挑战性的氛围。我想感谢我在研究生项目中的朋友们，Abdolreza Shirvani、Omar Ghaleb、Anselm Ogbunugafor、Sania Hamid、Gurpreet Saran、Sean Benjamin、Steven Porretta、Kenniy Olorunnimbe、Moitry Das、Yajing Deng、Tansin Jahan 和 Tahira Ghani。我还想感谢我在担任助教期间在本科水平上的亲密朋友 Saranya Ravi 和 Geetika Sharma，以及我在智能系统实验室的朋友和同事 Vojislav Radonjic。他们都非常支持，以友善和慷慨的友谊帮助我克服了当时的明显压力。我特别想感谢 Sania Hamid，她帮助我输入了手稿的部分内容。尽管如此，我对此书中包含的任何印刷错误承担全部责任。

我要感谢我的朋友们 Rasine Ukene、Yewande Marquis、Iyanu Obidele、Bukunmi Oyedeji、Deborah Braide、Akinola Odunlade、Damilola Adesinha、Chinenye Nwaneri、Chiamaka Chukwuemeka、Deji Marcus、Okoh Hogan、Somto Akaraiwe、Kingsley Munu 和 Ernest Onuiri，他们在旅途中给予了我鼓励的话语。在我本科学习期间，Ernest 先生在巴巴克大学开设的人工智能课程开启了我在这个领域的旅程。Somto 提供了宝贵的反馈，指导我改进了本书章节和部分的结构。同样，我要感谢卡尔顿大学非洲研究学院的同事们，Femi Ajidahun 和 June Creighton Payne，特别感谢一位导师和资深朋友，他把我当作儿子，非常善良、支持和慷慨，（已故）皮乌斯·阿德桑米教授，当时是该学院院长。阿德桑米教授不幸在 2019 年 3 月 10 日从亚的斯亚贝巴起飞后不久，在埃塞俄比亚航空 302 号（ET 302）航班失事中丧生。我要感谢 Muyiwa Adesanmi 夫人和 Tise Adesanmi，感谢他们的爱、友谊和力量。愿他们在现在和未来都能找到安慰。

我要感谢 Emmanuel Okoi、Jesam Ujong、Adie Patrick、Redarokim Ikonga 以及希望沃德尔老学生协会（HWOSA）的家庭；他们在压力时期提供了社区和乐趣，缓解了情绪。特别感谢当时我的室友，Jonathan Austin、Christina Austin、Margherita Ciccozzi、Thai Chin 和 Chris Teal；我有一个极好的地方可以称之为家。

我特别感谢我在 Pythian 的前同事和朋友们，Vanessa Simmons、Alex Gorbachev 和 Paul Spiegelhalter，他们在旅途中给予了我帮助和支持。我要感谢在渥太华、多伦多、温尼伯、卡拉斯和奥韦里的 House Fellowships 的兄弟姐妹们；他们构成了我的公司，是终生的朋友。我还要感谢标准世界广播网络（SWBN）的员工和机组人员，他们共同努力，在我参与这个项目时保持愿景的运行。特别感谢 Susan McDermott、Rita Fernando 以及 Apress 的出版和编辑团队对这一项目的支持和信念。感谢 Vikram Tiwari 和 Gonzalo Gasca Meza，他们为这份手稿提供了技术审查。

最后，我通过特别感谢我的家人来结束，从我的爱父爱母 Francis Ebuta 和 Nonso Ngozika Bisong 开始；他们是我生命中的支柱，一直在我身边提供咨询和鼓励，我真正感激我的父母。感谢我的兄弟姐妹 Osowo-Ayim、Chidera 和 Ginika Bisong，以及 Bisong 家族的扩展成员，他们不断给予爱和支援。最后，向我的阿姨和好友 Joy Duncan 致以崇高的敬意，感谢她的爱和友谊；她是我心中最亲爱的人。还要感谢 Wilfred Achu 叔叔和 Blessing Bisong 阿姨，他们对我非常善良，我非常感激。

### 目录

第一部分：Google Cloud Platform 入门 1 第一章：什么是云计算？ 3 云解决方案类别 4 云计算模型 5 第二章：Google Cloud Platform 服务概述 7 云计算 7 云存储 8 大数据与分析 9 云人工智能（AI） 10 第三章：Google Cloud SDK 和 Web CLI 11 在 Google Cloud Platform 上设置账户 12 GCP 资源：项目 14 访问云平台服务 16 账户用户和权限 16 云 Shell 17 Google Cloud SDK 19 第四章：Google Cloud Storage (GCS) 25 创建存储桶 25 将数据上传到存储桶 27 从存储桶中删除对象 30 释放存储资源 30 从命令行操作 GCS 32 第五章：Google Compute Engine (GCE) 35 配置虚拟机实例 35 连接到虚拟机实例 41 拆除实例 44 从命令行操作 GCE 45 第六章：JupyterLab 笔记本 49 配置笔记本实例 49 关闭/删除笔记本实例 53 从命令行启动笔记本实例 54 第七章：Google Colaboratory 59 Colab 入门 59 更改运行时设置 61 存储笔记本 62 上传笔记本 64

### 关于作者和关于技术审稿人

### 关于作者

### 关于技术审稿人
