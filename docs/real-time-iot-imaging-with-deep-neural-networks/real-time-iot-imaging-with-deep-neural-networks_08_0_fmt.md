# 3. 树莓派 4 上的视觉处理

> *想象一个世界，你唯一的财产是一颗覆盆子，而你把它送给了朋友。*
>
> ——格尔达·韦斯曼·克莱因，《除了我的生命》（Hill and Wang，1995）

在第 1 章和第 2 章中，你了解了如何在普通台式计算机上使用 Java 启动并运行计算机视觉。设置过程有些手动，但所有不同的构建模块都是免费提供的，因此只需将这些模块组合在一起即可。

在本章中，你将向组合中添加另一个模块——树莓派 4。本章的首要目标是执行与在计算机上相同的任务，但尽可能在物联网设备上运行，同时从本地桌面输入代码并控制代码执行。换句话说，你将在计算机上编写代码，但在树莓派上执行。

对于代码执行部分，你需要一些特定的硬件和线缆，因此你应该准备好去购物。我们将提供一份购物清单，然后继续设置。

具体来说，我们将介绍如何创建 SD 卡和设置树莓派，但假设这些步骤中的大部分已在其他书籍和博客中充分介绍过，因此如果你需要比此处介绍的基础知识更多的内容，可以阅读那些资料。

在本章结束时，你将很好地掌握树莓派在运行不同视觉操作和检测算法时的速度及能耗。你还将准备好将此功能大规模集成到你设计的语音控制家庭助手中，从而扩展你的个人家庭或工作办公室自动化。

## 让树莓派“活”起来

*“天哪，”我喊道，“你竟在智慧的骄傲中如此无知！”*
——玛丽·雪莱，《弗兰肯斯坦》（1818 年）

我们不是弗兰肯斯坦博士，但要让树莓派“活”起来，我们必须像设计一个生命体一样工作：插上电缆，组装，然后给这个小设备通电。需要采购一些物品，在开始做任何有用的事情之前，我们必须先收集好物理硬件。幸运的是，大部分要买的东西都很便宜，而且可以从别人的旧电脑或你奶奶的阁楼里找到可重复利用的。她不会发现的。

采购完毕，树莓派和所有必要部件都组装好后，我们将继续在 Visual Studio Code 上设置与第 1 章相同的视觉项目，并再次学习如何用 Java 编写 OpenCV 程序。

### 采购清单

虽然树莓派本身是一台功能强大的计算机，但大多数时候，如果没有一些额外的配件，你什么也做不了。在本章中，你需要一些附件。

表 3-1 提供了一份采购清单，但你可以自行决定购买的类型和地点。

**小心**

如今市面上的电缆和电源适配器种类繁多。如果你想进行超频，请务必购买输出电流在 4.8A 到 5A 左右的电源适配器。你选择的 USB 线缆也应能承载相应的电流，不过现在大多数线缆都能做到。

**表 3-1** 采购清单

| 名称 | 用途 | 预估价格（美元） | 必需？ | 始终需要？ |   |
| --- | --- | --- | --- | --- | --- |
| 树莓派 4 4GB | 主计算机 | $55 | 是 | 是 | □ |
| USB-C 转 USB-A 线缆 | 连接计算机与电源 | $6 | 是 | 是 | □ |
| 带 USB-A 接口的电源适配器（至少 4.8A，超频需 5A） | 电力供应 | $10 | 是 | 是 | □ |
| HDMI 线缆 | 连接屏幕 | $7 | 是 | 否 | □ |
| SD 卡 128GB | 硬盘 | $10 | 是 | 是 | □ |
| ReSpeaker 麦克风 | 推荐的语音助手麦克风（第 3 章） | $25 | 否 | 如果购买，则需要 | □ |
| 带 HDMI 接口的显示器 | 初始调试用显示器 | $25 | 是 | 否 | □ |
| USB 摄像头 | 录制视频 | $20 | 是 | 是 | □ |
| 键盘和鼠标 |   | $20 | 可能 | 否 | □ |
| 总计 |   | $180 |   |   |   |

一旦你收集齐所有部件（图 3-1），就该让软件在树莓派上运行了。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig1_HTML.jpg](img/490964_1_En_3_Fig1_HTML.jpg)

**图 3-1** 树莓派 4、电源适配器、摄像头、屏幕等

### 下载操作系统

没有操作系统，树莓派甚至不会看一眼像马塞尔这样漂亮的猫。树莓派官方网站上提供了几个现成的操作系统。

以下是主要操作系统的简短列表：

*   Ubuntu Mate

*   Ubuntu Core

*   Windows 10 IoT Core

*   Noobs

*   Raspbian（官方操作系统）

Ubuntu Mate 和 Ubuntu Core 是新的不错选择，但我们将专注于官方操作系统 Raspbian，因为大多数树莓派的文档都基于它。而且它上手也非常简单。

Raspbian 是一个基于 Debian 的操作系统，因此大多数 Linux 爱好者会对其感到熟悉，包括其内置的 `aptitude` 包管理器。尽管软件包必须专门为树莓派的 ARM 架构 CPU 编译，但软件包列表非常新，并且大部分与主线 Debian 保持同步。

为了能够将操作系统用作树莓派可启动的镜像，我们首先需要将镜像文件下载到本地计算机，并创建一个专门用于树莓派的可启动磁盘。

要下载这个非常大的操作系统文件（不要在用手机热点时尝试），请访问 [`https://www.raspberrypi.org/downloads/`](https://www.raspberrypi.org/downloads/) 并找到图 3-2 中所示的 Raspbian 图标。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig2_HTML.jpg](img/490964_1_En_3_Fig2_HTML.jpg)

**图 3-2** Raspbian 图标

点击该图标。你将进入 Raspbian 专属的下载页面，如图 3-3 所示，你可以在其中找到要下载的 zip 文件链接。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig3_HTML.jpg](img/490964_1_En_3_Fig3_HTML.jpg)

**图 3-3** Raspbian 的种子或 zip 文件

我们不需要完整的桌面安装和所有推荐的软件，但为了简化操作，“带桌面的 Raspbian Buster”版本目前对我们来说就足够了。

### 创建可启动 SD 卡

将操作系统文件下载到计算机后，你就可以创建可启动设备了。正如树莓派网站上所述，将操作系统安装到树莓派上最简单的方法是使用 Etcher，这是一款可以从 zip 文件或 `.img` 文件创建可启动 SD 卡的软件。

Etcher 可从 [`https://www.balena.io/etcher/`](https://www.balena.io/etcher/) 获取，它可以在任何平台上运行，例如 macOS、Windows 和 Linux。本书中的截图是在 Windows 上截取的。

启动应用程序后的第一步是选择下载的 zip 文件，如图 3-4 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig4_HTML.jpg](img/490964_1_En_3_Fig4_HTML.jpg)

**图 3-4** 选择镜像文件

选择好镜像文件并将 SD 卡插入计算机的 SD 卡槽后，让我们选择要写入操作系统的 SD 卡，如图 3-5 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig5_HTML.jpg](img/490964_1_En_3_Fig5_HTML.jpg)

**图 3-5** 选择 micro SD 卡

这里特意选择了一张容量很大的卡，所以你可以忽略关于其大小的提示信息，如图 3-6 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig6_HTML.jpg](img/490964_1_En_3_Fig6_HTML.jpg)

**图 3-6** SD 卡容量很大

最后，让我们继续“烧录”这张卡，换句话说，就是将文件写入卡中。进度会显示出来，通常不会超过两三分钟，如图 3-7 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig7_HTML.jpg](img/490964_1_En_3_Fig7_HTML.jpg)

**图 3-7** 烧录中

在最后一步，Etcher 会检查已复制数据的完整性，如图 3-8 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig8_HTML.jpg](img/490964_1_En_3_Fig8_HTML.jpg)

**图 3-8** 验证中

一旦烧录和验证完成，Windows 会像以前一样尝试使用该驱动器，但现在 SD 卡的格式在没有特定驱动程序的情况下无法读取，如图 3-9 所示。你可以安全地忽略此消息并单击“取消”。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig9_HTML.jpg](img/490964_1_En_3_Fig9_HTML.jpg)

**图 3-9** 别害怕……

卡已经准备好了，现在让我们插上树莓派的线缆吧。

#### 连接线缆

> *电是危险的。我的侄子曾试图把一枚硬币塞进插座。说“一分钱走不远”的人，真该看看他当时被弹飞出去的样子。我告诉他，他被“接地”了。*
>
> ——蒂姆·艾伦

我发现把各种线缆插接在一起既有趣又放松，实际上，这甚至可以用于个人心理治疗，来应对我们所处的这个艰难世界。

表 3-2 列出了需要进行的连接（并非详尽清单）。

**表 3-2** 插接清单

| 从 | 到 | 原因 |   |
| --- | --- | --- | --- |
| 电源交流电 | 电源适配器 | 获取电力 | □ |
| USB-A | 电源适配器 | 为树莓派供电 | □ |
| 网络摄像头 | 树莓派 USB 端口 | 获取视频流 | □ |
| 鼠标 | 树莓派 USB 端口 | 使用指针 | □ |
| 键盘 | 树莓派 USB 端口 | 使用键盘 | □ |
| 屏幕 HDMI | 树莓派 Micro HDMI | 显示画面 | □ |
| USB-C | 树莓派 | 为树莓派供电 | □ |

**注意**

请务必最后再插入电源适配器或 USB-C 转树莓派适配器；否则，通电后主板将启动，但无事可做。

有图通常更容易理解，因此请务必参考图 3-10，了解各部件在实际主板上的连接位置。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig10_HTML.jpg](img/490964_1_En_3_Fig10_HTML.jpg)

**图 3-10** 树莓派 4 的原始蓝图及连接示意图

最后，将 micro SD 卡插入树莓派 4 后，你就可以准备通电启动了（或者你已经启动了？）。

#### 首次启动

首次启动非常快。启动过程包括一个需要调整文件系统大小的步骤，现在 Raspbian 系统会在首次启动时自动处理。你应该会看到一个欢迎屏幕，如图 3-11 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig11_HTML.jpg](img/490964_1_En_3_Fig11_HTML.jpg)

**图 3-11** 首次启动

连接好键盘、鼠标和屏幕后，你可以做很多事情，但对于本章要完成的大多数任务，我们需要通过 SSH 进行操作，这需要网络连接。

你可以将以太网线插入树莓派的以太网端口，但使用 WiFi 效果也很好。在右上角，点击 WiFi 图标，并在对话框中输入你的 WiFi 设置，如图 3-12 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig12_HTML.jpg](img/490964_1_En_3_Fig12_HTML.jpg)

**图 3-12** 设置 WiFi

在通过 SSH 连接之前，还需要最后一步：启用树莓派内置的 OpenSSH 服务器。可以通过以下菜单启用：

```
首选项 ➤ 树莓派配置 ➤ 接口
```

点击启用 SSH 服务器，如图 3-13 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig13_HTML.jpg](img/490964_1_En_3_Fig13_HTML.jpg)

**图 3-13** 启用已安装的 SSH 服务器

现在，让我们通过 SSH 连接到树莓派。

#### 使用 `nmap` 查找你的树莓派

你可能知道，要远程连接到树莓派，你需要它的 IP 地址。你可以在树莓派上打开终端并输入以下命令来找到它：

```
ip addr
```

这将显示你需要用来连接树莓派的 IP 地址。

如果你运气不佳，没有键盘、屏幕、鼠标，或者这些都没有，你可以采用一些简单的命令行应对措施，使用 `nmap` 来查找网络上的所有树莓派（以及所有电脑和国产智能手机）。

这可以通过在主计算机上运行类似以下的命令来完成：

```
sudo nmap -O -p 22 192.168.1.*
```

这里，`192.168.1.*` 是你自己桌面 IP 地址的开头部分，也是树莓派可能所在的网段。请注意，要使此命令生效，我们假设你的计算机和树莓派在同一个网络上。

这个命令实际上是做什么的？简而言之，`nmap` 会扫描本地网络，查找可用的机器和开放的端口。

`-p` 参数用于扫描指定端口，`-O` 参数则表示尽可能多地收集每个找到的主机的信息。

在本地网络上运行上述 `nmap` 命令，会得到类似图 3-14 的结果。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig14_HTML.jpg](img/490964_1_En_3_Fig14_HTML.jpg)

**图 3-14** 查找你的树莓派

已在网络上找到一台树莓派（希望是你的！）。现在，你可以使用默认用户名/密码（应该仍然是 `pi / raspberry`）通过 SSH 连接到它，命令如下：

```
ssh pi@192.16.1.17
```

然后即可连接成功，如图 3-15 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig15_HTML.jpg](img/490964_1_En_3_Fig15_HTML.jpg)

**图 3-15** 启动 SSH 连接

我们建立了一次连接，但后续需要反复执行这些连接步骤，因此，让我们通过修改本地和远程树莓派上的配置文件，来保存一些 SSH 设置以备后用。

#### 轻松设置 SSH

通常，在通过 SSH 连接时，你会禁用密码认证，尤其是树莓派的默认认证方式，然后从本地机器设置一个密钥来连接远程机器，再将该远程机器配置为只接受一个或有限数量的密钥，以防止随机的安全攻击。

其工作原理如下：

1.  创建你个人使用的 SSH 密钥对。一个私钥保留在你的电脑上，一个公钥作为已知实体添加到远程机器上。在量子计算到来之前，这种密钥对被认为是安全的，并且仅通过公钥几乎不可能猜出某人的私钥。有关更多信息，请阅读 Joshua Davies 在 [`https://commandlinefanatic.com/cgi-bin/showarticle.cgi?article=art054`](https://commandlinefanatic.com/cgi-bin/showarticle.cgi%253Farticle%253Dart054) 上发表的文章。

2.  将生成的公钥注册到远程设备（此处为树莓派）上。你可以随意将公钥提供给任意多的人；默认情况下，这是以 `.pub` 扩展名结尾的文件。

3.  添加一些 SSH 配置，以便在通过快捷方式连接到树莓派时使用生成的私钥。不过，那个私钥是属于你的，请确保不要将其提供给任何人！

在亚马逊云服务（AWS）上，你通常会得到一个 `.pem` 文件来连接新的运行实例。该 `.pem` 文件会自动为你注册到新的虚拟机上，因此只有你能访问这个新创建的虚拟机，从而确保对你云端机器的安全访问。

要在你自己的本地机器上创建密钥，请使用 `ssh-keygen` 命令，配合 `-t` 参数指定算法，以及 `-b` 参数指定位数（目前，4096 位被认为是相当安全的）。

```
% ssh-keygen -t rsa -b 4096
Generating public/private rsa key pair.
Enter file in which to save the key (~/.ssh/id_rsa):
```

命令成功运行后，会生成两个文件：`id_rsa.pub` 文件和 `id_rsa` 文件。`pub` 文件是你的公钥，可以提供给任何人；`id_rsa` 是你的私钥，必须极其小心地处理。

为了让树莓派知道你的公钥并识别你，你需要将其作为一行内容添加到 `/home/pi/.ssh/authorized_keys` 文件中，如果必要的话，还需创建该文件及其父文件夹 `.ssh`。

完成后，在你的电脑上，就该将 SSH 连接的详细信息添加到 `$HOME/.ssh/config` 文件中，同样，如果该文件不存在，则创建它。

```
Host pi4
User pi
IdentityFile ~/.ssh/id_rsa
HostName 192.168.1.17
ForwardX11 yes
ForwardX11Trusted yes
```

以下是这些连接设置的含义：

- `pi4` 是我们从现在开始用来连接树莓派的快捷方式。

- `User` 是树莓派上的默认用户，通常是 `pi`。

- `IdentifyFile` 是你电脑上用于验证你身份的私钥路径；这是你刚才创建的文件。

- `HostName` 是树莓派的 IP 地址。如果你忘记了，可以再次使用 `nmap` 查找。

- `ForwardX11` 和 `ForwardX11Trusted` 是必需的，因为我们稍后会将视频流重定向到主电脑。

有了所有这些设置，你可能会想这一切是否值得。让我们立即获得一点成就感来证明它是值得的。通过 SSH 连接，我们现在将打开一个带有用户界面的应用程序，例如内置的 Chromium 浏览器。

你可以通过更改 `DISPLAY` 变量的设置来切换树莓派屏幕上的窗口显示。

- 要显示树莓派屏幕，请使用 `export DISPLAY=:0`。
- 要显示本地电脑屏幕，请使用 `export DISPLAY=:10`。

以下是命令列表：

```
ssh pi4
export DISPLAY=:10
chromium-browser
```

图 3-16 展示了 Chromium 如何在树莓派上执行，但其窗口却打开在电脑上。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig16_HTML.jpg](img/490964_1_En_3_Fig16_HTML.jpg)

图 3-16 打开远程应用程序

### 为远程使用设置 Visual Studio Code

本书的主要概念之一是，你将能够在一台大屏幕的机器上编写代码，然后让其在远程设备上运行，并将视觉输出重定向到大屏幕上。

到目前为止，你一直在 Visual Studio Code 中编写代码并在本地运行。从现在开始，你将在本地编写代码，并在远程运行它。

本章前面所有的 SSH 设置工作都是因为 Visual Studio Code 有一个名为 Remote-SSH 的插件，一旦你完成了 SSH 设置，它就会为你完成所有其余工作，并与编辑器的生态系统完全集成。

让我们首先安装 Remote-SSH 插件，如图 3-17 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig17_HTML.jpg](img/490964_1_En_3_Fig17_HTML.jpg)

图 3-17 安装 Remote-SSH 插件

插件安装完毕并快速刷新编辑器后，你可以从命令启动器中启动“连接到主机”命令，如图 3-18 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig18_HTML.jpg](img/490964_1_En_3_Fig18_HTML.jpg)

图 3-18 连接到主机

注意：如果当前项目文件过多，Remote-SSH 插件可能无法正常工作。如果出现这种情况，请确保在打开新的 Remote-SSH 会话之前，先在 Visual Studio Code 中打开一个空白的全新窗口。

图 3-19、图 3-20 和图 3-21 向你展示了连接到树莓派所需的步骤。请注意，`pi4` 这个快捷方式取自你刚才在 `$HOME/.ssh/config` 文件中设置的那个。如果你使用了其他快捷方式，这里会有所不同。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig21_HTML.jpg](img/490964_1_En_3_Fig21_HTML.jpg)

图 3-21 从 Visual Studio Code 连接到树莓派

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig20_HTML.jpg](img/490964_1_En_3_Fig20_HTML.jpg)

图 3-20 设置 SSH 主机

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig19_HTML.jpg](img/490964_1_En_3_Fig19_HTML.jpg)

图 3-19 选择 pi4

此时，在 Visual Studio Code 内部，你拥有一个远程连接到树莓派的终端。这意味着你可以随意启动远程命令，而且你还可以拔掉树莓派上的键盘和鼠标；其余大部分工作都将在你的本地电脑上完成。

作为第一次检查，我们可以运行一个命令来检查树莓派的 IP 地址，如下所示：

```
ip addr
```

输出将显示在 Visual Studio Code 的“终端”选项卡中，如图 3-22 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig22_HTML.jpg](img/490964_1_En_3_Fig22_HTML.jpg)

图 3-22 在“终端”选项卡上确认 IP 地址

输出 IP 地址显然不是我们能用那个终端做的主要事情，所以让我们继续，现在安装所需的 Java 开发工具包。

#### 安装 Java OpenJDK

本书中的所有代码均为 Java 代码，因此我们需要在树莓派上安装 Java 开发工具包和运行时环境。

作为可选依赖项，你还可以安装 Maven。本书中的大部分示例并不真正需要 Maven，但这可以阻止 Visual Studio Code 显示烦人的消息，提示它找不到（非必需的）Maven 可执行文件。

因此，在 Visual Studio Code 的终端标签页中，让我们使用软件包安装程序 `apt` 来为我们安装 Java。这通过下面显示的小代码片段完成：

```
sudo apt update
sudo apt install openjdk-11-jdk
```

可选地，也可以运行以下命令来安装 Maven：

```
sudo apt install maven
```

每个命令行开头的 `sudo` 用于获取所需的提升权限，以便使用 `apt` 安装新软件。

前面命令的输出应类似于图 3-23。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig23_HTML.jpg](img/490964_1_En_3_Fig23_HTML.jpg)

图 3-23 安装 Java

#### 安装 Java SDK 的替代方案

当前在运行 Buster（最新的 Raspbian 版本）的树莓派 4 上可选的 JDK 相当丰富。然而，有时，尤其是在较旧版本的 Raspbian 上，你可能会长时间困于过时的 JDK。

值得注意的是，像 Azul 和 Bellsoft 这样的公司已经发布了专门为 ARM CPU（树莓派上运行的处理器）编译的最新版本 JDK。

访问这两个网站以获取 JDK 软件包的直接链接：

```
https://bell-sw.com/pages/java-11.0.5%20for%20Embedded/
https://www.azul.com/downloads/zulu-community/
```

以下简短代码片段将从 Bellsoft 下载并安装 JDK 11：

```
sudo wget https://download.bell-sw.com/java/11.0.5+11/bellsoft-jdk11.0.5+11-linux-arm32-vfp-hflt.deb
sudo dpkg -i bellsoft-jdk11.0.5+11-linux-arm32-vfp-hflt.deb
```

可用的版本确实是 Java 11，如下面的输出所示：

```
pi@raspberrypi:~ $ java -version
openjdk version "11.0.5-BellSoft" 2019-10-15
OpenJDK Runtime Environment (build 11.0.5-BellSoft+11)
OpenJDK 32-Bit Server VM (build 11.0.5-BellSoft+11, mixed mode)
```

现在，Java 开发工具包已安装在你的树莓派上，让我们进入本章的核心内容：在 Java 和树莓派上运行 OpenCV 代码。

#### 检出 OpenCV/Java 模板

我们将从直接在树莓派上检出 OpenCV/Java 项目模板开始。有几种方法可以做到这一点；主要的三种如下：

- 直接对模板执行 Git 克隆
- 从 Git 项目模板下载 zip 文件
- 使用 Maven 生成模板项目

让我们逐一快速回顾这三种方法。

##### 执行 Git 克隆

直接从 Visual Studio Code 将代码检出到你的树莓派是可行且方便的。这通过命令面板并输入任何以 `git` 开头的内容来完成。

你将看到几个选项，其中之一是 Git: Clone，如图 3-24 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig24_HTML.jpg](img/490964_1_En_3_Fig24_HTML.jpg)

图 3-24 Visual Studio Code 中的 Git:Clone 选项

在此处，输入模板仓库的位置，如下所示，如图 3-25 所示：

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig25_HTML.jpg](img/490964_1_En_3_Fig25_HTML.jpg)

图 3-25 输入模板 URL

```
https://github.com/hellonico/opencv-java-template.git
```

最后，在对话框中点击“打开”进行确认（图 3-26）。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig26_HTML.jpg](img/490964_1_En_3_Fig26_HTML.jpg)

图 3-26 当然，要打开克隆的仓库

现在项目已检出到树莓派上，可以执行了。

##### 下载 Zip 文件

或者，你可以从项目模板下载 zip 文件。如果你根本不想使用 Git，可以在终端中使用 `wget` 命令来完成此操作。

```
wget https://github.com/hellonico/opencv-java-template/archive/master.zip
unzip master.zip
```

图 3-27 显示了 Visual Code Studio 的终端标签页。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig27_HTML.jpg](img/490964_1_En_3_Fig27_HTML.jpg)

图 3-27 从树莓派下载 zip 文件

这样做的好处是不依赖 Git，但后果是在不同的树莓派之间共享更新比较困难。

##### 使用 Maven

设置项目的第三种方法与最初创建 Git 模板仓库所使用的技术相同：使用构建工具 Maven。在 Java 生态系统中，相当多的人对使用 Maven 的方式有些畏惧，但对于一组给定的任务来说，Maven 是一个相当可靠的选择。

Maven 一旦安装，就可以通过 Maven 原型为你创建一个包含所有必需文件的项目。

以下命令，一旦复制并粘贴到终端标签页，将为你生成项目结构，如下所示：

```
mvn org.apache.maven.plugins:maven-archetype-plugin:2.4:generate \
-DarchetypeArtifactId=maven-archetype \
-DarchetypeGroupId=origami \
-DarchetypeVersion=1.0 \
-DarchetypeCatalog=https://repository.hellonico.info/repository/hellonico/ \
-Dversion=1.0-SNAPSHOT \
-DgroupId=hello \
-DartifactId=opencv-java-template
```

命令的重要参数已在之前的代码片段中突出显示，并在此处进行说明：

- `archetypeVersion`：这是项目结构的版本，此处为 1.0。可能很快会有一些更新，但不要期望太多。最好创建你自己的版本。
- `version`：这是你项目的版本；通常以 `1.0-SNAPSHOT` 开头。
- `groupId`：这是你项目的 Maven 组，例如 `com.google` 等。
- `artifactId`：这是 `groupId` 内项目的名称。它必须是唯一的。

这里我们使用的是较旧版本的原型插件；使用此版本，更容易从 [`repository.hellonico.info`](http://repository.hellonico.info) 的自定义 Maven 仓库中检索工件。

较新版本的原型插件需要对 Maven 框架有更深入的了解，这超出了本书的范围。

无论你使用哪种技术，现在都可以打开远程文件夹。你应该会看到类似于图 3-28 中的树状结构。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig28_HTML.jpg](img/490964_1_En_3_Fig28_HTML.jpg)

图 3-28 树莓派上的项目文件

再次强调，文件位于树莓派上，Java 代码将在树莓派上执行，但编辑器实际上是在你的本地计算机上运行。

#### 远程安装 Visual Code Java 扩展包

如果你还记得第 1 章的内容，我们曾安装插件让 Visual Studio Code 能够识别并运行 Java。这里我们将进行同样的操作，不过这次是在树莓派上。

图 3-29 展示了“市场”选项卡，你可以在其中搜索之前安装过的同一个插件，即 Java 扩展包。请注意图 3-30 中 Visual Studio Code 如何友好地询问你是否要在树莓派上安装该插件。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig30_HTML.jpg](img/490964_1_En_3_Fig30_HTML.jpg)

图 3-30 再次查找 Java 扩展包

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig29_HTML.jpg](img/490964_1_En_3_Fig29_HTML.jpg)

图 3-29 “SSH. PI4”尚未安装任何扩展

图 3-31 和图 3-32 展示了在树莓派上安装 Visual Code Java 插件的最后两个步骤。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig32_HTML.jpg](img/490964_1_En_3_Fig32_HTML.jpg)

图 3-32 需要再次重新加载编辑器

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig31_HTML.jpg](img/490964_1_En_3_Fig31_HTML.jpg)

图 3-31 再次查找 Java 扩展包

现在，你的编辑器应该会直接显示熟悉的“运行”和“调试”链接，如图 3-33 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig33_HTML.jpg](img/490964_1_En_3_Fig33_HTML.jpg)

图 3-33 显示熟悉的“运行”和“调试”按钮

第一次执行所有这些步骤时可能会比较耗时，但现在你已经准备好迎接愉快的体验了。你已经完全配置好，可以在树莓派上运行 OpenCV/Java 程序了。

#### 运行第一个 OpenCV 示例

闲话少叙，让我们点击“运行”按钮来执行程序。

点击“运行”按钮会执行与第 1 章相同的程序，但这次程序是在树莓派上运行的。输出结果会显示熟悉的 3×3 矩阵，如图 3-34 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig34_HTML.jpg](img/490964_1_En_3_Fig34_HTML.jpg)

图 3-34 代码已在树莓派上执行

请注意，首次运行时后台会进行各种下载，因此命令会花费一些时间。第二次运行时，速度会快很多。

提醒一下，这里的调试方式与第 1 章相同。作为练习，尝试添加一个断点，在运行调试后暂停程序执行，然后为 `System.getProperty()` 添加一个监视表达式，以显示 `java.vm.vendor` 的值。由于我们现在是在树莓派上运行，该值应为 `Raspbian`，如图 3-35 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig35_HTML.jpg](img/490964_1_En_3_Fig35_HTML.jpg)

图 3-35 调试并显示 `java.vm.vendor`

## 改为在 Linux 或 AWS 虚拟机上运行

那么，如果你实际上不是在树莓派上运行，而是在远程 Linux 服务器或云虚拟机上运行呢？

到目前为止你所做的所有设置实际上都非常通用，因此只要你拥有 SSH 访问权限，就可以连接到任何运行 Linux 的标准机器。

例如，在图 3-36 中，你可以看到相同的设置连接到了一个远程 Linux 机器，并运行着我最喜欢的桌面发行版 Manjaro Linux。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig36_HTML.jpg](img/490964_1_En_3_Fig36_HTML.jpg)

图 3-36 远程连接到正在运行的 Linux 机器

为了给你提供更多创意，我也使用相同的设置来连接 Amazon Web Services 上的远程机器。

## 捕获视频直播流

如果您的网络摄像头已通过 USB 端口正确连接，那么现在就可以编写代码，直接从您的物联网设备捕获视频流了。

我们使用两个基础组件来开始。

- OpenCV 的 `VideoCapture` Java 类，用于将硬件连接到软件（更准确地说，是从视频流连接到 OpenCV 的 `Mat` 对象）。在此示例中，`VideoCapture` 对象接受一个 `int` 值作为要使用的视频捕获设备的 ID。

- `ImShow` 类，它不属于 OpenCV Java 核心包。这是一个类似于 OpenCV 的 `HighGui` 类（[`https://docs.opencv.org/master/javadoc/org/opencv/highgui/HighGui.html`](https://docs.opencv.org/master/javadoc/org/opencv/highgui/HighGui.html)）的小型类，但它能更轻松地接入视频流并在屏幕上显示。

我们还使用一个名为 `Origami.init()` 的便捷方法来 `init` OpenCV，该方法会为我们搜索并加载正确的 OpenCV 二进制库。

清单 3-1 展示了这个首个流式示例的完整代码。

```java
package hello;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import origami.ImShow;
import origami.Origami;
public class Webcam {
    public static void main(String[] args) {
        Origami.init();
        VideoCapture cap = new VideoCapture(0);
        Mat matFrame = new Mat();
        ImShow ims = new ImShow("Camera", 800, 600);
        while (cap.grab()) {
            cap.retrieve(matFrame);
            ims.showImage(matFrame);
        }
        cap.release();
    }
}
```

清单 3-1 访问并显示视频流

这段代码是非常标准的 OpenCV 代码。运行代码后，应该会得到如图 3-37 所示的画面（不显示帧率）。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig37_HTML.jpg](img/490964_1_En_3_Fig37_HTML.jpg)

图 3-37 不带帧率的视频流

您会注意到速度非常接近实时。可以将清单 3-2 中的代码添加到帧循环中，以显示帧率，如图 3-38 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig38_HTML.jpg](img/490964_1_En_3_Fig38_HTML.jpg)

图 3-38 带帧率的视频流

```java
long now = System.currentTimeMillis();
frame++;
Imgproc.putText(matFrame, "FPS " + (frame / (1 + (now - start) / 1000)), new Point(50, 50),
    Imgproc.FONT_HERSHEY_COMPLEX, 2.0, new Scalar(255, 255, 255));
```

清单 3-2 在帧上添加每秒帧数

对于这个练习，通常期望的帧率在每秒 15 到 20 帧之间。

在运行前面的示例时，您可能遇到了问题。有时程序会崩溃，并显示清单 3-3 中的堆栈跟踪信息。

```
Exception in thread "main" java.awt.HeadlessException:
No X11 DISPLAY variable was set, but this program performed an operation which requires it.
at java.desktop/java.awt.GraphicsEnvironment.checkHeadless(GraphicsEnvironment.java:208)
at java.desktop/java.awt.Window.(Window.java:548)
at java.desktop/java.awt.Frame.(Frame.java:423)
at java.desktop/java.awt.Frame.(Frame.java:388)
at java.desktop/javax.swing.JFrame.(JFrame.java:180)
at origami.ImShow.(ImShow.java:52)
at hello.Webcam.main(Webcam.java:22)
```

清单 3-3 未设置 X11 DISPLAY

要通过 SSH 运行图形程序（就像您现在正在做的那样），Java 需要设置一个系统环境变量。正如您所见，这个变量的名称是 `DISPLAY`。您可以在终端标签页中使用以下命令检查其值：

```bash
echo $DISPLAY
```

根据此变量的值，您可能处于三种主要状态，如下所示：

- `<empty>`，这不太好。您需要自己设置该值。

- `:0` 或 `localhost:0`，这意味着任何视觉帧都将显示在树莓派的屏幕上。这对于调试设置是可以的，但通常不是您在电脑上工作时想要的状态。

- `:10` 或 `localhost:10`，这意味着帧将显示在您的电脑上。在 macOS 上，这是通过 XQuartz 应用程序完成的，它运行一个与树莓派兼容的图形环境。在 Windows 上，等效的软件是 Xming：[`https://sourceforge.net/projects/xming/`](https://sourceforge.net/projects/xming/)。

如果您正确设置了该变量，那么流将显示在您的电脑屏幕上，如图 3-39 所示。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig39_HTML.jpg](img/490964_1_En_3_Fig39_HTML.jpg)

图 3-39 远程树莓派流直接传输到您的电脑

您已经了解了设置树莓派进行远程开发的所有知识。接下来，让我们看一些分析视频流内容的技巧。

### 播放视频

您可能并不总能访问视频流，有时需要回放录制的视频。使用 `VideoCapture` 类也可以实现这一点。您可以向其提供要播放的视频文件的路径，如下所示：

```java
public class PlayVideo {
    public static void main(String[] args) {
        Origami.init();
        VideoCapture cap = new VideoCapture("marcel_1.mp4");
        Mat matFrame = new Mat();
        ImShow ims = new ImShow("Camera", 400, 300);
        long start = System.currentTimeMillis();
        long frame = 0;
        while (cap.grab()) {
            cap.retrieve(matFrame);
            long now = System.currentTimeMillis();
            frame++;
            Imgproc.putText(matFrame, "FPS " + (frame / (1 + (now - start) / 1000)), new Point(50, 50),
                Imgproc.FONT_HERSHEY_COMPLEX, 2.0, new Scalar(255, 255, 255));
            ims.showImage(matFrame);
        }
        cap.release();
    }
}
```

这段代码与播放来自网络摄像头的流相同，只是文件名作为参数传递给 `VideoCapture` 构造函数。

请注意，您还可以在以下网站上找到一些免费的示例视频：

```
https://sample-videos.com/index.php
```

图 3-40 展示了 Marcel 早晨的一些活动。

![../images/490964_1_En_3_Chapter/490964_1_En_3_Fig40_HTML.jpg](img/490964_1_En_3_Fig40_HTML.jpg)

图 3-40 早晨的 Marcel

如果您尝试输入一个 `http://` URL 作为 `VideoCapture` 的路径，您会发现事情并不像预期的那样工作。您想要播放的远程视频将无法加载。

路径中的 URL 用于播放来自网络摄像头的流，而不是静态文件。如果您手头有网络摄像头，那么将网络摄像头的 Web URL 直接插入 `VideoCapture` 对象是一个很好的练习。

## 在树莓派上分析视频流

在本章中，您将学习如何使用函数式编程的概念来分析视频流。具体来说，您将使用 `Filter` 接口，并将其与 `Pipeline` 对象结合，然后将它们应用于视频流。

我们将从过滤器的概述开始。然后，我们将研究不同的基本、有趣的过滤器，并逐步过渡到使用不同视觉技术的对象检测。最后，我们将讨论神经网络。