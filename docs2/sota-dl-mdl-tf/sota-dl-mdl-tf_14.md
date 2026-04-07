# 14. 强化学习简介

**强化学习**（RL）是机器学习的一个领域，它专注于教授智能体如何在环境中采取行动以最大化累积奖励。在 RL 中，“累积奖励”是所有奖励的总和，作为训练步骤数量的函数。

我们通过使用奖励和惩罚来训练机器学习（ML）模型。当智能体做出正确决定时，我们用正点奖励它。对于错误的决定，我们用负点惩罚它。从这些反应中，模型学习如何在该特定情况（或环境）中做出反应。所以 RL 背后的想法是，智能体通过与环境的互动并因执行行动而获得奖励（和惩罚）来从环境中学习。

从与环境的互动中学习源于我们的自然经验。想象你是一个客厅里的孩子。你看到一个壁炉，你走近它。它很温暖，很积极，你感觉很好。你理解火是一个积极的事物。但然后你触摸了火。哎呦！它烧伤了你的手。从你的互动中，你了解到当你保持足够距离时，火是积极的，因为它产生温暖。但太靠近了就会烧伤。所以人类通过与环境互动来学习。

RL 只是从行动中学习的一种计算方法。如果这个场景是一个 RL 实验，智能体会因为靠近火而获得+1 的正奖励，但会因为被烧伤而获得-1 的负奖励。

RL 是三种基本 ML 范式之一，与监督学习和无监督学习并列。监督学习模型在有指导的情况下从标记数据集中学习。无监督学习模型在没有指导的情况下从未标记数据中学习。RL 模型在智能体与环境交互、执行行动并因行动正确而获得奖励或因行动错误而受到惩罚时通过试错来学习。

强化学习的三个组成部分是智能体、环境和行动。一个*环境*是要解决的问题。一个*智能体*是与环境交互以解决问题的算法。*行动*是智能体与环境的交互。

简而言之，智能体从环境中接收观察结果并采取行动。环境使用奖励或惩罚作为信号，根据智能体采取的行动来表示积极或消极的行为。因此，智能体通过与环境进行试错互动来学习如何解决问题。

尽管设计者设置了奖励策略，但他们没有给学习模型任何关于如何解决问题的提示或建议。模型通过智能体与环境之间的试错互动来“自行”学习如何最大化奖励。

**奖励策略**是一组规则，用于最大化给定强化学习环境的奖励函数。因此，*奖励函数*规定了设计者希望智能体完成的目标。

## 强化学习的挑战

强化学习的主要挑战是**创建环境**。创建一个有效的环境有三个问题。

首先，环境高度依赖于要解决的问题。因此，它是上下文特定的。问题域的上下文驱动着每个强化学习环境的构建，以解决特定的一组任务。简单来说，每个新的强化学习任务都需要设计一个新的环境。例如，自动驾驶汽车的环境不能转移到无人机的环境中。相比之下，监督学习分类模型可以被其他分类任务重用。

其次，奖励策略是有限的且结构化的。国际象棋、围棋和 Atari 游戏的环境相对简单。无论这些游戏可能看起来多么复杂，它们的规则都是结构化和有限的。即使是自动驾驶汽车的极其复杂的环境也是有限和结构化的。尽管这样的环境必须处理许多未知因素，但其奖励策略是由设计者创建的。即使是最聪明的设计师也无法创建无限的奖励策略。例如，自动驾驶汽车可能被设计成在各种条件下都能良好工作，但它可能无法适应与设计时不同的环境。自动驾驶汽车可能在行驶特定路线时工作，但如果需要绕行，而绕行标志尚未设置，会发生什么？

第三，在安全是关注点的地方，错误的空间几乎为零。医疗保健行业的强化学习实验就值得考虑。这些模型必须在许多阶段进行测试和调整，才能准备好进行简单的测试。将模型从测试阶段转移到现实世界是真正工作的开始。

缩放和调整控制智能体的神经网络（或其他机器学习模型）是另一个挑战。与网络通信的唯一方式是通过其奖励和惩罚系统。通过这个单一的通道，可能出现灾难性遗忘（或灾难性干扰）。**灾难性遗忘**是指神经网络在学习新信息时完全突然忘记之前学习的信息的倾向。因此，当神经网络获取新信息时，一些旧信息会被从网络中删除。灾难性遗忘发生是因为在学习新信息时，神经网络中许多（存储信息的）权重发生了变化，这使得保留先验知识变得不太可能。在顺序学习中，进入神经网络的新输入可能会删除原始输入权重。

另一个挑战是达到局部最优。**局部最优**是在可能解的小邻域内问题的最佳解决方案。相比之下，**全局最优**是在考虑了**所有**可能解时的最优解决方案。当目标是学习如何移动（行走、跑步和跳跃）时，局部最优将是行走。在这种情况下，智能体以次优（局部最优）的方式完成任务，但不是（全局）最优的方式。

最后一个挑战是吸引有才华的设计师。找到一位合格的数据科学家可能不难，但平均年薪可能超过 150,000 美元。此外，可能很难找到一位有创建特定类型环境经验的，因为环境是上下文特定的。

各章节的笔记本位于以下 URL：

[`https://github.com/paperd/deep-learning-models`](https://github.com/paperd/deep-learning-models)

我们通过使用一个**非常简单**的强化学习环境进行代码实验来展示强化学习。通过导入主 TensorFlow 库并实例化 GPU 来开始设置 Colab 生态系统。

## 导入 TensorFlow 库

导入库并将其别名为 **tf**：

```py
import tensorflow as tf
```

将 TensorFlow 库别名为 tf 是常见做法。

## GPU 硬件加速器

为了方便，我们提供了在 Colab 笔记本中启用 GPU 的步骤：

1.  在右上角菜单中点击**运行时**。

1.  从下拉菜单中选择**更改运行时类型**。

1.  从**硬件加速器**下拉菜单中选择**GPU**。

1.  点击**保存**。

验证 GPU 是否处于活动状态：

```py
tf.__version__, tf.test.gpu_device_name()
```

如果显示 `/device:GPU:0`，则 GPU 处于活动状态。如果显示 `..`，则常规 CPU 处于活动状态。

备注

如果出现错误 `NAME ‘TF’ IS NOT DEFINED`，请重新执行代码以导入 TensorFlow 库！

## 强化学习实验

我们通过 Cart-Pole 进行一个简单的强化学习实验。*Cart-Pole* 是一个杆通过非驱动关节连接到小车，小车在无摩擦轨道上移动的游戏。游戏的目标是保持杆垂直竖立。起始状态在 -0.05 和 0.05 之间随机初始化。起始状态包括小车位置、小车速度、杆角度和杆速度。杆速度是在杆尖测量的。Cart-Pole 游戏在 2D 空间中。因此，小车只能左右移动以平衡杆。

实验的目的是通过强化学习模型教会智能体如何在车上平衡杆。我们可以创建自己的环境来训练智能体，但不必这样做，因为解决这个简单的杆平衡问题的环境已经存在。我们使用 OpenAI Gym 的环境。*OpenAI Gym* 是一个工具包，提供了包括 Atari 游戏、棋类游戏和（2D 和 3D）物理模拟在内的各种模拟环境。

### 在 Colab 上安装和配置 OpenAI Gym

对于 OpenAI Gym 的 Python 包的大部分要求已经在 Colab 上得到满足。但我们仍然需要安装依赖项：

```py
!pip install gym
!apt-get install python-opengl -y
!apt install xvfb -y
```

我们需要 gym、opengl 和 xvfb 依赖项来启用 Python 库中的 OpenAI Gym、用于可视化的图形库以及用于渲染图形的显示服务器。*Gym* 是一个开源接口，用于 RL 任务。它支持教授代理从行走到玩 Pong 或弹球等游戏的一切。

*OpenGL* 是一个支持多个平台（包括 Windows、Linux 和 MacOS）的图形库。*Xvfb*（X 虚拟帧缓冲区）是一个实现 X11 显示服务器协议的显示服务器。它在内存中运行，不需要物理显示。

我们还需要安装用于显示从环境中渲染输出的依赖项：

```py
!pip install pyvirtualdisplay
!pip install piglet
```

*pyvirtualdisplay* 库使虚拟显示成为可能。*piglet* 库提供了一个面向对象的 API，用于创建游戏和其他多媒体应用程序。

### 导入库

导入并激活虚拟显示库：

```py
import pyvirtualdisplay
display = pyvirtualdisplay.Display(
visible=0, size=(1400, 900)).start()
```

我们为虚拟显示选择了 1400 × 900。我们选择的大小是任意的。请随意尝试不同的显示尺寸。

导入 *gym* 库：

```py
import gym
```

`gym` 库是一系列为测试和开发 RL 算法而设计的环境。它使我们免于自己创建复杂的环境。

### 创建环境

创建 *Cart-Pole* 环境：

```py
env = gym.make('CartPole-v1')
```

Cart-Pole 环境是一个 2D 模拟，它通过加速小车向左或向右移动来平衡放置在其顶部的小杆。一根杆通过非驱动关节连接到沿无摩擦轨道移动的小车。系统通过向小车施加+1 或-1 的力来控制。摆锤（或杆）开始时是竖直的，目标是防止它倒下。

初始化环境：

```py
env.seed(0)
obs = env.reset()
obs
```

使用 *reset* 方法初始化环境。初始化后，该方法返回一个观察值。

观察值取决于环境。在这种情况下，一个观察值是一个由四个浮点数组成的 1D NumPy 数组，这些浮点数代表小车的水平位置、速度、杆的角度（0 = 垂直）和角速度。任何正数表示杆的角度和角速度向 *右* 移动。任何负数表示向 *左* 移动。对于水平位置，负数表示杆 *向左倾斜*，正数表示 *向右倾斜*。对于速度，正数表示小车 *加速*，负数表示 *减速*。

初始渲染状态（种子为 0）生成一个 1D NumPy 数组：

array([-0.04456399, 0.04653909, 0.01326909, -0.02099827])

因此，杆的初始状态并非完全水平（obs[0]略为负值），其速度缓慢增加（obs[1]略为正值），杆略微向右倾斜（obs[2]略为正值），角速度向左移动（obs[3]略为负值）。

可以通过调用其 *render* 方法来可视化环境，我们可以选择渲染模式（渲染选项取决于环境）。显示初始状态下的渲染环境：

```py
env.render()
```

在我们的情况下，env.render() 命令显示 *True*，这是初始状态，因为我们刚刚这样初始化它。

在强化学习中，我们必须将环境渲染（初始化）到其初始状态，作为训练的前奏。原因是我们要从初始状态开始环境，不给代理任何线索。我们希望代理能够从头开始学习如何解决问题。

### 显示环境渲染结果

设置 mode=‘rgb_array’ 以获取环境图像作为 NumPy 数组：

```py
img = env.render(mode='rgb_array')
img.shape
```

图像形状是从 Cart-Pole 环境渲染的。

创建一个函数来显示环境配置如列表 14-1 中所示的环境渲染的极点位置图像。

```py
def plot_environment(env, figsize=(5,4)):
plt.figure(figsize=figsize)
img = env.render(mode='rgb_array')
plt.imshow(img)
plt.axis('off')
return img
Listing 14-1
Function to Display the Rendered Environment
```

显示：

```py
import matplotlib.pyplot as plt
plot_environment(env)
plt.show()
```

我们看到环境在初始化状态下的渲染结果。所以极点在车上几乎是垂直的，但略微向右倾斜。

### 显示动作

让我们看看如何与我们所创建的环境交互。代理需要从行为空间中选择一个动作。**行为空间**是代理可以采取的可能行为的集合。

询问环境关于可能动作的信息：

```py
env.action_space
```

*Discrete(2)* 表示 Cart-Pole 环境的可能动作是整数 0 和 1。向左加速是 0，向右加速是 1。所以环境的行为空间有两个可能的行为，这意味着代理可以加速向左或向右。当然，其他环境可能有额外的离散动作或其他类型的动作，如连续动作。

注意

Cart-Pole 环境是可以创建的最简单的一个！现实世界的强化学习环境具有巨大的动作空间，包含许多可能的行为。

重置环境并查看极点的倾斜角度：

```py
env.seed(0)
obs = env.reset()
indx = 2
obs[indx]
```

在 *obs* 数组的第三个位置（索引为 2）是极点的角度。如果值小于 0，极点向左倾斜。如果大于 0，则向右倾斜。这个值几乎超过零。所以极点略微向右倾斜，因为 obs[2] 大于 0。

注意

我们不需要再次重置环境，因为我们已经在上一节中这样做过了。但是，应该在实验开始时重置环境，以确保代理能够从头开始学习如何解决问题（没有任何线索）。所以我们再次这样做，只是为了培养良好的强化学习习惯。

如我们所知，Cart-Pole 环境只有两个动作，左（0）和右（1）。让我们通过设置 *action=1* 来加速小车向右：

```py
action = 1
obs, reward, done, info = env.step(action)
print ('obs array:', obs)
print ('reward:', reward)
print ('done:', done)
print ('info:', info)
```

*step*方法执行给定的动作并返回四个值。*obs*是新的观察。由于 obs[1] > 0，小车现在向右移动。杆仍然向右倾斜，因为 obs[2] > 0，但它的角速度现在是负的，因为 obs[3] < 0。因此，在下一步后它可能会向左倾斜。

在这个最简单的环境中，*奖励*在每一步都是 1.0。因此，目标是尽可能长时间地保持游戏进行。当游戏结束时，*done*值为 True。如果杆倾斜过多或超出屏幕或我们赢得游戏，则游戏结束。*info*值提供额外信息。在这种情况下，没有额外信息。一旦我们完成对环境的操作，调用*close*方法来释放资源。

注意

由于我们的实验非常简单，因此没有必要释放资源。但是，对于更复杂的 RL 任务来说，这是一个好主意，因为它们往往会消耗大量的计算机资源。

环境告诉代理每个新的观察、奖励、游戏结束时的情况以及它在最后一步获得的信息。显示杆的位置：

```py
plot_environment(env)
plt.show()
```

杆仍然向右倾斜。

显示代理在最后一步收到的奖励：

```py
reward
```

当然，奖励是 1.0，因为在这个最简单的实验中它始终是 1.0。

测试游戏是否结束：

```py
done
```

由于值为*False*，游戏尚未结束。如果输出为*True*，则任务完成，游戏结束。

*Episode*是从环境重置的那一刻开始直到完成的步骤序列。在游戏结束时（即当 step 方法返回 done=True 时），在继续使用之前重置环境。

为了在游戏结束时自动重置：

```py
if done:
obs = env.reset()
else:
print ('game is not over!')
```

注意

我们仅为了信息目的向您展示了前面的代码。我们在 RL 实验的上下文中向您展示了何时使用 env.reset()。

### 简单神经网络奖励策略

我们如何让杆保持直立？我们需要定义一个奖励策略。在行动中，奖励策略是代理在每个步骤选择动作的策略。代理可以使用所有过去的行为和观察来决定下一步该做什么。

让我们创建一个神经网络，它以观察作为输入，并为每个观察输出要采取的动作。为了选择一个动作，我们的网络估计每个动作的概率，并根据估计的概率随机选择一个动作。在 Cart-Pole 环境中，只有两种可能的行为（左和右）。因此，我们只需要一个输出神经元，该神经元输出动作 0（左）的概率*p*和动作 1（右）的概率*1* – *p*。

清除之前的模型并生成一个种子：

```py
import numpy as np
tf.keras.backend.clear_session()
tf.random.set_seed(0)
np.random.seed(0)
```

确定观察空间：

```py
obs_space = env.observation_space.shape
obs_space
```

*观察空间*是奖励策略的另一个术语。观察空间（如前所述的观察数组中所示）是一个由四个浮点数组成的 1D NumPy 数组，这些浮点数代表滑车的水平位置、速度、杆的角度（0 = 垂直）和角速度。因此，观察空间包含在一个具有 4 个元素的 1D NumPy 数组中。

注意

术语策略、奖励策略、观察数组和观察空间可以互换使用。

设置策略网络的输入数量：

```py
n_inputs = env.observation_space.shape[0]
n_inputs
```

创建策略网络：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
Dense(5, activation='elu', input_shape=[n_inputs]),
Dense(1, activation='sigmoid')
])
```

策略网络是一个简单的 *Sequential* 神经网络。输入数量是观察空间的大小，在我们的例子中是 4。我们只在第一层包含五个神经元，因为这是一个如此简单的问题。我们只需要输出一个概率（向左移动的概率）。因此，我们在输出层使用 sigmoid 激活来生成单个输出神经元作为 logit。我们只需要输出一个概率 p，因为我们可以通过 1 – p 得到向右移动的概率。如果我们有超过两个可能的动作，我们仍然会为每个动作使用一个输出神经元，并在输出层替换 softmax 激活。

在这个特定环境中，由于每个观察包含环境的完整状态，因此可以安全地忽略过去的行为和观察。如果有某些隐藏状态，我们可能需要考虑过去的行为和观察来尝试推断环境的隐藏状态。例如，如果环境只揭示了滑车的位置而没有其速度，我们就必须考虑当前观察以及之前的观察，以便估计当前速度。另一个例子是，如果观察是噪声的，我们可能希望使用过去几项观察来估计最可能的状态。我们的问题非常简单，因为当前观察是无噪声的，并且包含环境的完整状态。

为什么我们根据策略网络的概率随机选择动作，而不是仅仅选择概率最高的动作？因为这种方法让智能体在探索新动作和利用已知效果良好的动作之间找到正确的平衡。

### 模型预测

创建一个函数，运行模型以播放一个回合并返回帧，以便我们可以显示如图 14-2 所示的动画。

```py
def render_policy_net(model, n_max_steps=200, seed=0):
frames = []
env = gym.make('CartPole-v1')
env.seed(seed)
np.random.seed(seed)
obs = env.reset()
for step in range(n_max_steps):
frames.append(env.render(mode='rgb_array'))
left_proba = model.predict(obs.reshape(1, -1))
action = int(np.random.rand() > left_proba)
obs, reward, done, info = env.step(action)
if done:
break
env.close()
return frames
Listing 14-2
Function to Return the Frames from One Episode
```

建立 Cart-Pole 环境并重置它。创建一个循环，运行多个步骤直到回合结束。在每个步骤开始时，将环境渲染的可视化追加到 *frames* 列表中。接着从模型中进行动作预测。然后，根据预测建立动作。根据动作执行 *step* 方法。继续循环直到回合结束。最后，返回帧列表。

创建函数以显示如图 14-3 所示的帧的动画。

```py
import matplotlib.animation as animation
import matplotlib as mpl
def update_scene(num, frames, patch):
patch.set_data(frames[num])
return patch,
def plot_animation(frames, repeat=False, interval=40):
fig = plt.figure()
patch = plt.imshow(frames[0])
plt.axis('off')
anim = animation.FuncAnimation(
fig, update_scene, fargs=(frames, patch), blit=True,
frames=len(frames), repeat=repeat, interval=interval)
plt.close()
return anim
Listing 14-3
Functions to Animate Frames
```

`*plot_animation*`函数通过重复调用`*update_scene*`函数来创建动画。`plot_animation`函数接受帧列表。然后，它从帧列表中提取一个图像（块）。接着，它调用`update_scene`来设置块的 x 和 y 坐标。动画返回的每个帧图像都批处理了块坐标。

根据策略网络创建帧列表：

```py
frames = render_policy_net(model)
```

### 动画

创建动画：

```py
anim = plot_animation(frames, interval=100)
```

尝试调整*间隔*参数以观察其对接动画的影响。我们设置间隔大小为 100，仅仅是因为我们喜欢这个结果。

注意

增加间隔只是延长了动画运行的时间。

渲染并显示动画。我们展示了两种实现方式。第一种方法使用 HTML 库来显示 HTML 元素：

```py
from IPython.display import HTML
method1 = HTML(anim.to_html5_video())
method1
```

使用`to_html5_video`方法将动画渲染为*html5 视频*，并通过 HTML 模块显示。

第二种方法使用运行时配置库：

```py
from matplotlib import rc
method2 = rc('animation', html='html5')
```

要实现第二种方法，只需运行动画对象：

```py
anim
```

哎！杆子向左倒了！原因是我们还没有实现奖励策略。

### 实现基本奖励策略

在上一节中，我们只是让策略网络随机预测，与**代理**没有交互。如果杆倾斜向左，代理无法知道这是否是一个坏动作。我们需要开发一个具有奖励策略的环境，让代理可以与之交互，以学习如何在车上平衡杆。

创建一个如列表 14-4 所示的多样化环境空间。

```py
n_environments = 50
n_iterations = 5000
envs = [gym.make(
'CartPole-v1') for _ in range(n_environments)]
for index, env in enumerate(envs):
env.seed(index)
np.random.seed(0)
observations = [env.reset() for env in envs]
optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.binary_crossentropy
for iteration in range(n_iterations):
target_probas = np.array(
[([1.] if obs[2] \
left_probas.numpy()).astype(np.int32)
for env_index, env in enumerate(envs):
obs, reward, done, info = env.step(
actions[env_index][0])
observations[env_index] = obs if not done else env.reset()
for env in envs:
env.close()
Listing 14-4
Environment Space for the Experiment
```

我们并行创建了一个包含 50 个不同环境的环境空间，以在每个网络步骤中提供多样化的训练批次。我们对网络进行了 5,000 次迭代训练。当然，你可以调整环境数量和迭代次数，但要注意可用的计算机资源。

我们使用 RMSprop 优化器，因为它似乎效果很好。你可以尝试不同的优化器，看看代理如何学习任务。我们使用二元交叉熵作为损失函数，因为我们只有两个离散的可能动作（左和右）。迭代循环的第一行检查杆的角度。如果角度 < 0，目标动作是左（proba(left) = 1.）。否则，目标动作是右（proba(left) = 0）。

**优化器**是用于改变神经网络（如权重和学习率）属性以减少损失的计算算法（或方法）。因此，优化器通过最小化损失函数来解决优化问题。

要查看 TensorFlow 优化器的完整列表，请参阅

[`www.tensorflow.org/api_docs/python/tf/keras/optimizers`](http://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

梯度带部分记录了代理在训练期间的动作。杆子的倾斜被确定。代理采取行动使杆子在车上稳定。然后，这些动作被输入到 50 个环境中，这些环境决定了给予代理的奖励。这个过程重复了 5,000 次。

最后，环境被重置以释放资源。我们使用自定义训练循环进行训练，这样我们就可以轻松地使用每个训练步骤的预测来推进环境。

创建动画的框架：

```py
frames = render_policy_net(model)
```

我们现在有了基于代理在训练期间学习到的框架。我们使用这些框架来创建动画。

动画：

```py
anim = plot_animation(frames, repeat=True, interval=100)
anim
```

哇！动画验证了我们的强化学习模型是有效的。也就是说，我们教会了代理如何在车上平衡杆子。但我们能否做得更好？

### 强化策略梯度算法

我们尚未展示强化学习的真正突破。在上一节中，代理从奖励策略中学习。但代理能否自己学习更好的策略？

我们可以使用强化策略梯度算法来自动化代理学习。*策略梯度*通过跟随梯度指向更高的奖励来优化策略的参数。

#### 策略梯度是如何工作的？

要使用策略梯度，让神经网络策略玩游戏几次。在每一步，计算使所选动作更可能发生的梯度。但在此点不要应用梯度。

在运行了几个剧集后，计算每个动作的优势，每个步骤都有一个折现因子。*折现因子*是通过评估动作基于该动作之后所有奖励的总和来计算的。如果一个动作的优势是正的，那么这个动作可能很好。所以**现在**应用梯度，使该动作在未来更有可能被选择。如果它是负的，则应用相反的梯度，使该动作不太可能被选择。最后，计算所有结果梯度向量的平均值，并使用它来执行*梯度下降*步骤。所有结果梯度向量的平均值是通过计算每个梯度（或相反的梯度）乘以其动作优势的平均值来计算的。最终结果是创建了一个有效的策略梯度下降算法，该算法最小化了损失函数。

#### 使用策略梯度训练模型

首先，创建函数来播放一个步骤、播放多个剧集、折现奖励和归一化折现奖励。

创建一个函数，如列表 14-5 所示，来播放一个步骤。

```py
def play_one_step(env, obs, model, loss_fn):
with tf.GradientTape() as tape:
left_proba = model(obs[np.newaxis])
action = (tf.random.uniform([1, 1]) > left_proba)
y_target = tf.constant(
[[1.]]) - tf.cast(action, tf.float32)
loss = tf.reduce_mean(loss_fn(
y_target, left_proba))
grads = tape.gradient(loss, model.trainable_variables)
obs, reward, done, info = env.step(
int(action[0, 0].numpy()))
return obs, reward, done, grads
Listing 14-5
Function to Play a Single Step
```

在 *GradientTape* 块中，用单个观察值调用模型。将观察值重塑为一个包含单个实例（模型期望一个批次）的批次。通过在 0 和 1 之间采样一个随机浮点数并检查它是否大于概率来获取向左移动的概率。如果概率是 left_proba，则 *action* 为 False；如果概率是 1 – left_proba，则 *action* 为 True。将这个布尔值（True 或 False）转换为 0（左）或 1（右）的数字，并使用适当的概率。然后我们定义向左移动（1 – action）或向右移动（action）的目标概率。如果动作是 0（左），则向左移动的目标概率是 1。如果动作是 1（右），则目标概率是 0。

继续计算损失，并使用带子计算模型可训练变量的损失梯度。根据该动作表现的好坏调整梯度。最后，执行选定的动作并返回新的观察值、奖励、回合是否结束以及梯度。

创建一个函数来播放多个回合，并返回每个回合和每一步的奖励和梯度，如列表 14-6 所示。

```py
def play_multiple_episodes(
env, n_episodes, n_max_steps, model, loss_fn):
all_rewards = []
all_grads = []
for episode in range(n_episodes):
current_rewards = []
current_grads = []
obs = env.reset()
for step in range(n_max_steps):
obs, reward, done, grads = play_one_step(
env, obs, model, loss_fn)
current_rewards.append(reward)
current_grads.append(grads)
if done:
break
all_rewards.append(current_rewards)
all_grads.append(current_grads)
return all_rewards, all_grads
Listing 14-6
Play Multiple Episodes Function
```

函数通过调用 *play_one_step* 函数来返回所需步数的奖励列表。此列表包含每个回合一个奖励列表。每个奖励列表包含每一步一个奖励。该函数还返回一个梯度列表。此列表包含每个回合一个梯度列表。每个梯度列表包含每一步一个梯度元组。每个元组包含一个可训练变量的梯度张量。

简单来说，策略梯度算法使用 *play_multiple_episodes* 函数多次玩游戏。然后它回过头来查看所有奖励以折扣和归一化它们。

#### 折扣和归一化奖励

为了折扣和归一化奖励，我们折扣奖励并将它们归一化。*折扣因子*决定了在训练过程中 RL 算法的当前状态下每个奖励的重要性。想法是教会智能体如何在学习过程中评估每个奖励的重要性，以便它可以与训练过程中的未来奖励进行比较。*归一化折扣奖励*使得奖励（正强化）的梯度更陡峭，而惩罚（负强化）的梯度更平缓。简单来说，奖励触发更陡峭的梯度，而惩罚在归一化折扣奖励时触发更平缓的梯度。更陡峭的梯度可以加速损失函数的最小化，这是任何机器学习算法的目标。

*discount_rewards* 函数按照列表 14-7 所示对奖励进行折扣。

```py
def discount_rewards(rewards, discount_rate):
discounted = np.array(rewards)
for step in range(len(rewards) - 2, -1, -1):
discounted[step] += discounted[step + 1] * discount_rate
return discounted
Listing 14-7
Discount Rewards Function
```

验证函数是否正常工作：

```py
discount_rewards([10, 0, -50], discount_rate=0.8)
```

我们给函数提供了三个动作。每个动作后都有一个奖励。第一个奖励是 10，第二个是 0，第三个是-50。我们使用 80%的折扣因子。所以第三个动作得到-50（最后一个奖励的全部信用）。但第二个动作只得到-40（最后一个奖励的 80%信用）。最后，第一个动作得到-40 的 80%（-32）加上第一个奖励的全部信用（+10），导致折扣奖励为-22。

注意

我们给*discount_rewards*函数提供了三个动作，一个包含三个值的列表（或向量），即[10, 0, -50]。我们可以通过向列表中添加或删除值来改变动作的数量。

*discount_and_normalize_rewards*函数按照列表 14-8 所示归一化折扣奖励。

```py
def discount_and_normalize_rewards(
all_rewards, discount_rate):
all_discounted_rewards =\
[discount_rewards(rewards, discount_rate)
for rewards in all_rewards]
flat_rewards = np.concatenate(all_discounted_rewards)
reward_mean = flat_rewards.mean()
reward_std = flat_rewards.std()
return [(discounted_rewards - reward_mean) / reward_std
for discounted_rewards in all_discounted_rewards]
Listing 14-8
Normalize Discounted Rewards Function
```

为了在所有剧集中对所有折扣奖励进行归一化，计算所有折扣奖励的均值和标准差。从每个折扣奖励中减去均值，然后除以标准差。让我们用两个剧集来尝试这个函数：

```py
discount_and_normalize_rewards(
[[10, 0, -50], [10, 20]], discount_rate=0.8)
```

第一剧集（第一个数组）中的所有动作都被认为是坏的，因为归一化优势都是负的。这很有道理，因为奖励的总和是-40（10 + 0 + -50）。相反，第二剧集的动作（第二个数组）是好的，因为归一化优势是正的。奖励的总和是 30（10 + 20）。

#### 训练学习者

如我们所知，RL 使用学习模型和环境空间来使智能体学会如何解决问题。因此，智能体在训练过程中学会如何解决问题。因此，我们可以将这个过程视为训练学习者。

首先定义一组超参数：

```py
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95
```

我们设置了 150 次训练迭代。当然，你可以调整这个数字和其他值。每次迭代玩 10 个游戏剧集，每个剧集最多持续 200 步，并使用折扣率 0.95。

定义策略网络的优化器和损失函数：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.binary_crossentropy
```

使用二元交叉熵，因为我们正在训练一个二元分类器（两个可能动作：左和右）。

生成一个种子，清除之前的模型，并创建一个简单的策略网络：

```py
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
model = Sequential([
Dense(5, activation='elu', input_shape=[4]),
Dense(1, activation='sigmoid'),
])
```

强化学习中的策略网络非常简单。创建环境空间是困难的部分。

按照列表 14-9 所示训练学习者。

```py
env = gym.make('CartPole-v1')
env.seed(42);
for iteration in range(n_iterations):
all_rewards, all_grads = play_multiple_episodes(
env, n_episodes_per_update, n_max_steps,
model, loss_fn)
total_rewards = sum(map(sum, all_rewards))
print('\rIteration: {}, mean rewards: {:.1f}'.format(
iteration, total_rewards / n_episodes_per_update),
end='')
all_final_rewards = discount_and_normalize_rewards(
all_rewards, discount_rate)
all_mean_grads = []
for var_index in range(len(model.trainable_variables)):
mean_grads = tf.reduce_mean(
[final_reward * all_grads[episode_index][step][var_index]
for episode_index, final_rewards in enumerate(
all_final_rewards)
for step, final_reward in enumerate(
final_rewards)], axis=0)
all_mean_grads.append(mean_grads)
optimizer.apply_gradients(
zip(all_mean_grads, model.trainable_variables))
env.close()
Listing 14-9
Train the Learner
```

要训练学习者，首先调用*play_multiple_episodes*（在每个训练迭代中）来玩 10 次游戏，并返回每个剧集和每一步的所有奖励和梯度。接下来，调用*discount_and_normalize_rewards*来计算每个动作的归一化优势，这为我们提供了一个衡量每个动作实际上是好是坏的反向指标。对于每个可训练变量，计算所有剧集和所有步骤的加权平均梯度，加权因子为最终奖励。*final_reward*是每个动作的归一化优势。最后，使用优化器应用平均梯度，这会调整模型的可训练变量，以期使策略变得更好。

注意

训练学习器需要一些时间。所以请耐心等待。

#### 从强化策略梯度算法渲染帧

渲染帧：

```py
frames_ra = render_policy_net(model)
```

#### 动画策略

动画：

```py
anim = plot_animation(frames_ra, repeat=True, interval=100)
anim
```

极端似乎不太摇晃。代理自己学会更好的策略，这真是一种惊人的事情！

## 摘要

我们介绍了强化学习（RL）的概念，并通过一个简单的实验来演示它。即使在最简单的环境空间中，实验中的代码也非常复杂。将强化学习应用于现实世界问题的进化，在于设计者创建能够真实反映我们所处世界的环境空间的能力。
