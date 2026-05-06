# 张量：归一化图像。

```
for t, m, s in zip(tensor, self.mean, self.std):
    t.mul_(s).add_(m)
return tensor
```

`model.cpu()`
`unnormalizer = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2255))`
`class_dict = {0: 'anomaly', 1: 'clean'}`
`example_sample(model=model, data_loader=test_loader_cln, unnormalizer=unnormalizer, class_dict=class_dict)`

输出结果如图 7-3 所示。

![](img/520381_1_En_7_Fig3_HTML.jpg)

输出异常检测分类模型的图像集合。第一行中的五张图像分别是：一匹棕色的马、一辆红色的汽车、一辆黑色的汽车、一只黑色的猫，以及另一辆黑色的汽车。这些图像顶部都有标签，依次为：`P: anomaly T: anomaly`；`P: clean T: clean`，后续图像也遵循相同的顺序。

**图 7-3** 输出图像

混淆矩阵如下所示：

```
def conf_matrix(model, input_data_ldr, input_dvc):
    trgt_data, pred_data = [], []
    with torch.no_grad():
        for i, (features, targets) in enumerate(input_data_ldr):
            features = features.to(input_dvc)
            targets = targets
            preds = model(features)
            _, predicted_labels = torch.max(preds, 1)
            trgt_data.extend(targets.to('cpu'))
            pred_data.extend(predicted_labels.to('cpu'))
    pred_data = pred_data
    pred_data = np.array(pred_data)
    trgt_data = np.array(trgt_data)
    lable_values = np.unique(np.concatenate((trgt_data, pred_data)))
    if lable_values.shape[0] == 1:
        if lable_values[0] != 0:
            lable_values = np.array([0, lable_values[0]])
        else:
            lable_values = np.array([lable_values[0], 1])
    n_labels = lable_values.shape[0]
    lst = []
    z = list(zip(trgt_data, pred_data))
    for combi in product(lable_values, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat

def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None):
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues
    if figsize is None:
        figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)
    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)
    if colorbar:
        fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
            if show_normed:
                cell_text += "\n" + '('
                cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax

mat = conf_matrix(model=model, data_loader=test_loader_cln, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_dict.values())
plt.show()
```

输出结果如图 7-4 所示。

![](img/520381_1_En_7_Fig4_HTML.jpg)

该图展示了混淆矩阵。x 轴标记为“预测标签”，y 轴标记为“真实标签”。两个轴上的值分别为“异常”和“干净”。按顺时针方向的值依次为：15、0、48 和 0。

**图 7-4** 混淆矩阵

## 方法 2：使用自编码器

构建一个自编码器训练网络，该网络包含以下部分：

*   **编码器**：对原始图像进行编码（基于像素值）。
*   **解码器**：根据编码器的输出重建图像。

评估原始图像与重建图像之间的模型。基于误差度量分数，检测出最异常的图像。

以下是五个实现步骤：

1.  步骤 1：准备数据集对象。
2.  步骤 2：构建自编码器网络。
3.  步骤 3：训练自编码器网络。
4.  步骤 4：基于原始数据计算重建损失。
5.  步骤 5：基于误差度量分数选择最异常的图像。

### 步骤 1：准备数据集对象

加载 `input.csv` 文件。`input.csv` 文件中的每条记录包含 65 个值。前 64 个值表示手写数字的灰度像素值。最后一个值表示数字的原始类别，其范围在 0 到 9 之间。

将 CSV 数据记录转换为张量。之后，对像素值和原始类别数据进行归一化。

```
# 步骤 1
# 准备数据集对象
print("\n 加载 CSV 数据，转换为归一化张量数据 ")
# 加载包含 65 个值的 .csv 数据集
# 前 64 个值表示 64 个灰度格式的像素值
# 最后 1 个值表示实际数字（范围在 0 到 9 之间）
csv_data = "hand_written_digits.txt"
# 使用辅助函数 "tensor_converter" 将 CSV 格式转换为归一化张量
tensor_data = tensor_converter(csv_data)
```

### 步骤 2：构建自编码器网络

这里我们构建自编码器网络，该网络包含编码器和解码器架构。

*   编码器将原始数字像素值转换为更低维度的空间。（例如，它将 64 个灰度像素值转换为 8 个值。）
*   解码器从低维度重建原始数字。（例如，它使用 8 个值重建 64 像素的灰度图像。）

在此问题陈述中，编码和解码过程均使用全连接层。编码网络使用三个全连接（FC）层构建，其中第一个 FC 层将 65 个值（64+1）转换为 48 个，第二个层将 48 个值转换为 32 个。最后一层将 32 个值转换为 8 个。编码过程 ——> 65–48-32-8。

解码网络使用三个全连接层构建，其中第一个层将 8 个值转换为 32 个，第二个 FC 层将 32 个值转换为 48 个。最后一层将 48 个值转换为 65 个。解码过程 ——> 8-32-48-65。

```
def __init__(self):
    super(Autoencoder, self).__init__()
    self.fc1 = T.nn.Linear(65, 48)
    self.fc2 = T.nn.Linear(48, 32)
    self.fc3 = T.nn.Linear(32, 8)
    self.fc4 = T.nn.Linear(8, 32)
    self.fc5 = T.nn.Linear(32, 48)
    self.fc6 = T.nn.Linear(48, 65)

def encode(self, x):
    # 65-48-32-8
    z = T.tanh(self.fc1(x))
    z = T.tanh(self.fc2(z))
    z = T.tanh(self.fc3(z))
    return z

def decode(self, x):
    # 8-32-48-65
    z = T.tanh(self.fc4(x))
    z = T.tanh(self.fc5(z))
    z = T.sigmoid(self.fc6(z))
    return z
```

### 步骤 3：训练自编码器网络

使用超参数（如学习率、周期数、批次大小、损失度量和损失优化器）训练自编码器网络。训练时，一个辅助函数会接收自编码器网络、张量数据以及之前指定的所有其他超参数。

```
# 步骤 3. 训练自编码器模型
batch_size = 10
max_epochs = 200
log_interval = 8
learning_rate = 0.002
train(autoenc, tensor_data, batch_size, max_epochs, log_interval, learning_rate)
```



### 第 4 步：基于原始数据计算重建损失

通过比较原始手写数字与重建数字来评估训练好的模型。计算并存储图像重建损失：

```
# 设置自编码器为评估模式
autoenc.eval()
# 存储重建的 MSE 损失
MSE_list = make_err_list(autoenc, tensor_data)
# 根据 MSE 损失从高到低对列表进行排序
MSE_list.sort(key=lambda x: x[1], reverse=True)
```

### 第 5 步：基于误差度量分数选择最异常的数字

基于最高的 MSE 损失，我们需要找出数据集中属于异常的数字。

```
# 第 5 步：显示数据集中最异常的手写数字
print("数据集中基于最高 MSE 给出的异常数字：")
(idx, MSE) = MSE_list[0]
print(" 索引 : %4d , MSE : %0.4f" % (idx, MSE))
display_digit(tensor_data, idx)
```

### 输出

```
数据集中基于最高 MSE 给出的异常数字：
索引 :  486 , MSE : 0.1360
```

输出结果如图 7-5 所示。

![](img/520381_1_En_7_Fig5_HTML.jpg)

一张带有数值块的图表。Y 轴表示从 0 到 7 的数字，而 X 轴表示数值 0、2、4 和 6。数值为近似值。该数据集呈现上升趋势。

**图 7-5** 异常输出

```
digit =  7
```

## 总结

我们使用了 VGG 架构来确定样本图像数据集中的异常。我们逐步讲解了代码，并开发了一个端到端的流程。该模型只需极少的改动即可应用于工业级问题。

现在我们已经了解了异常检测，下一章将讨论最前沿的用例，即“图像超分辨率”。我们见过许多提升图像质量和分辨率的应用。你能通过在`PyTorch`中构建模型来自行实现吗？让我们在下一章中一探究竟。

