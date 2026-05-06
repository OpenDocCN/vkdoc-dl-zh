# 第 3 章 在线评论中的自然语言处理

## 构建神经网络

现在你已经将自变量和因变量调整为正确的形状，你将开始构建神经网络的过程。你的神经网络将是一个浅层网络，包含四层，随着网络的深入，层中的节点数量将减少。你将拥有 `500`、`200`、`100`、`50` 层。最后一层将有三个输入，并使用 softmax 激活函数，因为你正在解决一个包含三个类别的多类问题。请随意调整网络以获得更好的结果。由于你面临类别不平衡问题，你还需要为神经网络提供类别权重，以偏向于少数类别。请参见列表 3-42 至列表 3-45。

***列表 3-42.***

```
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
```

***列表 3-43.***

```
def get_nn_mod(list_layers, dp):
    model = Sequential()
    model.add(Dense(list_layers[0], input_dim=x_train4.shape[1],
                    activation='tanh', kernel_initializer='lecun_uniform'))
    model.add(Dropout(dp))
    for i in list_layers[1:]:
        model.add(Dense(i, input_dim=x_train4.shape[1], activation='tanh'))
        model.add(Dropout(dp))
```

