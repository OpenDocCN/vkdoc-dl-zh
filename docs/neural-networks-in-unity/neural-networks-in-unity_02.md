# sigmoid 函数

```
def nonlin(x,deriv=False):
if(deriv==True):
return x*(1-x)
return 1/(1+np.exp(-x))
```

#### 输入数据集

```
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
```

#### 输出数据集

```
Y = np.array([[0],[1],[1],[0]])
```

#### 设定随机数种子，使计算结果具有确定性（这是一个良好的实践）

```
np.random.seed(1)
```

#### 以均值为 0 随机初始化权重

```
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
```

