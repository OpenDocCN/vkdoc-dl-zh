## 第 5 章 虚拟助手中的自然语言处理

**图 5-26.** 清单 5-70 中展示了分词后的单词。从这里开始，模型只需 3 行代码。

我们将 `bert` 作为参数传递给 `text_classifier`，以便对 BERT 模型进行微调。`preproc` 参数会自动生成 BERT 所需的格式。我们使用学习率 `2e-5`，因为这是推荐的学习率。参见清单 5-71 和图 5-27。

**清单 5-71.**

```python
model = text.text_classifier('bert', train_data=(x_train, y_train),
                             preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train),
                             batch_size=6)
hist = learner.fit_onecycle(2e-5, 2)
```

**图 5-27.** 如您所见，训练准确率达到 99.6%。在清单 5-72 中，我们在一个示例句子上执行模型。