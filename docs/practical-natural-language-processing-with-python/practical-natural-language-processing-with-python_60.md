# 第 5 章 虚拟助手中的自然语言处理

## 编码器输出

首先，准备一个层来获取编码器状态。这些状态是解码器的初始状态。

## 解码器输入

模型运行时，解码器输入是未知的。因此，你必须利用架构一次预测一个词，并用该词来预测下一个词。

模型的解码器部分接收解码器（时间延迟输入）。对于第一个词，解码器模型使用编码器状态进行初始化。`decoder_lstm` 层通过解码器输入和初始化的编码器状态被调用。该层提供两组输出：单元的输出集（`decoder_output`）以及 `cell_state` 和 `hidden_state` 的最终输出。`decoder_output` 被传递到密集层，以获取机器人文本的最终预测。解码器状态的输出用于更新下一次运行（下一个词预测）的状态。`bot_text` 被追加到解码器输入中，并重复相同的过程。参见清单 5-49。

***清单 5-49.***

```
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(100,))
decoder_state_input_c = Input(shape=(100,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
```

现在，你必须为模型设置训练和推理所需的层。在清单 5-50 中，你将训练模型。

