
# 假正例——被误解的请求

**假正例**是指聊天机器人匹配了错误意图的结果。它本应匹配另一个意图，或者如果该意图不存在，则应匹配回退意图，但由于它误解了用户话语，导致结果错误。当你的机器人无法确定正确的用户意图时，就会发生被误解的请求错误。这种错误可能是由于机器人未考虑问题的上下文，或者回答了与所提问题略有不同的问题所致。

例如，当你的虚拟助手有一个名为`Block credit card`的意图，并且它使用训练短语`I want to block my credit card.`进行训练。你的虚拟助手还有另一个名为`Renew credit card`的意图，并使用训练短语`I want to renew my credit card.`进行训练。我可以针对上述场景创建测试数据，用户话语为：`My account is blocked, can I get a new credit card?` 我期望它匹配`Renew credit card`意图；然而，Dialogflow 检测到了`Block credit card`意图。因此，这是一个假正例。虽然匹配了，但匹配的是错误的意图！

你可以将类似上述的测试用例存储在 BigQuery 中，这样每次通过 Dialogflow API 对智能助手进行更改时，都可以重新运行该场景。你需要收集的数据字段包括`用户话语`、`预期意图名称`、`检测到的意图名称`以及`结果`，即`FP`。
