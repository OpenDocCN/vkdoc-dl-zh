
# 情感分数

您可以使用情感分数来根据特定情感查找所有记录。例如，您可能对本周最“负面”的聊天记录感兴趣，以了解客户不满的原因。

像 Dialogflow 这样的工具可以为某些语言启用情感分析。但是，当您想对“荷兰语”等不支持的语言使用情感分析时，您必须首先翻译用户的话语。

您可以使用以下工具来实现这一点：

- Cloud Translate
- Cloud Natural Language

```javascript
const sentiment_score = queryTextSentiment.score;
const sentiment_magnitude = queryTextSentiment.magnitude;
```

文档情感的`分数`表示文档的整体情绪。文档情感的`幅度`表示文档中存在的情感内容量，该值通常与文档长度成正比。

每个情感分数应与会话 ID 及其他指标一起存储在数据仓库中。这样，您可以轻松运行一个 SQL 查询，该查询将返回聊天消息表中情感分数为负（低于 0）的所有内容，并按情感分数升序排列，从最差到最好：

```sql
SELECT * FROM `chat_msg_table` WHERE SENTIMENT_SCORE < 0 ORDER BY SENTIMENT_SCORE ASC
```

## 提示

如何处理讽刺？事实上，讽刺有时连人类都难以理解。不过，我们在 Google Cloud 中确实有工具可以使用自定义情感来构建模型。考虑一下这样的用户话语：“耶！《超越善恶》游戏又延期了！干得漂亮！”Google Cloud 的内置情感分析可能会因为“耶！”和“干得漂亮！”这些词而检测到这是非常积极的。甚至可能给出 90%（0.9）的情感分数。

然而，我们希望将其归类为非常负面，因为这里讨论的是游戏延期，这并不有趣，所以我们期望得到一个负面的情感分数。了解您可以使用 Google Cloud 中的 AutoML Natural Language 训练自己的情感模型是很有用的。它有一个功能：情感分析。该模型检查文档并识别其中占主导地位的情感观点，特别是确定作者的态度是积极、消极还是中性。
