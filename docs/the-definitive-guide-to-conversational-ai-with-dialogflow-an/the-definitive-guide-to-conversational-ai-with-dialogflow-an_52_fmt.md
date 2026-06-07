
# 语言和关键词

您可以使用对话`语言`来查找特定语言的所有记录。例如，您可能想根据语言查找记录。也许您还想将其与某个`关键词`结合使用。

在 Dialogflow 中，您可以从 `detectIntentResponse` 的 `queryResult` 中检索 `languageCode`：

```javascript
queryResult.languageCode
```

您不需要单独存储关键词；您可以从用户话语中检索它们。在本节后面，我们将讨论主题挖掘。通过这种技术，您可以将关键词存储在单独的 BigQuery 列中。

每个语言指标应与会话 ID、用户话语及其他指标一起存储在数据仓库中。这样，您可以轻松运行一个 SQL 查询，该查询将返回聊天消息表中语言代码等于“NL”且您要查找的词是“fraud”的所有内容。

```sql
SELECT * FROM `chat_msg_table` WHERE LANGUAGE = "en-us" AND USER_UTTERANCE LIKE '%my keyword%'
```
