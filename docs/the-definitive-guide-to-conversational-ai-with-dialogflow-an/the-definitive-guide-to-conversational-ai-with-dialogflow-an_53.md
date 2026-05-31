# 平台

您可以使用`平台`设置来查找特定平台的所有记录。例如，您可能想查找 Google Assistant 上的最新对话。

您需要根据所使用的实现自行设置平台指标。某些实现可能有自己的设置，例如 Google Assistant 具有针对每个设备表面的配置：Surface Capabilities。

当您使用集成了聊天的自定义 Web 界面时，您需要自行设置平台名称。

```javascript
const platform = 'web';
```

每个平台指标应与会话 ID 及其他指标一起存储到 BigQuery 中。这样，您可以轻松运行一个 SQL 查询，该查询将返回聊天消息表中平台等于“web”的所有内容：

```sql
SELECT * FROM `chat_msg_table` WHERE PLATFORM = "web"
```

