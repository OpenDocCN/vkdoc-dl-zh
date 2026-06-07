
# 会话 ID

`会话 ID` 对于查找特定会话的所有记录、阅读完整对话记录以及计算独立用户总数非常有用。

像 Dialogflow 这样的聊天机器人构建工具会为每个聊天会话维护一个会话路径。此聊天会话路径绑定到您的代理账户（因此其他人无法窥探），并且由于包含 uuid 而具有唯一性。

```javascript
const sessionId = uuid.v4();
const sessionClient = new df.SessionsClient();
const sessionPath = sessionClient.sessionPath(projectId, sessionId);
```

当您使用 BigQuery 时，应收集每个传入的用户话语及其会话 ID。一旦您知道某个特定的会话 ID，就可以检索包含所有其他已存储数据字段的完整聊天记录。用于检索会话的 SQL 查询如下所示：

```sql
SELECT * FROM `chat_msg_table` WHERE SESSION_ID = 'projects/myagent/agent/sessions/db33b345-663c-4867-8021-fecd50c5e8b1' ORDER BY DATETIME
```

从聊天消息表中返回会话 ID 等于某个字符串的所有内容，并按日期和时间排序，因此会话的第一条消息将首先显示。
