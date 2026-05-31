# 日期/时间戳

您需要一个`日期/时间戳`来根据特定时间查找所有记录，并计算完整的会话时长。

在将对象存储到 BigQuery 之前，您可以设置自己的`日期/时间戳`。确保其格式符合数据仓库的预期格式：

```javascript
const timestamp = new Date().getTime()/1000;
```

每个日期和时间戳应与会话 ID 及其他指标一起存储到 BigQuery 中。这样，您可以轻松运行一个 SQL 查询，该查询将返回聊天消息表中日期时间在 8 月 1 日到 8 月 10 日之间的所有内容，并按日期时间降序排列，使最新的记录排在最前面：

```sql
SELECT * FROM `chat_msg_table` WHERE DATETIME > '2020-08-01 10:00:00' AND POSTED < '2020-08-10 00:00:00' ORDER BY DATETIME DESC
```

