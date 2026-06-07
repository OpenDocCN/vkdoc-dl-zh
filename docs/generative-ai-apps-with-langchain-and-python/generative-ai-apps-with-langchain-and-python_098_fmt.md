# 运行条件链

```
query = "I love your product!"
result = conditional_chain.run(query)
print(result)
```

在此示例中，您有两个链：`positive_chain` 和 `negative_chain`。您创建了一个 `ConditionalChain`，它会根据输入查询的情感执行相应的链。如果情感为正面，则执行 `positive_chain`；如果情感为负面，则执行 `negative_chain`；如果情感为中性或无法确定，则回退到默认的 `positive_chain`。最后，您使用一个查询运行条件链并打印响应。

## 医疗案例研究：使用条件链处理患者咨询

一家医疗机构实施了条件链，以简化患者沟通并有效分流紧急情况。

**实施方式：** 条件链被用于根据描述的症状分析患者咨询。非紧急查询自动转至预约安排或一般建议，而潜在紧急情况则立即触发对医务人员的警报。

**成果：** 该系统将关键病例的响应时间提升了 40%，确保在需要时能快速获得医疗关注，同时高效处理常规咨询，减轻了整体行政负担。