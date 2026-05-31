# 假设你有一个名为 `documents_vectorstore` 的向量存储

```
retriever = VectorstoreRetriever(documents_vectorstore)

query = "使用 RAG 的主要好处是什么？"

relevant_docs = retriever.get_relevant_documents(query)

for doc in relevant_docs:
    print(doc.page_content)
```

在此示例中，你通过传入向量存储创建了一个 `VectorstoreRetriever` 实例。然后，你定义了一个查询，并使用 `get_relevant_documents` 方法根据查询检索最相关的文档。最后，你遍历检索到的文档并打印其内容。

### 动手尝试

请记住，选择正确的检索器取决于你的具体用例和数据性质。请大胆尝试并探索不同的选项，以找到最适合你的方案。

