# Lamini 的应用场景与用例

`Lamini` 功能多样，支持广泛的应用场景和用例，使企业能够：

-   **部署用于客户服务的 AI 驱动聊天机器人**，这些机器人能够理解并操作公司特定的术语和流程。
-   **使用 AI 工具生成和优化内容**，使其符合品牌指南和风格。
-   **简化编码和调试工作流程**，通过擅长处理公司独特代码库和编程方法的 AI 模型来实现。

**示例应用：**

要使用 `Lamini`，首先需要在你的笔记本或终端中安装并升级它，并在其网站上注册以获取 API 密钥。使用以下命令安装和升级 `Lamini`：

```
pip install lamini
pip install --upgrade lamini
```

**然后使用以下代码：**

```
import lamini
lamini.api_key = "YOUR-API-KEY"
from lamini import LaminiClassifier
llm = LaminiClassifier()
prompts={
"cat": "Cats are generally more independent and aloof than dogs, who are often more social and affectionate.",
"dog": "Dogs are more pack-oriented and tend to be more loyal to their human family.",
}
llm.add_data_to_class("dog", ["woof", "group oriented"])
llm.add_data_to_class("cat", ["Oh, I prefer to do stuff on my own than dogs", "meow"])  # list of examples is valid too
llm.prompt_train(prompts)
llm.predict(["I'm more independent than dogs", "woof"])
```

**示例输出：**

```
0%|          | 0/10 [00:34<?, ?it/s]
ERROR:lamini.classify.llama_classifier:Failed to generate examples for class cat
ERROR:lamini.classify.llama_classifier:string indices must be integers
0%|          | 0/10 [00:00<?, ?it/s]ERROR:lamini.classify.llama_classifier:string indices must be integers
ERROR:lamini.classify.llama_classifier:Consider rerunning the generation task if the error is transient, e.g. 500
0%|          | 0/10 [00:33<?, ?it/s]
ERROR:lamini.classify.llama_classifier:Failed to generate examples for class dog
ERROR:lamini.classify.llama_classifier:string indices must be integers
ERROR:lamini.classify.llama_classifier:string indices must be integers
ERROR:lamini.classify.llama_classifier:Consider rerunning the generation task if the error is transient, e.g. 500
100%|██████████| 2/2 [00:01<00:00,  1.62it/s]
['cat', 'dog']
```

**解释**

上述代码利用 `Lamini` 库进行文本分类，具体是将文本分类到不同的类别中。以下是代码各部分作用的详细说明：

1.  **安装 Lamini**

前两行（`"!pip install lamini"` 和 `"!pip install --upgrade lamini"`）使用 Python 的包管理系统 `pip` 来安装 `Lamini` 库并确保其为最新版本。

2.  **导入 LaminiClassifier**

`from lamini import LaminiClassifier` 从 `lamini` 模块中导入 `LaminiClassifier` 类，该类是 `Lamini` 库的一部分。

3.  **创建 LaminiClassifier 实例**

`llm = LaminiClassifier()` 创建了 `LaminiClassifier` 类的一个实例，该实例将用于文本分类任务。

4.  **定义提示词**
    -   `prompts` 是一个字典，包含与不同类别关联的示例文本。字典中的每个键值对代表一个类别标签及其对应的示例文本。
    -   在本例中，有两个类别：`"cat"` 和 `"dog"`，示例文本描述了每个类别相关的特征或行为。

5.  **向类别添加数据**
    -   `llm.add_data_to_class()` 方法用于向每个类别添加训练数据。它将提供的文本示例与相应的类别标签关联起来。
    -   例如，`llm.add_data_to_class("dog", ["woof", "group oriented"])` 将提供的文本 `"woof"` 和 `"group oriented"` 添加到 `"dog"` 类别。
    -   类似地，`llm.add_data_to_class("cat", ["Oh, I prefer to do stuff on my own than dogs", "meow"])` 将提供的文本添加到 `"cat"` 类别。

6.  **使用提示词训练模型**

`llm.prompt_train(prompts)` 使用提供的提示词和关联的示例文本训练 `Lamini` 分类器。此步骤涉及训练模型识别文本数据中的模式，并学习根据提供的示例将新的文本输入分类到适当的类别中。

7.  **进行预测**
    -   `llm.predict(["I'm more independent than dogs", "woof"])` 对提供的文本输入进行预测。
    -   在本例中，模型根据使用提示词和示例文本进行的训练，为给定的文本 `"I'm more independent than dogs"` 和 `"woof"` 预测类别标签。

**注意**

环境变量

*出于安全考虑，建议将你的 API 密钥添加为环境变量。*



