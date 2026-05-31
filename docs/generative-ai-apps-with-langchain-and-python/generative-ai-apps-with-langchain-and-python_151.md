# 使用示例

`user_id = "1234"`

`recommendations = generate_recommendations(user_id)`

`print(recommendations)`

第 10 章项目：构建通用用例的智能体应用 在此示例中，我们有一个模拟的用户偏好数据集（`user_preferences_data`），它将用户 ID 映射到其喜爱的类型、演员和导演。`user_preference_tool` 用于检索给定用户 ID 的偏好，而 `recommendation_generator_tool` 则根据这些偏好生成个性化推荐。

智能体使用推荐工具和推荐提示进行初始化。`generate_recommendations` 函数接收一个用户 ID，检索其偏好，并使用智能体生成个性化推荐。

当你使用示例用户 ID 运行此代码时，它将根据用户喜爱的类型、演员和导演输出一组个性化电影推荐，如下所示：

根据你喜爱的类型（动作、科幻、喜剧），我们推荐：

1. 《盗梦空间》（动作、科幻）
2. 《黑客帝国》（动作、科幻）
3. 《银河护卫队》（动作、喜剧、科幻）

考虑到你喜爱的演员（汤姆·克鲁斯、布拉德·皮特、詹妮弗·劳伦斯），你可能会喜欢：

1. 《碟中谍 6：全面瓦解》（主演：汤姆·克鲁斯）
2. 《好莱坞往事》（主演：布拉德·皮特）
3. 《饥饿游戏》（主演：詹妮弗·劳伦斯）

鉴于你对克里斯托弗·诺兰和昆汀·塔伦蒂诺的欣赏，我们建议：

1. 《黑暗骑士三部曲》（导演：克里斯托弗·诺兰）
2. 《低俗小说》（导演：昆汀·塔伦蒂诺）
3. 《星际穿越》（导演：克里斯托弗·诺兰）

请记住，这只是一个用于说明概念的简化示例。在实际场景中，你需要将智能体与真实的用户数据、推荐算法和领域特定知识集成，以生成更准确和多样化的推荐。

## 第 10 章项目：构建通用用例的智能体应用 - 实时数据分析与决策

在此用例示例中，你利用流入系统的数据流来做出快速、明智的决策。无论是监控传感器读数、分析用户行为，还是优化资源分配，实时数据分析与决策都至关重要。

让我们深入了解构建由智能体驱动的实时数据分析与决策系统的分步过程：

1. **识别数据源：**
   - 确定实时数据的来源，例如传感器、日志文件、API 或数据库。
   - 确保你拥有必要的权限和访问权限，以便实时检索数据。

2. **设置数据摄取：**
   - 实现一种机制，将实时数据持续摄入到你的系统中。
   - 这可能涉及设置数据管道、流处理框架或事件驱动架构。
   - 考虑使用 Apache Kafka、Apache Flink 或 AWS Kinesis 等工具进行高效的数据摄取。

3. **预处理和转换数据：**
   - 执行必要的数据预处理和转换步骤，以确保数据质量和一致性。
   - 处理缺失值、异常值和数据格式问题。
   - 应用相关的特征工程技术，从原始数据中提取有意义的信息。

4. **设置智能体：**
   - 使用 LangChain 构建你的实时数据分析与决策系统。
   - 安装并导入必要的库：

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
```

5. **定义数据分析工具：**
   - 创建允许智能体访问和分析实时数据的工具：

```python
# 模拟数据检索和分析函数
def data_retrieval_tool():
    # 检索最新一批实时数据
    # 以适合分析的格式返回数据
    # 这是一个返回随机数据的模拟示例
    import random
    temperature = random.randint(20, 30)
    humidity = random.randint(40, 60)
    return f"温度：{temperature}°C，湿度：{humidity}%"

def data_analysis_tool(data):
    # 对提供的数据执行数据分析任务
    # 返回洞察、模式或异常
    # 这是一个检查高温和高湿度的模拟示例
    try:
        temperature = int(data.split("温度：")[1].split("°C")[0])
        humidity = int(data.split("湿度：")[1].split("%")[0])
        if temperature > 25 and humidity > 50:
            return "检测到高温和高湿度。可能需要进行调整。"
        else:
            return "温度和湿度在正常范围内。无需采取行动。"
    except (IndexError, ValueError):
        return "错误：无效的数据格式。无法分析。"

# 设置工具
tools = [
    Tool(
        name="数据分析工具",
        func=data_analysis_tool,
        description="对提供的数据执行数据分析任务。"
    )
]
```

6. **设置决策提示：**
   - 定义一个提示模板，指导智能体根据分析后的数据做出决策：

```python
decision_prompt = PromptTemplate(
    input_variables=["data_insights"],
    template="""
基于数据洞察：{data_insights}，
就应采取的行动做出决策。
提供清晰简洁的决策以及简要的理由说明。
"""
)
```

7. **初始化智能体：**

```python
decision_agent = initialize_agent(
    tools,
    OpenAI(temperature=0.7),
    agent="zero-shot-react-description",
    verbose=True
)
```

8. **处理实时数据并做出决策：**
   - 持续检索最新一批实时数据。
   - 使用智能体分析数据并根据洞察做出决策：

```python
# 处理实时数据并做出决策
def process_data_and_make_decision():
    data = data_retrieval_tool()
    decision = decision_agent.run(decision_prompt.format(data_insights=data))
    return decision

# 使用示例
while True:
    decision = process_data_and_make_decision()
    print("决策：", decision)
    # 根据决策执行操作
    input("按 Enter 键检索下一批数据...")
```

9. **将决策集成到你的应用程序中：**
   - 使用智能体做出的决策来触发应用程序中的相应操作或更新。
   - 这可能涉及发送警报、调整系统参数或触发自动化流程。
   - 监控决策的影响，并持续优化智能体的决策能力。

以下是一个完整的代码示例，演示了一个基本的实时数据分析与决策系统：

```bash
%pip install --upgrade --quiet langchain langchain-openai langchain_community langchain_openai python-dotenv
```

```python
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 从环境变量获取 OpenAI API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 导入新的聊天补全 API：
os.environ["OPENAI_API_KEY"] = "你的 OpenAI 密钥"

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 模拟数据检索和分析函数
def data_retrieval_tool():
    # 检索最新一批实时数据
    # 以适合分析的格式返回数据
    # 这是一个返回随机数据的模拟示例
    import random
    temperature = random.randint(20, 30)
    humidity = random.randint(40, 60)
    return f"温度：{temperature}°C，湿度：{humidity}%"

def data_analysis_tool(data):
    # 对提供的数据执行数据分析任务
    # 返回洞察、模式或异常
```



