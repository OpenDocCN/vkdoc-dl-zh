# 根据用户偏好生成个性化推荐

`recommendations = []`

`if user_preferences:`
    `favorite_genres = user_preferences.get("favorite_genres", [])`
    `favorite_actors = user_preferences.get("favorite_actors", [])`
    `favorite_directors = user_preferences.get("favorite_directors", [])`

    `if favorite_genres:`
        `recommendations.append(f"根据您喜欢的类型 ({', '.join(favorite_genres)})，我们推荐：")`
        `recommendations.append("1. 盗梦空间 (动作, 科幻)")`
        `recommendations.append("2. 黑客帝国 (动作, 科幻)")`
        `recommendations.append("3. 银河护卫队 (动作, 喜剧, 科幻)")`

    `if favorite_actors:`
        `recommendations.append(f"考虑到您喜欢的演员 ({', '.join(favorite_actors)})，您可能会喜欢：")`
        `recommendations.append("1. 碟中谍 6：全面瓦解 (汤姆·克鲁斯主演)")`
        `recommendations.append("2. 好莱坞往事 (布拉德·皮特主演)")`
        `recommendations.append("3. 饥饿游戏 (詹妮弗·劳伦斯主演)")`

    `if favorite_directors:`
        `recommendations.append(f"鉴于您对{'和'.join(favorite_directors)}的欣赏，我们建议：")`
        `recommendations.append("1. 黑暗骑士三部曲 (克里斯托弗·诺兰执导)")`
        `recommendations.append("2. 低俗小说 (昆汀·塔伦蒂诺执导)")`
        `recommendations.append("3. 星际穿越 (克里斯托弗·诺兰执导)")`

`if not recommendations:`
    `recommendations.append("哎呀！我们无法根据您的偏好找到个性化推荐。")`
    `recommendations.append("请提供更多关于您喜欢的类型、演员或导演的信息。")`

`return "\n".join(recommendations)`

## 第 10 章 项目：为常见用例构建智能体应用

### 5. 设置工具

`tools = [`
    `Tool(`
        `name="用户偏好工具",`
        `func=user_preference_tool,`
        `description="根据用户 ID 检索用户偏好。"`
    `),`
    `Tool(`
        `name="推荐生成工具",`
        `func=recommendation_generator_tool,`
        `description="根据用户偏好生成个性化推荐。"`
    `)`
`]`

### 6. 设置推荐提示

定义一个提示模板，指导智能体生成个性化推荐：

`recommendation_prompt = PromptTemplate(`
    `input_variables=["user_id"],`
    `template="""`
    `给定用户 ID {user_id}，检索其偏好并生成个性化推荐。`
    `提供一份最佳推荐列表，并为每条推荐附上简要说明。`
    `"""`
`)`

### 7. 初始化智能体

`recommendation_agent = initialize_agent(`
    `tools,`
    `OpenAI(temperature=0.7),`
    `agent="zero-shot-react-description",`
    `verbose=True`
`)`

### 8. 生成个性化推荐

使用智能体为特定用户生成个性化推荐：

```
# 生成个性化推荐
def generate_recommendations(user_id):
    user_preferences = user_preference_tool(user_id)
    recommendations = recommendation_generator_tool(user_preferences)
    return recommendations

# 使用示例
user_id = "1234"
recommendations = generate_recommendations(user_id)
print(recommendations)
```

### 9. 以下是使用示例

`user_id = "1234"`
`recommendations = generate_recommendations(user_id)`
`print(recommendations)`

### 10. 将推荐集成到您的应用程序中

-   以视觉上吸引人且直观的方式向用户展示个性化推荐。
-   考虑推荐的放置位置、时机和格式等因素，以最大化其影响力。
-   监控用户参与度并收集反馈，以持续改进推荐系统。

以下是一个完整的代码示例，演示了基本的个性化推荐系统：

```
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 推荐工具函数
def user_preference_tool(user_id):
    return user_preferences_data.get(user_id, {})

def recommendation_generator_tool(user_preferences):
    # 根据用户偏好生成个性化推荐
    recommendations = []
    if user_preferences:
        favorite_genres = user_preferences.get("favorite_genres", [])
        favorite_actors = user_preferences.get("favorite_actors", [])
        favorite_directors = user_preferences.get("favorite_directors", [])

        if favorite_genres:
            recommendations.append(f"根据您喜欢的类型 ({', '.join(favorite_genres)})，我们推荐：")
            recommendations.append("1. 盗梦空间 (动作, 科幻)")
            recommendations.append("2. 黑客帝国 (动作, 科幻)")
            recommendations.append("3. 银河护卫队 (动作, 喜剧, 科幻)")

        if favorite_actors:
            recommendations.append(f"考虑到您喜欢的演员 ({', '.join(favorite_actors)})，您可能会喜欢：")
            recommendations.append("1. 碟中谍 6：全面瓦解 (汤姆·克鲁斯主演)")
            recommendations.append("2. 好莱坞往事 (布拉德·皮特主演)")
            recommendations.append("3. 饥饿游戏 (詹妮弗·劳伦斯主演)")

        if favorite_directors:
            recommendations.append(f"鉴于您对{'和'.join(favorite_directors)}的欣赏，我们建议：")
            recommendations.append("1. 黑暗骑士三部曲 (克里斯托弗·诺兰执导)")
            recommendations.append("2. 低俗小说 (昆汀·塔伦蒂诺执导)")
            recommendations.append("3. 星际穿越 (克里斯托弗·诺兰执导)")

    if not recommendations:
        recommendations.append("哎呀！我们无法根据您的偏好找到个性化推荐。")
        recommendations.append("请提供更多关于您喜欢的类型、演员或导演的信息。")

    return "\n".join(recommendations)

# 设置工具
tools = [
    Tool(
        name="用户偏好工具",
        func=user_preference_tool,
        description="根据用户 ID 检索用户偏好。"
    ),
    Tool(
        name="推荐生成工具",
        func=recommendation_generator_tool,
        description="根据用户偏好生成个性化推荐。"
    )
]

# 设置推荐提示
recommendation_prompt = PromptTemplate(
    input_variables=["user_id"],
    template="""
    给定用户 ID {user_id}，检索其偏好并生成个性化推荐。
    提供一份最佳推荐列表，并为每条推荐附上简要说明。
    """
)

### 初始化智能体
recommendation_agent = initialize_agent(
    tools,
    OpenAI(temperature=0.7),
    agent="zero-shot-react-description",
    verbose=True
)

# 生成个性化推荐
def generate_recommendations(user_id):
    user_preferences = user_preference_tool(user_id)
    recommendations = recommendation_generator_tool(user_preferences)
    return recommendations
```



