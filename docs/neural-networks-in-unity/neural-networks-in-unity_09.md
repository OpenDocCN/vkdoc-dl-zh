# 第 5 章 Unity 中的数据可视化

```csharp
var header = Regex.Split(lines[0], SPLIT_RE);
//分割表头（第 0 行）

// 遍历各行
for (var i = 1; i < lines.Length; i++)
{
    var values = Regex.Split(lines[i], SPLIT_RE);
    //根据 SPLIT_RE 分割行，存储到变量中（通常是字符串数组）
    if (values.Length == 0 || values[0] == "") continue;
    // 如果值为 0，则跳过本次循环（continue）
```

在本节中，我们将声明字典对象并修剪 CSV 文件中的字符。

```csharp
    长度或第一个值为空
    var entry = new Dictionary<string, object>();
    // 创建字典对象

    // 遍历每个值
    for (var j = 0; j < header.Length && j < values.Length; j++)
    {
        string value = values[j]; // 设置局部变量 value
        value = value.TrimStart(TRIM_CHARS).TrimEnd(TRIM_CHARS).Replace("\\", "");
        // 修剪字符
        object finalvalue = value;
        //设置最终值
```

