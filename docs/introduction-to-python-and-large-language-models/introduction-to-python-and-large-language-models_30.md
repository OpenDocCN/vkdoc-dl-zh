# 字符串定界符与特性

在 Python 中，你可以使用单引号或双引号来定义字符串字面量。起始定界符与对应结束定界符之间包含的任何字符都被视为字符串的一部分。Python 对字符串的长度没有严格限制，只要机器内存资源允许，它可以包含任意数量的字符。此外，字符串甚至可以是空的。

*当需要将引号本身作为字符串的一部分包含进来时，该如何处理？* 正如你所见，直接的方法会遇到问题。在给定的示例中，字符串以单引号开头，导致 Python 将括号内的下一个单引号解释为结束定界符，无意中将其视为字符串的一部分。因此，最后一个单引号变成了多余的字符，导致语法错误：

```
>>> print('This is a single quote (') character.')
SyntaxError: invalid syntax
```

要在字符串中包含任何一种引号，一个简单有效的方法是使用相反类型的引号将字符串括起来。*如果你想包含单引号，就用双引号将字符串括起来；反之，如果需要包含双引号，就用单引号将字符串括起来。* 这种方法可以确保所需的引号被正确解释为字符串的一部分。

```
>>> print("This contains a single quote (') character.")
This string contains a single quote (') character.
>>> print('This contains a double quote (") character.')
This string contains a double quote (") character.
```

## 处理字符串中的特殊字符

在某些情况下，你可能需要 Python 以不同的方式解释字符串中的特定字符。

**这可以通过两种方式实现：**

- **抑制特殊解释：** 你可能希望阻止某些字符在字符串中具有其通常的特殊含义。
- **应用特殊解释：** 或者，你可能希望赋予通常按字面意义处理的字符以特殊含义。

为此，你可以使用反斜杠（`\`）字符。当反斜杠出现在字符串中时，它表示紧随其后的一个或多个字符应以特殊方式处理。这种机制被称为“转义序列”，因为反斜杠导致后续字符序列偏离其标准解释。

```
>>> print('String contains a single quote (') character.')
SyntaxError: invalid syntax
>>> print('String contains a single quote (\') character.')
This string contains a single quote (') character.
```

### 原始字符串

要表示原始字符串字面量，你可以使用前缀 `r` 或 `R`，这表示字符串中的转义序列将保持不变。这意味着反斜杠字符不会被解释为转义字符：

**原始字符串：**

```
print(r'foo\nbar')
foo\nbar
```

**原始字符串：**

```
print(R'foo\\bar')
foo\\bar
```

在原始字符串中，反斜杠不被视为转义字符，从而保留其字面表示。

## Python 中的三引号字符串

Python 提供了另一种定义字符串的方法，称为三引号字符串。这些字符串由三个连续的单引号（`'''`）或三个连续的双引号（`"""`）括起来。虽然转义序列在三引号字符串中仍然有效，但你可以包含单引号、双引号甚至换行符而无需转义。此功能简化了包含单引号和双引号的字符串的创建：

例如：

```
print('''This string has a single (') and a double (") quote.''')
This string has a single (') and a double (") quote.
```

在三引号字符串中，包含单引号和双引号不需要转义字符，使其成为此类场景的便捷选择。

