# 4. Python 与其他编程方法

Python 丰富的自然语言处理生态系统、活跃的社区支持、可读性、与数据科学和机器学习工具的集成、预训练模型的可用性、开源特性、广泛采用、多功能性以及跨平台兼容性，使其成为自然语言处理任务的理想选择。其流行度和强大的生态系统使 Python 成为自然语言处理研究和开发的事实标准语言。

## Python 中的面向对象编程

Python 中的面向对象编程是一种使用对象对复杂问题进行建模的编码方法。这种方法基于一个指导问题解决策略的理论框架。

在面向对象编程的背景下，这涉及运用一套原则和设计模式，通过对象的视角来应对挑战。在 Python 中，对象是将数据（称为属性）和函数（称为方法）封装到一个单一实体中。对象可以类比为现实世界中的有形实体。

在这种方案中，属性通常是名词，表示对象的属性；而方法是动作，用动词表示，定义了对象能做什么。

这种将数据存储和操作功能分开的方式构成了面向对象编程的核心。它涉及创建不仅包含信息，而且体现特定行为的对象。



### 为什么要在 Python 中使用面向对象编程？

我们在 Python 中使用面向对象编程（OOP），是为了将代码组织成可复用、模块化的单元，这些单元被称为对象，它们将数据与功能结合在一起。这种方法简化了复杂的软件开发，使代码更易于理解、维护和扩展。

**面向对象编程**

-   通过**继承**增强代码复用性。
-   通过**多态**实现灵活的代码交互。
-   通过将问题分解为可管理的对象来改进问题解决能力。
-   促进**数据封装**，确保代码管理安全且结构清晰。这种方法论与 Python 追求清晰、逻辑化代码的理念高度契合，适用于从小型到大型的各种项目。

**OOP 与结构化编程的对比**

-   **维护性：** OOP 通常比结构化编程更易于维护，后者维护起来可能更具挑战性。
-   **代码重复：** OOP 遵循“不要重复自己”（DRY）原则，最大限度地减少代码重复；而结构化编程可能涉及代码在多个位置重复。
-   **代码复用性：** 在 OOP 中，小而可复用的代码段很常见，这与结构化编程中大型代码块集中化的特点不同。
-   **编程模型：** OOP 采用基于对象的模型，而结构化编程依赖于顺序执行的代码块模型。
-   **调试：** 由于其模块化特性，OOP 中的调试通常更直接，而在结构化编程中可能更复杂。
-   **学习曲线：** OOP 的学习曲线较陡，而结构化编程通常被认为对初学者更友好。
-   **项目适用性：** OOP 更适合复杂的大型项目，而结构化编程更适用于简单的小型程序。

#### Python 中一切都是对象

这里有一个内行才知道的小知识：在你学习 Python 的整个过程中，你其实一直沉浸在面向对象编程（OOP）中，也许你自己都没有意识到。在 Python 中，无论你使用哪种编程范式，你几乎在每一个操作中都在与对象打交道。

这源于 Python 的核心原则：**Python 中的一切都被视为对象**。回想一下对象的构成：在 Python 中，一个对象封装了数据（称为属性）和功能（称为方法）。这个概念完美地适用于 Python 中的每一种数据类型。

以字符串为例；它本质上是一个数据集合（其中的字符）加上一组功能或行为（如 `upper()`、`lower()` 等方法）。这个原则同样适用于其他数据类型，如整数、浮点数、布尔值、列表和字典。

**示例：**

```
text = "Hello, Python!"
print(text.upper())  # 使用 upper() 方法
```

**输出：**

```
HELLO PYTHON
```

#### 属性和方法

**在继续之前，回顾一下我们所说的属性和方法会很有帮助：**

-   **属性**类似于对象内部的变量，用于保存与该对象相关的数据。
-   **方法**类似于函数，区别在于它们绑定在对象本身上，提供了可以对对象数据执行或操作的特定行为或动作。

### 你的第一个 Python 对象

让我们通过创建一个“Book”类，然后实例化它来创建一个“Book”对象，以此考虑一个简单的 Python 对象示例。这个示例将说明 Python 如何将一切视为对象，包括属性（数据）和方法（功能）。

#### 定义 'Book' 类

首先，我们定义一个名为“Book”的类，它就像创建“Book”对象的蓝图。这个类将拥有存储数据（如书名和作者）的属性，以及一个显示这些信息的方法。

```
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def display_info(self):
        print(f"Book Title: {self.title}\nAuthor: {self.author}")
```

**解释：**

-   `__init__` 是 Python 中的一个特殊方法，称为构造函数。它从类中初始化新对象，设置它们的初始状态。在这个例子中，它接收 `title` 和 `author` 作为参数，并将它们赋值给对象的属性（`self.title` 和 `self.author`）。每次我们实例化一个对象时，它都会被调用。
-   `display_info` 是一个定义好的方法，用于执行与对象相关的操作，即显示书的标题和作者。

#### 创建并使用一个 “Book” 对象

现在，让我们使用“Book”类蓝图创建一个实际的“Book”对象，然后使用它的方法。

```
# 创建一个 Book 对象
my_book = Book("The Catcher in the Rye", "J.D. Salinger")

# 使用 Book 对象的 display_info 方法
my_book.display_info()
```

**这将输出：**

```
Book Title: The Catcher in the Rye
Author: J.D. Salinger
```

`my_book` 实例是 `Book` 类的一个对象。它封装了数据（书名和作者属性），并通过一个方法（`display_info`）提供了功能。这个对象是 Python 中 OOP 原则的一个实际体现，展示了数据和相关行为是如何捆绑在一起的。每个 `Book` 对象可以保存不同的数据，展示了对象在以结构化方式管理状态和行为方面的强大能力。

### Python 中的面向对象编程（OOP）建立在四个基本概念之上

#### 抽象

抽象涉及隐藏系统的复杂现实，以简化用户（可能是最终用户或其他开发者）与它的交互。这类似于知道如何操作智能手机，却无需了解其内部过程的复杂细节，或者使用 Python 开发应用程序，却无需掌握其底层机制。在编程中，抽象允许将通用功能提炼到类中，从而使对象及其交互的管理更加直接。

**示例：**

```
from abc import ABC, abstractmethod

# 定义一个抽象基类 (ABC)
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

# 创建一个继承自 Shape 的具体类
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.1415 * self.radius ** 2

# 创建一个继承自 Shape 的具体类
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# 实例化对象并计算面积
circle = Circle(5)
rectangle = Rectangle(4, 6)

print("Circle Area:", circle.area())  # 输出: Circle Area: 78.53750000000001
print("Rectangle Area:", rectangle.area())  # 输出: Rectangle Area: 24
```

**输出：**

```
Circle Area: 78.53750000000001
Rectangle Area: 24
```



### 继承

继承是类从已有类中派生特征和行为的机制。这一原则支持“不要重复自己”（DRY）的理念，通过将共享属性和方法纳入基类（超类）来实现代码复用。这个概念类似于生物遗传，子类从父类（超类）继承特征和行为，展示共享的属性以及可能共享的方法。

**示例：**

```python
# 父类（超类）
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        pass  # speak 方法的占位符

# 从 Animal 继承的子类
class Dog(Animal):
    def speak(self):
        return f"{self.name} 说 汪汪！"

# 从 Animal 继承的子类
class Cat(Animal):
    def speak(self):
        return f"{self.name} 说 喵喵！"

# 创建子类的实例
dog = Dog("Buddy")
cat = Cat("Whiskers")

# 为每个实例调用 speak 方法
print(dog.speak())  # 输出：Buddy 说 汪汪！
print(cat.speak())  # 输出：Whiskers 说 喵喵！
```

**输出：**

```
Buddy 说 汪汪！
Whiskers 说 喵喵！
```

### 多态

多态允许在子类中定制最初在超类中定义的方法和属性，体现了“多种形态”的概念。它允许在不同类中创建同名但实现不同的方法。回顾我们现实生活中的类比，孩子（子类）可能从父母那里继承了一个共同的行为，比如感到饥饿（一个方法），但具体细节，比如饥饿的频率，可能有所不同，这就展示了多态性。

**示例：**

```python
# 父类（超类）
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        pass  # speak 方法的占位符

# 从 Animal 继承的子类
class Dog(Animal):
    def speak(self):
        return f"{self.name} 说 汪汪！"

# 从 Animal 继承的子类
class Cat(Animal):
    def speak(self):
        return f"{self.name} 说 喵喵！"

# 接受 Animal 对象并调用其 speak 方法的函数
def animal_sound(animal):
    return animal.speak()

# 创建子类的实例
dog = Dog("Buddy")
cat = Cat("Whiskers")

# 使用不同的对象调用 animal_sound 函数
print(animal_sound(dog))  # 输出：Buddy 说 汪汪！
print(animal_sound(cat))  # 输出：Whiskers 说 喵喵！
```

**输出：**

```
Buddy 说 汪汪！
Whiskers 说 喵喵！
```

### 封装

封装确保类内部数据的安全，保护数据的完整性和隐私。尽管 Python 没有通过语法显式支持私有属性，但它通过名称修饰以及使用 getter 和 setter 方法来实现对数据的受控访问和修改，从而达到封装的目的。

**示例：**

```python
class Human:
    def __init__(self, name, age):
        self.__name = name  # 私有属性
        self.__age = age    # 私有属性

    # name 的 getter 方法
    def get_name(self):
        return self.__name

    # age 的 setter 方法
    def set_age(self, age):
        if age > 0 and age < 150:  # 添加验证
            self.__age = age
        else:
            print("无效的年龄")

    # 显示信息的方法
    def display_info(self):
        print(f"姓名：{self.__name}，年龄：{self.__age}")

# 创建 Human 类的实例
person = Human("Alice", 30)

# 直接访问属性（不推荐）
# 这将引发错误：AttributeError: 'Human' object has no attribute '__name'
# print(person.__name)

# 使用 getter 方法访问属性
print("姓名：", person.get_name())  # 输出：姓名：Alice

# 使用 setter 方法设置年龄（带验证）
person.set_age(35)
person.display_info()  # 输出：姓名：Alice，年龄：35

# 设置无效年龄
person.set_age(200)  # 输出：无效的年龄
```

**输出：**

```
姓名：Alice
姓名：Alice，年龄：35
无效的年龄
```

## 模块与文件处理

### Python 模块

Python 模块本质上是一个包含内置函数、类和变量集合的文件，每个元素都有特定用途。Python 拥有众多模块，每个模块都针对处理不同的任务而定制。在本文中，我们将深入探讨 Python 模块的领域，探索诸如创建自定义模块、在 Python 中导入模块、为模块名称使用别名等主题。

### 理解 Python 模块

Python 模块是 Python 定义和语句的容器，包含函数、类、变量甚至可执行代码。将相关代码组织在一个模块中，有助于增强代码的可理解性、逻辑组织性和可重用性。

### 创建 Python 模块

要创建一个 Python 模块，你只需编写所需的代码，并将其保存为扩展名为 `.py` 的文件，命名为 `math_operations`。

**示例：**

```python
# math_operations.py
# 将两个数相加的函数
def add(x, y):
    return x + y

# 将两个数相减的函数
def subtract(x, y):
    return x - y
```

### 在 Python 中导入模块

Python 允许我们使用源文件中的 `import` 语句，将在一个模块中定义的函数和类导入到另一个模块中。当解释器遇到 `import` 语句时，如果该模块位于搜索路径中，它就会导入该模块。

> **注意：** 搜索路径  
> *搜索路径本质上是解释器在搜索模块时扫描的目录列表。*

如果你保留我们之前创建的文件，并创建一个新文件来调用它，你应该具有以下结构：

```
├── calc.py
└── math_operations.py
```

现在，要导入名为 `math_operations.py` 的模块，请在脚本开头包含以下命令：

**示例：**

```python
# 导入 math_operations 模块
import math_operations

# 使用模块中的函数
result_add = math_operations.add(5, 3)
result_subtract = math_operations.subtract(10, 4)

print("加法结果：", result_add)
print("减法结果：", result_subtract)
```

**输出：**

```
加法结果：8
减法结果：6
```

> **注意：** 点运算符  
> *这会导入模块本身，而不是其函数或类。要访问模块的内容，你需要使用点（`.`）运算符。*

#### 使用 `from` 语句进行 Python 导入

Python 的 `from` 语句允许从模块中选择性地导入特定属性，而无需导入整个模块。

**示例：**

```python
# 导入 math 模块，并专门导入 sqrt 函数
from math import sqrt

# 使用导入的函数计算平方根
number = 25
result = sqrt(number)

# 显示结果
print(f"{number} 的平方根是 {result}")
```

**输出：**

```
25 的平方根是 5.0
```

#### 从 Python 模块导入特定属性

**示例：**

```python
# 从内置的 "math" 库中导入特定属性
from math import pi, sqrt

# 使用导入的属性
print(f"圆周率 pi 的值约为：{pi}")
number = 25
result = sqrt(number)
print(f"{number} 的平方根是：{result}")
```

**输出：**

```
圆周率 pi 的值约为：3.141592653589793
25 的平方根是：5.0
```

#### 导入所有名称

星号（`*`）符号与 `import` 语句结合使用时，会将模块中的所有名称导入到当前命名空间。

**语法：**

```python
from module_name import *
```

使用 `*` 有其优点和缺点。如果你确切知道需要从模块中获取什么，不建议使用它。请谨慎使用。



### 定位 Python 模块

在 Python 中导入模块时，解释器会在多个位置进行搜索。首先，它会检查内置模块；如果未找到，则会探索 `sys.path` 变量中定义的目录列表。`sys.path` 变量包含一个目录列表，Python 会在此列表中搜索所需的模块。

**示例：**

```python
import sys
import importlib
# 向模块搜索路径添加自定义目录
custom_module_path = "/path/to/custom_module_directory"
sys.path.append(custom_module_path)
# 尝试从自定义目录导入模块
module_name = "custom_module"
try:
    custom_module = importlib.import_module(module_name)
    print(f"成功从 {custom_module.__file__} 导入 {module_name}")
except ImportError:
    print(f"导入 {module_name} 失败")
# 打印当前模块搜索路径
print("当前模块搜索路径：")
for path in sys.path:
    print(path)
```

**我的电脑上的输出：**

```
Failed to import custom_module
Current module search path:
C:\Users\didog\OneDrive\Desktop\test\test
C:\Users\didog\OneDrive\Desktop\test\test
C:\Program Files\JetBrains\PyCharm 2023.2\plugins\python\helpers\pycharm_display
C:\Users\didog\AppData\Local\Programs\Python\Python311\python311.zip
C:\Users\didog\AppData\Local\Programs\Python\Python311\DLLs
C:\Users\didog\AppData\Local\Programs\Python\Python311\Lib
C:\Users\didog\AppData\Local\Programs\Python\Python311
C:\Users\didog\OneDrive\Desktop\test\test\.venv
C:\Users\didog\OneDrive\Desktop\test\test\.venv\Lib\site-packages
C:\Program Files\JetBrains\PyCharm 2023.2\plugins\python\helpers\pycharm_matplotlib_backend
/path/to/custom_module_directory
```

### 重命名 Python 模块

在导入模块时，可以使用 `as` 关键字为其重命名。

**语法：**

```python
import 模块名 as 别名
```

### Python 内置模块

Python 提供了大量内置模块，这些模块提供了各种功能和特性。Python 中一些最流行的内置模块包括：

- **math：** 提供数学函数和常量，例如 `sqrt`、`sin`、`cos`、`pi` 等。
- **os：** 允许与操作系统交互，支持文件和目录操作、环境变量以及进程管理等任务。
- **sys：** 提供对 Python 解释器变量和函数的访问，例如命令行参数和系统特定设置。
- **datetime：** 简化日期和时间的处理，允许你创建、操作和格式化日期及时间间隔。
- **random：** 提供生成随机数和执行随机抽样的函数。
- **json：** 支持 JSON（JavaScript 对象表示法）数据的编码和解码，便于处理 JSON 文件和 Web API。
- **re：** 支持用于模式匹配和文本操作的正则表达式。
- **collections：** 提供专门的数据结构，如 `Counter`、`defaultdict` 和 `namedtuple`，用于更高级的数据处理。
- **urllib：** 提供处理 URL 和从网络资源获取数据的工具，常用于网络爬虫和网络请求。
- **csv：** 提供用于读写 CSV（逗号分隔值）文件的实用程序，常用于数据存储和交换。
- **sqlite3：** 允许与 SQLite 数据库交互，适用于轻量级数据库操作。
- **argparse：** 通过解析命令行参数和生成帮助信息，简化命令行界面（CLI）的创建。
- **os.path：** 提供操作文件路径以及检查文件和目录是否存在的方法。
- **logging：** 提供一个灵活且可配置的日志记录框架，用于调试和监控代码。
- **socket：** 提供网络通信能力，包括创建和操作用于网络连接的套接字。
- **multiprocessing：** 通过提供更高级别的进程管理接口，促进并行和并发编程。

这些只是 Python 中一些流行的内置模块。Python 庞大的标准库包含更多模块，每个模块都有特定的用途，使 Python 成为适用于广泛应用程序的多功能语言。

### Python 文件处理

Python 提供了强大的文件处理能力，使用户能够与文件进行交互，涵盖读取、写入以及一系列其他与文件相关的操作。虽然文件处理的概念存在于各种编程语言中，但 Python 以其直接而简洁的实现方式脱颖而出。在进行文件操作时，Python 会根据文件是文本格式还是二进制格式进行不同处理，这是一个需要牢记的关键区别。

在 Python 中，每一行代码都由一系列字符组成，实际上形成了一个文本文件。值得注意的是，文本文件中的每一行都以一个称为行尾（EOL）的特殊字符结束，该字符可以是逗号 `{,}` 或换行符。这个 EOL 字符标志着当前行的结束，并向解释器指示新的一行即将开始。让我们深入探讨在 Python 中读写文件的基本要点。



