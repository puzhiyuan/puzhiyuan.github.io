---
title: python_tips
top: false
cover: false
toc: true
mathjax: true
abbrlink: 1412
date: 2024-12-07 13:38:22
password:
summary: python 语言相关的知识点总结。
tags: 
categories: python
---

## Q：python 函数调用是值传递还是引用传递

### A：是一种基于 **对象引用** 的传递方式

Python 既不是严格的值传递，也不是严格的引用传递，而是一种基于 **对象引用** 的传递方式，称为 **“传对象引用”（pass-by-object-reference）** 或 **“传不可变对象值”（pass-by-value for immutable objects）**。这意味着：

1. **变量是引用对象的“名字”**：在 Python 中，变量存储的不是值本身，而是对象的引用（类似于指针）。变量可以看作是对内存中实际对象的一个标识符。
2. **可变对象与不可变对象**：
   - **不可变对象**（例如 `int`、`float`、`str`、`tuple`）：函数内部对参数的修改不会影响函数外部的变量，因为不可变对象在修改时会生成新的对象。
   - **可变对象**（例如 `list`、`dict`、`set`）：函数内部对参数的修改会影响函数外部的变量，因为可变对象在内存中是共享的。

>#### 不可变对象
>
>```python
>def modify_value(x):
>    print(f"Before modification: x = {x}, id(x) = {id(x)}")
>    x += 1  # 重新赋值，创建了新的对象
>    print(f"After modification: x = {x}, id(x) = {id(x)}")
>
>a = 10
>print(f"Before function call: a = {a}, id(a) = {id(a)}")
>modify_value(a)
>print(f"After function call: a = {a}, id(a) = {id(a)}")
>```
>
>**输出**：
>
>```bash
>Before function call: a = 10, id(a) = 140708947025136
>Before modification: x = 10, id(x) = 140708947025136
>After modification: x = 11, id(x) = 140708947025168
>After function call: a = 10, id(a) = 140708947025136
>```
>
>- `a` 的值没有改变，因为 `x` 指向了新的对象。
>
>#### 可变对象
>
>```python
>def modify_list(lst):
>    print(f"Before modification: lst = {lst}, id(lst) = {id(lst)}")
>    lst.append(4)  # 修改对象本身
>    print(f"After modification: lst = {lst}, id(lst) = {id(lst)}")
>
>my_list = [1, 2, 3]
>print(f"Before function call: my_list = {my_list}, id(my_list) = {id(my_list)}")
>modify_list(my_list)
>print(f"After function call: my_list = {my_list}, id(my_list) = {id(my_list)}")
>```
>
>**输出**：
>
>```bash
>Before function call: my_list = [1, 2, 3], id(my_list) = 140708947497344
>Before modification: lst = [1, 2, 3], id(lst) = 140708947497344
>After modification: lst = [1, 2, 3, 4], id(lst) = 140708947497344
>After function call: my_list = [1, 2, 3, 4], id(my_list) = 140708947497344
>```
>
>- `my_list` 的值被修改了，因为 `lst` 和 `my_list` 引用的是同一个对象。

#### 总结

- **不可变对象**：函数内部的修改不会影响外部变量。
- **可变对象**：函数内部的修改会影响外部变量。

你可以认为：

- 对于不可变对象，表现类似值传递。
- 对于可变对象，表现类似引用传递。
