---
title: python_package_install
top: false
cover: false
toc: true
mathjax: true
date: 2024-09-19 20:27:09
password:
summary: 总结 Python 包的多种安装方法，包括 pip、setup.py 和 conda，适用于不同开发和环境管理场景。
tags: python
categories: language
---


# Python 安装包方法总结

## 1. 使用 `pip` 安装包

`pip` 是 Python 官方推荐的包管理工具，主要用于从 Python 包索引（PyPI）中安装、管理和升级包。

### 1.1 安装基本命令

```bash
# pip install 包名
pip install requests
```

### 1.2 安装指定版本的包

```bash
# pip install 包名==版本号
pip install numpy==1.21.0
```


### 1.3 从 `requirements.txt` 文件安装依赖

当一个项目依赖多个包时，可以将依赖包记录在 `requirements.txt` 文件中，并使用以下命令一次性安装所有依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件的格式通常如下：
```
requests==2.25.1
numpy==1.21.0
```



## 2. 使用 `python setup.py` 安装包

开发自己的 Python 包时，通常需要创建一个 `setup.py` 文件。这个文件包含包的配置信息，如包的名称、版本、依赖库等。通过 `setup.py`，可以将项目安装到当前的 Python 环境中。

### 2.1 标准安装

要将项目安装到 Python 环境中，可以使用以下命令：
```bash
python setup.py install
```

将项目的文件复制到 Python 环境中的 `site-packages` 目录，安装完成后，包就可以在任何地方通过 `import` 使用。

### 2.2 开发模式安装 (`python setup.py develop`)

开发模式下，包不会被复制到 `site-packages` 目录中，而是以**软链接**的方式安装。这样，你可以在不重新安装包的情况下，对代码进行修改，并且修改会**实时生效**。

```bash
python setup.py develop
```

在进行频繁的代码修改和测试时，省去了每次修改后重新安装包的步骤。



## 3. 使用 `pip install -e .` 进行开发模式安装

`pip install -e .` 是 Python 包开发过程中常用的一种开发模式安装方法。

- `-e` 参数代表 **editable**，即**可编辑模式**。当你使用这个参数时，安装的包不会被复制到 Python 环境的 `site-packages` 目录中，而是创建一个指向源代码目录的软链接。这意味着，当你修改源代码时，Python 环境中的安装包会自动更新，无需重新安装。

- 在命令 `pip install -e .` 中，`.` 代表当前目录，意思是将当前目录下的 Python 项目安装到环境中。一般这个目录包含一个 `setup.py` 文件，它定义了该项目的元数据和安装步骤。

例如，如果你正在开发一个名为 `my_project` 的项目，并且你的项目文件结构如下：

```
my_project/
├── setup.py
├── my_module/
│   └── __init__.py
└── tests/
    └── test_my_module.py
```

你可以在 `my_project` 目录下运行：
```bash
pip install -e .
```

运行 `pip install -e .` 后，以下几件事会发生：

1. **软链接创建**：`pip` 不会将你的代码复制到 Python 环境的 `site-packages` 中，而是会在 `site-packages` 目录中创建一个指向你项目目录的软链接。
   
2. **实时生效**：由于包被软链接到项目的源代码目录，任何对源代码的修改都**实时生效**，不需要重新执行安装命令。可以边修改代码边进行测试，不必每次修改后重新安装包。

3. **依赖安装**：如果项目的 `setup.py` 文件中定义了依赖包，`pip` 会自动为你安装这些依赖，确保项目所需的所有包都已安装。



## 4. 使用 `conda` 安装包

使用 Anaconda 或 Miniconda 作为 Python 环境管理工具，可以使用 `conda` 安装包：

```bash
# conda install 包名
conda install numpy
```



## 总结

- `pip install`：用于从 PyPI 安装包。
- `python setup.py install`：适用于自定义包的标准安装。
- `python setup.py develop`：开发模式安装，支持动态修改代码。
- `pip install -e .`：适合在本地开发模式下使用，实时反映代码修改。
- `conda install`：用于 Conda 环境下的包管理。
