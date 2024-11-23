---
title: Linux 环境变量
top: false
cover: false
toc: true
mathjax: true
summary: linux 系统的环境变量相关内容
tags:
  - env
categories:
  - Linux
abbrlink: 57af82ca
date: 2024-06-23 19:57:58
password:
---

## 环境变量

在 Linux 系统中，可以设置环境变量的文件有多种，具体取决于你想要设置的范围和影响的用户数量。以下是一些常用的环境变量设置文件及其用途：

1. **用户级环境变量设置文件**：
   - `~/.bashrc`：这是 Bash Shell 的用户配置文件。在用户登录时，Bash 会读取这个文件。你可以在这个文件中设置用户特定的环境变量。
   - `~/.bash_profile` 或 `~/.profile`：这是另一个用户级的 Shell 配置文件。在用户登录时，Bash 会读取这些文件之一（通常只会读取其中一个）。你可以在这些文件中设置用户特定的环境变量。
   - `~/.zshrc`：如果你使用 Zsh Shell，这个文件是 Zsh 的用户配置文件。在用户登录时，Zsh 会读取这个文件。你可以在这个文件中设置用户特定的环境变量。
2. **系统级环境变量设置文件**：
   - `/etc/profile`：这个文件是系统级的 Shell 配置文件。在用户登录时，系统会读取这个文件。你可以在这个文件中设置系统范围的环境变量。
   - `/etc/environment`：这是另一个系统级的环境变量设置文件。你可以在这个文件中定义系统范围的环境变量。这些变量会在系统启动时被加载，并对所有用户生效。
   - `/etc/bash.bashrc`：这是系统级的 Bash Shell 配置文件。在用户登录时，Bash 会读取这个文件。你可以在这个文件中设置系统范围的环境变量。
3. **其他 Shell 配置文件**： 
   - 如果你使用其他的 Shell（如 Zsh、Fish、Csh 等），它们也有相应的系统级配置文件（如 `/etc/zshrc` 或 `/etc/fish/config.fish`），可以用于设置环境变量。
