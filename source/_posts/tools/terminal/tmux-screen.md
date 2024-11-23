---
title: tmux&screen
top: false
cover: false
toc: true
mathjax: true
summary: 终端复用工具 tmux 和 screen，支持会话管理、窗口切换和分屏功能。tmux 提供更多的分屏与会话恢复功能，screen 适合简单的会话管理。
tags:
  - tmux
  - screen
categories:
  - tool
abbrlink: 46936
date: 2024-11-22 22:18:35
password:
---

## 1. 背景与简介

### 1.1 tmux

tmux 是一个终端复用工具，允许用户在单个终端会话中管理多个终端窗口（称为“面板”）。它支持会话恢复、分屏显示等功能，非常适合在远程服务器或需要长期保持任务运行的场景中使用。

### 1.2 screen

screen 是一个 GNU 项目下的终端复用工具，与 tmux 类似，可以在一个终端中运行多个程序，支持会话管理与恢复，是较早期的终端复用解决方案。



## 2. 安装

### 2.1 tmux 安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tmux

# CentOS/RHEL
sudo yum install tmux

# macOS
brew install tmux
```



### 2.2 screen 安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install screen

# CentOS/RHEL
sudo yum install screen

# macOS
brew install screen
```



## 3. 基本用法

|         功能         |                             tmux                             |                     Screen                      |
| :------------------: | :----------------------------------------------------------: | :---------------------------------------------: |
|       启动会话       |                            `tmux`                            |                    `screen`                     |
|     启动命名会话     |                  `tmux new -s session_name`                  |            `screen -S session_name`             |
|     查看会话列表     |                     `tmux list-sessions`                     |                  `screen -ls`                   |
|    切换到已有会话    |                `tmux attach -t session_name`                 |            `screen -r session_name`             |
|       杀掉会话       |             `tmux kill-session -t session_name`              |        `screen -X -S session_name quit`         |
|       新建窗口       |           `tmux new-window` 或者 `Ctrl+b` 后按 `c`           |                `Ctrl+a` 后按 `c`                |
|     在窗口间切换     |            `tmux select-window -t window_number`             | `Ctrl+a` 后按 `n`（下一窗口）或 `p`（上一窗口） |
|      重命名窗口      |    `tmux rename-window <new_name>` 或者 `Ctrl+b` 后按 `,`    |        `Ctrl+a` 后按 `A`，然后输入新名称        |
|     水平分割面板     |        `tmux split-window -h` 或者 `Ctrl+b` 后按 `%`         |                `Ctrl+a` 后按 `S`                |
|     垂直分割面板     |         `tmux split-window -v` 或者`Ctrl+b` 后按 `"`         |                `Ctrl+a` 后按 `|`                |
|       切换面板       | `tmux select-pane -t pane_number` 或者 `Ctrl+b` 后按 `o` 或方向键 |              `Ctrl+a` 后按 方向键               |
|     上下切换窗口     |            `Ctrl+b` 后按 `p` 和 `Ctrl+b` 后按 `n`            |     `Ctrl+a` 后按 `p` 和  `Ctrl+a` 后按 `n`     |
| 跳转到对应编号的窗口 |                     `Ctrl+b` 后按 `数字`                     |              `Ctrl+a` 后按 `数字`               |
|     快捷切换会话     |                      `Ctrl+b` 后按 `s`                       |                        /                        |
| 分离会话（保持运行） |                      `Ctrl+b` 后按 `d`                       |                `Ctrl+a` 后按 `d`                |
|       关闭面板       |                            `exit`                            |                     `exit`                      |
|   保存当前会话布局   |              `tmux save-buffer -b buffer_name`               |         不直接支持，需手动记录配置文件          |



## 4. tmux 和 screen 对比

| 特性         | tmux                     | screen                   |
| ------------ | ------------------------ | ------------------------ |
| **会话管理** | 支持命名会话，命令直观   | 支持命名会话，命令较复杂 |
| **窗口操作** | 多窗口，支持分屏         | 多窗口，不支持直接分屏   |
| **配置文件** | 配置简单，功能强大       | 配置功能有限             |
| **会话恢复** | 强大，会话可保存当前状态 | 支持恢复，但功能稍弱     |
| **社区支持** | 活跃，插件生态丰富       | 较少，开发较停滞         |
| **资源占用** | 较低                     | 较高                     |

------



## 5. 使用场景与案例

### 5.1 tmux 使用案例

#### 长期运行脚本

1. 启动 tmux 会话：

   ```bash
   tmux new -s my_script
   ```

2. 在会话中运行脚本：

   ```bash
   python long_running_script.py
   ```

3. 按 `Ctrl+b` 后按 `d` 退出，任务继续运行

4. 重新连接：

   ```bash
   tmux attach -t my_script
   ```

#### 分屏查看日志和调试

1. 启动 tmux 并分屏：

   ```bash
   tmux
   tmux split-window -h
   ```

2. 左侧运行服务：

   ```bash
   python app.py
   ```

3. 右侧查看日志：

   ```bash
   tail -f logs/app.log
   ```



### 5.2 screen 使用案例

#### 会话恢复

1. 启动会话并运行程序：

   ```bash
   screen -S session_name
   ./run_my_program.sh
   ```

2. 按 `Ctrl+a` 后按 `d` 退出

3. 重新连接会话：

   ```bash
   screen -r session_name
   ```

#### 日志监控

1. 启动会话并重命名窗口：

   ```bash
   screen -S log_session
   ```

   按 `Ctrl+a` 后按 `A` ，输入 `logs_window`

2. 运行日志监控命令：

   ```bash
   tail -f /var/log/syslog
   ```





## 6. 配置优化

### 6.1 tmux 配置文件

在 `~/.tmux.conf` 中添加：

```bash
set -g mouse on  # 开启鼠标支持
bind r source-file ~/.tmux.conf \; display "Config reloaded!"  # 重新加载配置
set -g prefix C-a  # 修改前缀键
```



### 6.2 screen 配置文件

在 `~/.screenrc` 中添加：

```bash
startup_message off  # 关闭启动消息
defscrollback 10000  # 增加滚动缓冲区
bindkey -k k3 prev   # 绑定快捷键切换窗口
```
