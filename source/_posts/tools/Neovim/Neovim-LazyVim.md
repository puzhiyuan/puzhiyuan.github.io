---
title: Neovim&LazyVim
top: false
cover: false
toc: true
mathjax: true
summary: '终端编辑器Neovim,以及LazyVim插件安装使用'
tags:
  - Neovim
categories:
  - tool
abbrlink: b188b029
date: 2024-07-11 19:53:59
password:
---

## 安装字体

 [Nerd Font字体下载](https://www.nerdfonts.com/font-downloads)在网站中选择喜欢的字体，右击下载按钮，复制下载地址，在终端中下载

```bash
wget https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/JetBrainsMono.zip
sudo unzip JetBrainsMono.zip -d /usr/share/fonts/JetBrainsMono
sudo fc-cache -fv
```

 **安装完成后注意修改终端字体！！！**

## 下载 NeoVim

```bash
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz

# 加入环境变量
export PATH="$PATH:/opt/nvim-linux64/bin"
# 或者使用软连接连接到已经存在PATH变量中的/usr/local/bin下
sudo ln -s /opt/nvim-linux64/bin/nvim /usr/local/bin/nvim
```

## 安装LazyVim

```bash
git clone https://github.com/LazyVim/starter ~/.config/nvim
rm -rf ~/.config/nvim/.git
```

## 自动安装配置插件

启动neovim，会自动安装需要的插件

```bash
nvim
```
