---
title: hexo小技巧
top: false
cover: false
toc: true
mathjax: true
summary: hexo的插入图片和引用已有文件的方法
categories:
  - Hexo tips
abbrlink: cd7cca12
date: 2024-06-23 18:41:49
password:
tags:
---

### Hexo文章中插入图片的记录

#### 安装图片支持插件

在使用Hexo搭建博客时，默认情况下可能不支持Markdown文件中直接插入图片的功能。为了解决这个问题，我们需要安装一个插件来增强Markdown的功能。

1. **安装`hexo-asset-image`插件**：
   打开终端，进入Hexo项目目录，执行以下命令来安装插件：

   ```sh
   npm install hexo-asset-image --save
   ```

2. **配置Hexo**：
   在Hexo的配置文件`_config.yml`中，修改以下配置：

   ```yaml
   post_asset_folder: true
   ```

#### 创建新文章

1. **创建新文章**：
   在终端中，进入Hexo项目目录，使用以下命令创建新文章：

   ```sh
   hexo new post "文章标题"
   ```

   这将会在`source/_posts`目录下生成一个以文章标题命名的新Markdown文件。

#### 上传图片

1. **放置图片**：
   在文章的Markdown文件所在的目录中，创建一个名为`images`的文件夹（如果尚未存在）：

   ```sh
   mkdir -p source/_posts/文章标题/images
   ```

   将你想要插入文章中的图片上传到这个`images`文件夹中。

#### 在Markdown中插入图片

1. **使用相对路径引用图片**：
   打开刚创建的Markdown文件，在文章中适当的位置，使用以下格式插入图片：

   ```markdown
   ![图片描述](./文章标题/images/图片文件名)
   ```

#### 生成并预览文章

1. **生成静态文件**：
   在终端中，执行以下命令来生成静态文件：

   ```sh
   hexo generate
   ```

2. **本地预览**：
   启动本地服务器预览你的文章：

   ```sh
   hexo server
   ```

   打开浏览器，访问`http://localhost:4000`（或相应的端口），查看你的文章和图片是否正确显示。

#### 发布文章

1. **部署文章**：
   当你满意文章的显示效果后，可以执行以下命令来部署你的博客：

   ```sh
   hexo deploy
   ```

通过以上步骤，你就可以在Hexo的文章中插入图片，并成功发布到你的博客上了。



### Hexo文章中插入文章引用的记录

#### 安装`hexo-relative-link`插件

1. **安装插件**：
   在终端中，进入你的Hexo项目目录，执行以下命令来安装`hexo-relative-link`插件：

   ```sh
   npm install hexo-relative-link
   ```

2. **启用插件**：
   在Hexo的配置文件`_config.yml`中，修改以下配置：

   ```yaml
   relative_link: true
   ```

#### 创建和引用文章

1. **创建新文章**：
   使用Hexo命令创建新文章：

   ```sh
   hexo new post "文章标题"
   ```

   这会在`source/_posts`目录下创建一个新的Markdown文件。

2. **编辑文章**：
   编辑Markdown文件，撰写你的文章内容。

3. **插入相对链接**：
   在Markdown文件中，使用标准的Markdown链接语法插入链接，例如：

   ```markdown
   [这是一个链接](../path/to/your/other-page)
   ```

   这里的`../path/to/your/other-page`是相对路径，指向你博客中的另一个页面或资源。

#### 生成静态文件

1. **生成静态文件**：
   在终端中，执行以下命令来生成静态文件：

   ```sh
   hexo generate
   ```

#### 本地预览

1. **启动本地服务器**：
   启动Hexo的本地服务器预览你的博客：

   ```sh
   hexo server
   ```

   打开浏览器，访问`http://localhost:4000`（或相应的端口）。

2. **检查链接**：
   浏览你的文章，检查相对链接是否正确工作。

#### 部署博客

1. **部署到远程服务器**：
   当你满意本地预览的效果后，执行以下命令将你的博客部署到远程服务器：

   ```sh
   hexo deploy
   ```

通过使用`hexo-relative-link`插件，你可以确保在不同环境中（如本地开发和生产环境）链接都能正确地解析，从而避免因绝对路径导致的链接错误。
