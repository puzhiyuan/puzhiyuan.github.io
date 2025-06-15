---
title: docker_mirrors
top: false
cover: false
toc: true
mathjax: true
summary: 通过配置国内 Docker 镜像加速源，可有效解决因网络不稳定或被墙导致的镜像拉取超时、构建失败等问题。
tags:
  - docker
categories:
  - tool
abbrlink: 3654
date: 2025-05-11 13:41:23
password:
---

## Docker mirrors

### ❗ 问题说明：无法从 Docker Hub 拉取镜像

在使用 Docker 构建镜像时，如果你遇到了类似以下错误：

```bash
failed to do request: Head "https://registry-1.docker.io/v2/nvidia/cuda/manifests/11.1.1-devel-ubuntu18.04": dial tcp 31.13.96.193:443: i/o timeout
```

这意味着 **Docker 客户端无法从 Docker Hub 拉取镜像**。由于访问 `registry-1.docker.io` 网络质量不稳定、连接超时，导致构建失败。

> Docker 在构建镜像时，会自动拉取 `Dockerfile` 中定义的基础镜像，例如：
>
> ```dockerfile
> FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
> ```
>
> 此操作本质上是发起一个 HTTPS 请求到官方的 Docker Hub：
>
> ```
> https://registry-1.docker.io/v2/nvidia/cuda/manifests/11.1.1-devel-ubuntu18.04
> ```
>
> 当你看到 `i/o timeout`，这通常表示：
>
> - 网络连接 Docker Hub 失败（被墙、丢包、DNS 解析失败等）
> - 远程服务器响应缓慢（如拉取大型镜像时）
> - 本地 DNS 或网络配置异常



### ✅ 解决方案：配置国内 Docker 镜像加速源（推荐）

将 Docker 的镜像拉取源设置为国内镜像代理，绕过连接 Docker Hub 直接失败的问题。

1. **创建 Docker 配置文件（如不存在）**

```bash
sudo mkdir -p /etc/docker
```

2. **编辑 daemon.json 文件，添加 registry-mirrors 字段**

   `registry-mirrors` 的列表里面可以加入对应的镜像地址，比如 `aicarbon.xyz` ，但是需要注意在前面加上 `https:// `，构成 `https://aicarbon.xyz` 。（以下配置截至编辑日期依旧可用，如不可用，参考 github：[DockerHub](https://github.com/dongyubin/DockerHub)）

```bash
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://aicarbon.xyz",
    "https://666860.xyz"
  ]
}
EOF
```

3. **重启 Docker 服务**

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```



### ✅ 验证配置是否生效

执行以下命令查看当前 Docker 的镜像加速设置：

```bash
docker info
```

输出示例中包含如下内容：

```bash
 Registry Mirrors:
  https://aicarbon.xyz/
  https://666860.xyz/
```

说明配置成功。

