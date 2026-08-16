---
title: nvidia_container_toolkit
top: false
cover: false
toc: true
mathjax: true
summary: 通过配置 nvidia-container-toolkit，让 Docker 容器可以访问宿主机的 GPU，实现深度学习等计算任务加速。
tags:
  - docker
  - GPU
categories:
  - tool
abbrlink: 1523
date: 2025-05-11 13:41:52
---

## 🧰 nvidia-container-toolkit

`nvidia-container-toolkit` 是 NVIDIA 官方提供的工具，用于 **让 Docker 等容器平台能够访问宿主机的 NVIDIA GPU 资源**。
 它充当了一座“桥梁”，在容器的隔离环境与宿主机的 GPU 驱动之间建立连接。



### 🔧 安装步骤（适用于 Debian/Ubuntu）

你可以参考 [官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，或按以下步骤快速安装：

1. 添加 NVIDIA 软件源与密钥：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

1. 更新并安装 toolkit：

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

1. 配置 Docker 使用 NVIDIA runtime：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```



### ✅ 验证是否安装成功

你可以运行以下命令，检查容器是否成功访问 GPU：

```bash
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

若输出了显卡信息（如 GPU 型号和驱动版本），说明安装成功。



### ⚠️ 注意事项

- 如果未安装 `nvidia-container-toolkit`，即使主机上已经正确安装了 CUDA 和驱动，容器中也无法使用 GPU。
- 安装前确保你已经正确安装了 NVIDIA 驱动，且主机系统能使用 `nvidia-smi` 命令。

