---
title: mount_disk
top: false
cover: false
toc: true
mathjax: true
summary: Linux下外接磁盘的分区和挂载操作
tags:
  - disk
categories:
  - Linux
abbrlink: 7cb0145c
date: 2024-10-18 22:31:25
password:
---

## 1. 插入设备

将U盘或外部磁盘通过接口连接到计算机。



## 2. 确认设备连接

使用以下命令查看当前连接的所有块设备：

```bash
lsblk
```

示例输出：

```bash
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
sda      8:0    0 465.8G  0 disk 
├─sda1   8:1    0     1M  0 part 
├─sda2   8:2    0   513M  0 part /boot/efi
└─sda3   8:3    0 465.3G  0 part /var/snap/firefox/common/host-hunspell
                                 /
sdb      8:16   0   1.8T  0 disk 
├─sdb1   8:17   0     1K  0 part 
├─sdb5   8:21   0   500G  0 part 
└─sdb6   8:22   0   1.3T  0 part 
```

在此示例中，`sdb` 是新插入的磁盘，包含三个分区 `sdb1`、`sdb5`、`sdb6`。



## 3. 查看分区和文件系统信息

### 3.1 检查磁盘是否已分区

在查看分区和文件系统信息之前，你可以使用 `fdisk` 命令来检查磁盘是否已经分区。运行以下命令：

```bash
sudo fdisk -l /dev/sdb
```

这将列出所有已检测到的磁盘及其分区。如果 `/dev/sdb` 没有分区，你将看到类似以下的输出：

```bash
Disk /dev/sdb: 1.8 TiB, 2000398934016 bytes, 3907029168 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: gpt
```

如果磁盘没有分区，你将需要创建新的分区。



### 3.2 创建新分区

记得在进行分区操作之前备份任何重要数据，因为分区操作会清除磁盘上的所有数据。

使用 `fdisk` 或 `parted` 来创建新的分区。以下是使用 `fdisk` 创建新分区的步骤：

1. 运行 `fdisk` 命令并指定磁盘设备：

```bash
sudo fdisk /dev/sdb
```

2. 按 `m` 键进入帮助模式，查看可用的命令列表。

3. 按 `n` 键创建新分区。根据提示选择分区类型（例如，`p` 为 primary partition）。

4. 输入分区的起始和结束位置，或者接受默认值。

5. 重复步骤 3 和 4 来创建更多分区。

6. 按 `w` 键写入分区表并退出 `fdisk`。



### 3.3 格式化新分区

创建分区后，你需要格式化分区以创建文件系统。例如，要创建一个 `ext4` 文件系统，可以使用以下命令：

```bash
# 将 /dev/sdb6 替换为你创建的分区设备名
sudo mkfs.ext4 /dev/sdb6
```



### 3.4 验证分区和文件系统信息

分区和格式化完成后，你可以再次使用 `blkid` 命令来查看分区的文件系统类型：

```bash
# 将 /dev/sdb6 替换为你创建的分区设备名
sudo blkid /dev/sdb6
```

示例输出：

```bash
/dev/sdb6: LABEL="datasets" UUID="8f17cfd9-4f90-45e6-a36b-2e1bfcb452a0" BLOCK_SIZE="4096" TYPE="ext4" PARTUUID="7f277d52-06"
```



## 4. 创建挂载点

选择一个目录作为挂载点，通常位于 `/mnt` 或 `/media` 下。例如，创建 `/mnt/sdb6` 目录：

```bash
sudo mkdir -p /mnt/sdb6
```



## 5. 挂载分区

使用 `mount` 命令将设备挂载到挂载点：

```bash
sudo mount /dev/sdb6 /mnt/sdb6
```

**注意**：根据文件系统类型，可能需要指定 `-t` 参数。例如，对于NTFS文件系统：

```bash
sudo mount -t ntfs /dev/sdb6 /mnt/sdb6
# 或者，对于FAT32
sudo mount -t vfat /dev/sdb6 /mnt/sdb6
```



## 6. 验证挂载

确保设备已成功挂载，可以使用以下命令查看挂载情况：

```bash
df -h | grep /mnt/sdb6
# 或使用 `lsblk` 
lsblk
```

示例输出：

```bash
sda      8:0    0 465.8G  0 disk 
├─sda1   8:1    0     1M  0 part 
├─sda2   8:2    0   513M  0 part /boot/efi
└─sda3   8:3    0 465.3G  0 part /var/snap/firefox/common/host-hunspell
                                 /
sdb      8:16   0   1.8T  0 disk 
├─sdb1   8:17   0     1K  0 part 
├─sdb5   8:21   0   500G  0 part 
└─sdb6   8:22   0   1.3T  0 part /mnt/sdb6
```



## 7. 设置开机自动挂载（可选）

如果希望设备在系统启动时自动挂载，可以编辑 `/etc/fstab` 文件。

### 7.1 获取设备UUID

使用 `blkid` 命令获取设备的UUID：

```bash
sudo blkid /dev/sdb1
```

示例输出：

```bash
/dev/sdb6: LABEL="datasets" UUID="8f17cfd9-4f90-45e6-a36b-2e1bfcb452a0" BLOCK_SIZE="4096" TYPE="ext4" PARTUUID="7f277d52-06"
```

记下 `UUID` 值，例如 `8f17cfd9-4f90-45e6-a36b-2e1bfcb452a0`。



### 7.2 备份当前 `fstab` 文件

在修改之前，建议备份现有的 `fstab` 文件：

```bash
sudo cp /etc/fstab /etc/fstab.backup
```



### 7.3 编辑 `fstab` 文件

使用文本编辑器打开 `fstab` 文件：

```bash
sudo vim /etc/fstab
```

在文件末尾添加一行，格式如下（根据实际情况替换 `UUID` 和 `vfat`（文件系统类型））：

```bash
UUID=8f17cfd9-4f90-45e6-a36b-2e1bfcb452a0  /mnt/sdb6  ext4  defaults  0  2
```

- **UUID=8f17cfd9-4f90-45e6-a36b-2e1bfcb452a0**：指定要挂载的分区的唯一标识符（UUID）。使用UUID可以避免设备名称变化带来的问题。
- **/mnt/sdb6**：挂载点，即分区将被挂载到的目录。
- **ext4**：文件系统类型。这里表示该分区使用的是 `ext4` 文件系统。
- **defaults**：挂载选项。`defaults` 表示使用默认的挂载选项，包括读写权限、自动挂载等。
- **0**：用于 `dump` 命令的备份标志。`0` 表示不需要备份此分区。
- **2**：文件系统检查顺序（`fsck`）。根文件系统通常为 `1`，其他分区为 `2`，表示启动时 `fsck` 检查的顺序。



### 7.4 测试 `fstab` 配置

使用以下命令测试 `fstab` 是否配置正确：

```bash
sudo mount -a
```

如果没有错误提示，说明配置正确。



## **8. 调整挂载点权限（可选）**

根据需要，可以更改挂载点的所有者和权限，以便普通用户访问：

```bash
sudo chown -R your_username:your_group /mnt/sdb6
```

将 `your_username` 和 `your_group` 替换为实际的用户名和用户组。



## 9. 卸载设备

在拔出设备之前，务必先卸载以防数据丢失：

```bash
sudo umount /mnt/sdb6
# 或使用设备名称
sudo umount /dev/sdb6
```



## 常见问题与注意事项

- **权限问题**：确保挂载点目录的权限设置正确，允许需要的用户访问。
- **文件系统兼容性**：不同的文件系统类型（如 NTFS、FAT32、ext4）在Linux中的支持程度不同，挂载时需指定正确的类型。
- **数据备份**：在格式化分区或进行重要操作前，务必备份重要数据。
- **设备名称变化**：有时设备名称（如 `sdb`）可能会变化，特别是在插入多个USB设备时。可以使用UUID来避免。
