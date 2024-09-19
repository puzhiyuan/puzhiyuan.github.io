---
title: string
top: false
cover: false
toc: true
mathjax: true
date: 2024-09-07 22:05:25
password:
summary: 总结了《代码随想录》中关于字符串的常见算法，包括反转字符串的双指针法及部分反转的实现方法。
tags: 代码随想录
categories: algorithm
---

### 1. [反转字符串](https://leetcode.cn/problems/reverse-string/)

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 `s` 的形式给出。

不要给另外的数组分配额外的空间，你必须**[原地](https://baike.baidu.com/item/原地算法)修改输入数组**、使用 O(1) 的额外空间解决这一问题。

#### 双指针法：

```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        length = len(s)
        for i in range(length/2):
            s[i], s[length-1] = s[length-1], s[i]
            length -= 1
        return s
```



### 2. [反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/)

给定一个字符串 `s` 和一个整数 `k`，从字符串开头算起，每计数至 `2k` 个字符，就反转这 `2k` 字符中的前 `k` 个字符。

- 如果剩余字符少于 `k` 个，则将剩余字符全部反转。
- 如果剩余字符小于 `2k` 但大于或等于 `k` 个，则反转前 `k` 个字符，其余字符保持原样。

 
