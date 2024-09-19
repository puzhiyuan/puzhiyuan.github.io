---
title: hashTable
top: false
cover: false
toc: true
mathjax: true
date: 2024-09-07 22:05:13
password:
summary: 总结了《代码随想录》中关于哈希表的常见算法与实现，涵盖了有效的字母异位词、数组交集、快乐数、两数之和、赎金信和三数之和等经典题目。
tags: 代码随想录
categories: algorithm
---

### 1. [有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

给定两个字符串 *s* 和 *t* ，编写一个函数来判断 *t* 是否是 *s* 的字母异位词。

**注意：**若 *s* 和 *t* 中每个字符出现的次数都相同，则称 *s* 和 *t* 互为字母异位词。

#### 1.1 集合

```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        char_count = {}
        for i in s:
            if not char_count.get(i):
                char_count[i] = 1
            else:
                char_count[i] += 1
        
        for i in t:
            if not char_count.get(i):
                return False
            else:
                char_count[i] -= 1
        
        for i in char_count.keys():
            if char_count[i] != 0:
                return False
        return True
```

#### 1.2 数组

```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        char_count = [0]*26
        for i in s:
            char_count[ord(i)-ord('a')] += 1
        
        for i in t:
            char_count[ord(i)-ord('a')] -= 1

        for i in range(26):
            if char_count[i] != 0:
                return False
        return True
```



### 2. [两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

给定两个数组 `nums1` 和 `nums2` ，返回 *它们的 交集* 。输出结果中的每个元素一定是 **唯一** 的。我们可以 **不考虑输出结果的顺序** 。

#### 2.1 集合

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        result = set()
        nums1_set = set(nums1)
        for i in nums2:
            if i in nums1_set:
                result.add(i)
        return list(result)
```

#### 2.2 列表&集合

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        nums1_count = [0] * 1001
        result = set()
        for i in nums1:
            nums1_count[i] += 1
        
        for i in nums2:
            if nums1_count[i] != 0:
                result.add(i)
        return list(result)
```



### 3. [快乐数](https://leetcode.cn/problems/happy-number/)

编写一个算法来判断一个数 `n` 是不是快乐数。

**「快乐数」** 定义为：

- 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
- 然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。
- 如果这个过程 **结果为** 1，那么这个数就是快乐数。

如果 `n` 是 *快乐数* 就返回 `true` ；不是，则返回 `false` 。

#### 3.1 集合

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        sum_set = set()
        while 1:
            sum = self.getSum(n)
            if sum == 1:
                return True
            if sum in sum_set:
                return False
            sum_set.add(sum)
            n = sum
       
    def getSum(self,n):
        sum = 0
        while n:
            sum += (n % 10) ** 2
            n /= 10
        return sum
```



### 4. [两数之和](https://leetcode.cn/problems/two-sum/)

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

#### 4.1 暴力破解

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]
```

#### 4.2 哈希表(字典)

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_count = {}
        for i in range(len(nums)):
            tmp = target - nums[i]
            if tmp in nums_count:
                return [i, nums_count[tmp]]
            nums_count[nums[i]] = i

```



### 5. [四数相加 II](https://leetcode.cn/problems/4sum-ii/)

给你四个整数数组 `nums1`、`nums2`、`nums3` 和 `nums4` ，数组长度都是 `n` ，请你计算有多少个元组 `(i, j, k, l)` 能满足：

- `0 <= i, j, k, l < n`
- `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`

#### 5.1 暴力破解（超时）

```python
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
        num = 0
        for i in nums1:
            for j in nums2:
                for k in nums3:
                    for l in nums4:
                        if i + j + k + l == 0:
                            num += 1
        return num
```

#### 5.2 哈希表（字典）

```python
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
        num = 0
        dic = {}
        for i in nums1:
            for j in nums2:
                if i + j not in dic:
                    dic[i+j] = 1
                else:
                    dic[i+j] += 1
        for k in nums3:
            for l in nums4:
                tmp = 0 - (k+l)
                if tmp in dic:
                    num += dic[tmp]
        return num
```



### 6. [赎金信](https://leetcode.cn/problems/ransom-note/)

给你两个字符串：`ransomNote` 和 `magazine` ，判断 `ransomNote` 能不能由 `magazine` 里面的字符构成。

如果可以，返回 `true` ；否则返回 `false` 。

`magazine` 中的每个字符只能在 `ransomNote` 中使用一次。

#### 6.1 数组

```python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        ransomNote_list = [0] * 26
        magazine_list = [0] * 26
        for i in ransomNote:
            ransomNote_list[ord(i) - ord('a')] += 1
        for i in magazine:
            magazine_list[ord(i) - ord('a')] += 1
        return all(magazine_list[i] >= ransomNote_list[i] for i in range(26))
```

#### 6.2 字典

```python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        char_set = {}
        for i in ransomNote:
            if i in char_set:
                char_set[i] += 1
            else:
                char_set[i] = 1
        for i in magazine:
            if i in char_set:
                char_set[i] -= 1
        for i in char_set.keys():
            if char_set[i] > 0:
                return False
        return True
```

### 7. [三数之和](https://leetcode.cn/problems/3sum/)

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请

你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                return result
            if i>0 and nums[i]==nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                if nums[i] + nums[left] + nums[right] < 0:
                    left += 1
                elif nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    while right > left and nums[right] == nums[right-1]:
                        right -= 1
                    while right > left and nums[left] == nums[left+1]:
                        left += 1
                    left += 1
                    right -= 1
        return result
```

