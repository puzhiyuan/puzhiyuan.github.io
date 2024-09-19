---
title: array
top: false
cover: false
toc: true
mathjax: true
date: 2024-09-07 22:04:13
password:
summary: 总结了《代码随想录》中关于数组的常见算法与实现，包括二分查找、数组元素移除、双指针等经典解法。
tags: 代码随想录
categories: algorithm
---

### 1. 二分查找

#### 1. 左闭右闭

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) -1 
        while(left <= right):
            mid = (left + right) / 2
            if target > nums[mid]:
                left = mid + 1
            if target < nums[mid]:
                right = mid - 1
            if target == nums[mid]:
                return mid
        return -1
```



#### 2. 左闭右开

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)  # 修改1
        while(left < right):  # 修改2
            mid = (left + right) / 2
            if target > nums[mid]:
                left = mid + 1
            if target < nums[mid]:
                right = mid  # 修改3
            if target == nums[mid]:
                return mid
        return -1
```

### 2. 数组移除元素

#### 1. 暴力破解

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        p, q = 0, len(nums)
        while p < q:
            if nums[p] == val:
                for i in range(p, q-1):
                    nums[i] = nums[i+1]
                p -= 1
                q -= 1
            p += 1
        return q
```

#### 2. 快慢指针

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        slowidx = fastidx = 0
        while fastidx < len(nums):
            if nums[fastidx] != val:
                nums[slowidx] = nums[fastidx]
                slowidx += 1
                fastidx += 1
            if nums[fastidx] == val:
                fastidx += 1
        return slowidx
```

代码优化之后

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        slowidx = fastidx = 0
        while fastidx < len(nums):
            if nums[fastidx] != val:
                nums[slowidx] = nums[fastidx]
                slowidx += 1
            fastidx += 1
        return slowidx
```

### 3. 有序数组的平方

#### 1. 暴力破解

```python
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(len(nums)):
            nums[i] = nums[i] * nums[i]
        nums.sort()
        return nums
```



#### 2. 双指针

```python
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        p, q = 0, len(nums)-1
        new_nums = []
        while p <= q:
            if math.fabs(nums[p]) > math.fabs(nums[q]):
                new_nums = [nums[p] * nums[p]] + new_nums
                p += 1
            else:
                new_nums = [nums[q] * nums[q]] + new_nums
                q -= 1
        return new_nums
```

### 4. 长度最小子数组

#### 1. 暴力破解

LeetCode 中该解法不能完成提交

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        min = 0
        for i in range(len(nums)):
            sum = 0
            for j in range(i, len(nums)):
                sum += nums[j]
                if sum >= target:
                    if min == 0:
                        min = j-i+1
                    if min > j-i+1:
                        min = j-i+1
                    break
        return min
```

#### 2. 滑动窗口

<img src="https://code-thinking.cdn.bcebos.com/gifs/209.%E9%95%BF%E5%BA%A6%E6%9C%80%E5%B0%8F%E7%9A%84%E5%AD%90%E6%95%B0%E7%BB%84.gif" alt="209.长度最小的子数组" style="zoom:50%;" />

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        sum = 0
        result = 0
        slow = 0
        for fast in range(len(nums)):
            sum += nums[fast]
            while sum >= target:
                min_len = fast - slow + 1
                if result == 0:
                    result = min_len
                if result > min_len:
                    result = min_len
                sum -= nums[slow]
                slow += 1
        return result
```

### 5. 螺旋矩阵Ⅱ

<img src="https://code-thinking-1253855093.file.myqcloud.com/pics/20220922102236.png" alt="img" style="zoom: 33%;" />

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        result = [[0] * n for _ in range(n)]
        loop,mid = n // 2, n // 2
        startx, starty = 0, 0
        offset = 1
        num = 1
        while loop:
            i = startx
            j = starty
            while j < n-offset:  # 从左到右
                result[i][j] = num
                num += 1
                j += 1
                        
            while i < n-offset:  # 从上到下
                result[i][j] = num
                num += 1
                i += 1
                        
            while j > starty:  # 从右到左
                result[i][j] = num
                num += 1
                j -= 1
            
            while i > startx:  # 从下到上
                result[i][j] = num
                num += 1
                i -= 1
            
            offset += 1
            startx += 1
            starty += 1
            loop -= 1
        
        if n % 2 == 1:
            result[mid][mid] = num
        return result
```

