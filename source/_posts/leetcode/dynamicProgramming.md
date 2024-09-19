---
title: dynamicProgramming
top: false
cover: false
toc: true
mathjax: true
date: 2024-09-07 22:07:06
password:
summary: 总结了《代码随想录》中关于动态规划的经典问题，提供不同的解法和代码实现，包括斐波那契数列、爬楼梯、不同路径等题目。
tags: 代码随想录
categories: algorithm
---

### 1. [斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

**斐波那契数** （通常用 `F(n)` 表示）形成的序列称为 **斐波那契数列** 。该数列由 `0` 和 `1` 开始，后面的每一项数字都是前面两项数字的和。也就是：

`F(0) = 0，F(1) = 1`
`F(n) = F(n - 1) + F(n - 2)，其中 n > 1`

给定 `n` ，请计算 `F(n)` 。



```python
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return n
        dp = []
        dp.append(0)
        dp.append(1)
        for i in range(2, n+1):
            dp.append(dp[i-1] + dp[i-2])
        return dp[n]
```

#### 优化空间复杂度：

```python
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return n
        dp0, dp1 = 0, 1
        for i in range(2, n+1):
            tmp = dp0 + dp1
            dp0, dp1 = dp1, tmp
        return dp1
```

#### 递归实现：

```python
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return n
        return self.fib(n-1) + self.fib(n-2)
```





### 2. [爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        dp = [0] * (n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

#### 优化空间复杂度

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        dp1, dp2 = 1, 2
        for i in range(3, n+1):
            tmp = dp1 + dp2
            dp1, dp2 = dp2, tmp
        return dp2
```

#### ❗注意这一题用递归会超出限制



### 3. [使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

 给你一个整数数组 `cost` ，其中 `cost[i]` 是从楼梯第 `i` 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 `0` 或下标为 `1` 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。

```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        max = len(cost)
        if max < 2:
            return 0
        dp = [0] * (max + 1)
        dp[0], dp[1] = 0, 0
        for i in range(2, (max+1)):
            dp[i] = min((dp[i-1]+cost[i-1]), (dp[i-2]+cost[i-2]))
        return dp[max]
```

#### 优化空间复杂度

```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        max = len(cost)
        if max < 2:
            return 0
        dp0, dp1 = 0, 0
        for i in range(2, (max+1)):
            tmp = min((dp0+cost[i-2]), (dp1+cost[i-1]))
            dp0, dp1 = dp1, tmp
        return dp1
```



### 4. [不同路径](https://leetcode.cn/problems/unique-paths/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0] * (n)] * m
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

#### 递归（❗超时）

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m == 1 or n == 1:
            return 1
        return self.uniquePaths(m-1,n) + self.uniquePaths(m,n-1)
```



### 5. [不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

:warning::warning::warning:初始化的时候如果遇到障碍，应该break，障碍之后的初始化都应该为0（之后的位置都无法到达）

```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        
        dp = [[0] * n for _ in range(m)]
        for i in range(n):
            if obstacleGrid[0][i] != 1:
                dp[0][i] = 1
            else:  # 这里需要break,置0是不对的
                # dp[0][i] = 0 
                break
        for j in range(m):
            if obstacleGrid[j][0] != 1:
                dp[j][0] = 1
            else:  # 这里需要break,置0是不对的
                # dp[j][0] = 0
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

```



### 6. [整数拆分](https://leetcode.cn/problems/integer-break/)

给定一个正整数 `n` ，将其拆分为 `k` 个 **正整数** 的和（ `k >= 2` ），并使这些整数的乘积最大化。

返回 *你可以获得的最大乘积* 。

#### 这一步在遍历 `j` 时不断更新 `dp[i]`，以保证得到的是最大的可能乘积。

:warning:初始化：`n>=2`， 初始化`dp[2]`，递推后面的结果。

:warning:递推公式：为什么`max()`中为什么要有`dp[i]`->这一步在遍历 `j` 时不断更新 `dp[i]`，以保证得到的是最大的可能乘积。

```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n+1):
            for j in range(2, i):
                dp[i] = max(dp[i], dp[i-j] * j, (i-j)*j)
        return dp[n]
```

