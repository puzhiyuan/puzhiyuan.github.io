---
title: binaryTree
top: false
cover: false
toc: true
mathjax: true
summary: 总结了《代码随想录》中关于二叉树的常见遍历方法，包括递归与非递归的前序、中序、后序和层序遍历等经典实现。
tags: 代码随想录
categories: algorithm
abbrlink: aa987e96
date: 2024-09-07 22:06:30
password:
---

### 二叉树结构定义

```python
Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```





### 递归遍历

#### 前序遍历

```python

```



#### 中序遍历

```python

```



#### 后序遍历

```python

```



### 非递归遍历



#### 前序遍历

#### 后序遍历

```python
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        stack = [root]
        res = []
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res[::-1]
```

#### 中序遍历

```python
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        res = []
        stack = []
        cur = root

        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res
```





### 层序遍历

```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return
        queue = deque([root])
        res = []
        while queue:
            res_level = []
            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                res_level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(res_level)
        return res 
```

#### 错误递归

```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        res = []
        level = 0
        self.traversal(root, level, res)
        return res
        

    def traversal(self, root, level, res):
        if root == None:
            return
        if len(res) != level+1:  # 这里不能这样写，会出错
            res.append([root.val])
        else:
            res[level].append(root.val)

        self.traversal(root.left, level+1, res)
        self.traversal(root.right, level+1, res)
```

#### 正确递归

```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        res = []
        level = 0
        self.traversal(root, level, res)
        return res
        

    def traversal(self, root, level, res):
        if root == None:
            return
        if len(res) == level:
            res.append([])
        
        res[level].append(root.val)
        self.traversal(root.left, level+1, res)
        self.traversal(root.right, level+1, res)
```

