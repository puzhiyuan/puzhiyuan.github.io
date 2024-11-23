---
title: linkList
top: false
cover: false
toc: true
mathjax: true
summary: 总结了《代码随想录》中关于链表的常见算法与实现，包括移除链表元素、设计链表、反转链表、两两交换节点、删除倒数第N个节点、链表相交、以及环形链表的检测与求解。
tags: 代码随想录
categories: algorithm
abbrlink: e5eac3e7
date: 2024-09-07 22:04:57
password:
---

**链表元素结构体**

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
```

### 1. 移除链表元素

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

#### 1.1 增加虚拟头节点

```python
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy_head = ListNode()
        dummy_head.next = head
        p = dummy_head
        while p.next != None:
            if p.next.val == val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy_head.next
```



### 2. 设计链表

你可以选择使用单链表或者双链表，设计并实现自己的链表。

单链表中的节点应该具备两个属性：`val` 和 `next` 。`val` 是当前节点的值，`next` 是指向下一个节点的指针/引用。

如果是双向链表，则还需要属性 `prev` 以指示链表中的上一个节点。假设链表中的所有节点下标从 **0** 开始。

实现 `MyLinkedList` 类：

- `MyLinkedList()` 初始化 `MyLinkedList` 对象。
- `int get(int index)` 获取链表中下标为 `index` 的节点的值。如果下标无效，则返回 `-1` 。
- `void addAtHead(int val)` 将一个值为 `val` 的节点插入到链表中第一个元素之前。在插入完成后，新节点会成为链表的第一个节点。
- `void addAtTail(int val)` 将一个值为 `val` 的节点追加到链表中作为链表的最后一个元素。
- `void addAtIndex(int index, int val)` 将一个值为 `val` 的节点插入到链表中下标为 `index` 的节点之前。如果 `index` 等于链表的长度，那么该节点会被追加到链表的末尾。如果 `index` 比长度更大，该节点将 **不会插入** 到链表中。
- `void deleteAtIndex(int index)` 如果下标有效，则删除链表中下标为 `index` 的节点。

```python
class MyLinkedList(object):

    def __init__(self):
        self.dummy_head = ListNode()
        self.size = 0


    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index < 0 or index > self.size-1:
            return -1
        p = self.dummy_head.next
        while index:
            p = p.next
            index -= 1
        return p.val


    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.dummy_head.next = ListNode(val, self.dummy_head.next)
        self.size += 1


    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """ 
        p = self.dummy_head
        while p.next:
            p = p.next
        p.next = ListNode(val)
        self.size += 1


    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        if index > self.size:
            return

        tmp = ListNode(val)
        p = self.dummy_head
        while index:
            p = p.next
            index -= 1
        p.next = ListNode(val,p.next)
        self.size += 1



    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if index > self.size-1:
            return
        p = self.dummy_head
        while index:
            p = p.next
            index -= 1
        # q = p.next
        # p.next = q.next
        p.next = p.next.next
        self.size -= 1




# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```



### 3. 反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

#### 3.1 双指针

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        cur = head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
```



#### 3.2 递归法

```python
class Solution(object):
    def reverse(self, pre, cur):
        if cur == None:
            return pre
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
        return self.reverse(pre, cur)

    def reverseList(self, head):
        return self.reverse(None, head)
```



### 4. 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

#### 4.1 双指针

```python
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy_head = ListNode(next=head)
        cur = dummy_head
        while cur.next and cur.next.next:
            tmp = cur.next
            cur.next = tmp.next
            tmp2 = cur.next.next
            cur.next.next = tmp
            tmp.next = tmp2
            # 别忘了指针右移
            cur = cur.next.next
        return dummy_head.next
```

#### 4.2 递归法

```python
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 递归终止
        if head == None or head.next == None:
            return head
        last = head.next.next
        pre = head
        head = pre.next
        head.next = pre
        pre.next = self.swapPairs(last)
        return head

```



### 5. 删除链表的倒数第N个节点

#### 5.1 暴力破解

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        # 暴力破解
        dummy_head = ListNode(next=head)
        size = 0 
        cur = dummy_head
        while cur.next:
            cur = cur.next
            size += 1
        cur = dummy_head
        for i in range(size-n):
            cur = cur.next
        cur.next = cur.next.next
        return dummy_head.next
```

#### 5.2 双指针

两个差距为n+1快指针到表尾的时候满指针在倒数n+1位置上

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy_head = ListNode(next=head)
        pre = cur = dummy_head

        while n and cur:
            cur = cur.next
            n -= 1
        cur = cur.next
        while cur:
            cur = cur.next
            pre = pre.next
        pre.next = pre.next.next
        return dummy_head.next
```



### 6. 链表相交

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

[题目链接](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/description/)

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lenA = self.getLength(headA)
        lenB = self.getLength(headB)

        curA, curB = headA, headB
        #将A设置为最长的链表
        if lenB > lenA:
            curA, curB = headB, headA
            lenA, lenB = lenB, lenA
        
        for _ in range(lenA-lenB):
            curA = curA.next
        
        while curA:
            if curA == curB:
                return curA
            curA = curA.next
            curB = curB.next
        return None    

        
    def getLength(self, head):
        len = 1
        while head.next:
            head = head.next
            len += 1
        return len
```



### 7. 环形链表Ⅱ

求出环入口指针

![142.环形链表II（求入口）](https://code-thinking.cdn.bcebos.com/gifs/142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II%EF%BC%88%E6%B1%82%E5%85%A5%E5%8F%A3%EF%BC%89.gif)

```python
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head
        while fast.next and fast.next.next:
            # 注意这里要先移动再判断
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                idx1 = head
                idx2 = fast
                while idx1 != idx2:
                    idx1 = idx1.next
                    idx2 = idx2.next
                return idx1
        return None
```

### 8. 总结

![img](https://code-thinking-1253855093.file.myqcloud.com/pics/%E9%93%BE%E8%A1%A8%E6%80%BB%E7%BB%93.png)
