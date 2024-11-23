---
title: Concurrent & Parallel
top: false
cover: false
toc: true
mathjax: true
summary: Python中多线程适合I/O密集型任务，线程共享内存空间；多进程适合CPU密集型任务，进程拥有独立内存空间，不受GIL限制。
tags: 并发编程
categories:
  - python
abbrlink: 623fb189
date: 2024-09-29 22:12:38
password:
---

### Python 多线程与多进程

#### 背景

在Python编程中，多线程（`multithreading`）和多进程（`multiprocessing`）是两种不同的并发处理方法。它们允许程序同时执行多个任务，从而提高程序的效率和响应速度。随着计算机硬件的发展，特别是多核处理器的普及，利用并发技术来提升程序性能变得越来越重要。

#### 解决问题的思路

- **多线程** 主要用于I/O密集型任务，如文件操作、网络请求等。Python中的GIL（全局解释器锁）限制了多线程在CPU密集型任务上的并行执行能力，但对于I/O密集型任务，多线程可以显著提高程序的响应性和效率。
- **多进程** 则更适用于CPU密集型任务，如大量计算。由于每个进程都有自己的Python解释器和内存空间，因此不受GIL的影响，能够充分利用多核处理器的能力。

#### 具体实现方式

##### 多线程

Python的`threading`模块提供了基本的线程和线程同步支持。使用`Thread`类可以创建线程，通过调用`start()`方法启动线程，`join()`方法等待线程完成。

```python
import threading

def worker(num):
    """thread worker function"""
    print(f'Worker: {num}')
    # 模拟耗时操作
    import time
    time.sleep(2)
    print(f'Done: {num}')

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

##### 多进程

对于多进程，Python提供了`multiprocessing`模块，它提供了一个类似于`threading`模块的API，但是创建的是进程而不是线程。每个进程拥有独立的Python解释器和内存空间，可以绕过GIL的限制。

```python
from multiprocessing import Process

def worker(num):
    """process worker function"""
    print(f'Worker: {num}')
    # 模拟耗时操作
    import time
    time.sleep(2)
    print(f'Done: {num}')

processes = []
for i in range(5):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
```

#### 常用库对比

- **`threading` vs `multiprocessing`**

  - **资源共享**：`threading`中的线程共享同一进程的内存空间，而`multiprocessing`中的进程各自拥有独立的内存空间。这意味着线程间的数据共享更容易，但进程间通信需要额外的工作（如管道、队列等）。
  
  - **性能**：对于I/O密集型应用，多线程通常足够高效；而对于CPU密集型应用，多进程能更好地利用多核资源，提高程序性能。
  
  - **复杂度**：多线程程序相对简单，因为它们共享内存，易于实现简单的数据共享。然而，多进程程序可能需要处理更复杂的IPC（进程间通信）问题。

- **`concurrent.futures` 模块**

  - 这个模块提供了一个高级接口来管理线程池和进程池，简化了并发编程。它包含了`ThreadPoolExecutor`和`ProcessPoolExecutor`两个类，分别用于创建线程池和进程池。

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def worker(num):
    print(f'Worker: {num}')
    import time
    time.sleep(2)
    return f'Done: {num}'

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(worker, i) for i in range(5)]
    for future in futures:
        print(future.result())

with ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(worker, i) for i in range(5)]
    for future in futures:
        print(future.result())
```

#### 结论

选择多线程还是多进程取决于你的具体需求。如果你的应用主要是进行大量的I/O操作，那么使用多线程可能是更好的选择。如果你的应用需要进行大量的计算，或者你想充分利用多核处理器的优势，那么多进程将是更好的选择。`concurrent.futures`模块提供了一种简洁的方式来管理线程和进程，推荐在实际项目中使用。
