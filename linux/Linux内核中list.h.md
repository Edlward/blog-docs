# 一、链表的定义和操作
## 1.1 链表的定义
list.h中对链表进行了定义:
```
struct list_head {
    struct list_head *next, *prev;
};
```
这一不含数据域的双向链表，可以内嵌到任何结构中，比如可以按照以下方式定义含有数据域的链表：

```
struct my_list {
    void *mydata;
    struct list_head list;
}
```
可以看出：
- list域隐藏了链表的指针特性
- struct list_head可以位于结构的任何问题，可以给起任何名字
- 在一个结构中可以有多个list域

## 1.2 链表的声明和初始化宏

```
//仅初始化
#define LIST_HEAD_INIT(name) { &(name), &(name) }

//声明并初始化
#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)
```
如果要申明并初始化自己的链表头mylist_head，则直接调用LIST_HEAD:

```
LIST_HEAD(mylist_head)
```
调用之后,mylist_head的next、prev指针都初始化为指向自己

## 1.3 在链表中增加一个结点
list.h中增加结点的函数为：

```
static inline void list_add();
static inline void list_add_tail();
```
在内核代码中，函数名前加两个下划线表示内部函数，具体代码如下：

```
static inline void
list_add(struct list_head *entry, struct list_head *head)
{
    __list_add(entry, head, head->next);
}

static inline void
__list_add(struct list_head *entry,
                struct list_head *prev, struct list_head *next)
{
    next->prev = entry;
    entry->next = next;
    entry->prev = prev;
    prev->next = entry;
}

static inline void
list_add_tail(struct list_head *entry, struct list_head *head)
{
    __list_add(entry, head->prev, head);
}
```
- __list_add:通过三个结点（新添加结点、前结点、后结点），在前结点和后结点之间添加新结点
- list_add:通过调用__list_add(new, head, head->next),向指定链表的head结点后插入new结点
- list_add_tail:通过调用__list_add(new, head->prev, head),向指定链表的head结点前插入new结点

## 1.4 遍历链表
list.h里定义了遍历链表的宏：

```
#define list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)
```
- 这种遍历找到一个结点在链表中的偏移位置pos

利用pos得到的结点的起始地址：
```
#define list_entry(ptr, type, member) \
    container_of(ptr, type, member)
    
#ifndef container_of
#define container_of(ptr, type, member) \
    (type *)((char *)(ptr) - (char *) &((type *)0)->member)
#endif
```
- 指针ptr指向结构体type中的成员member；
- 通过指针ptr，返回结构体type的起始地址
- (char*)&((type *)0)->member: 获得member在type类型结构体中的偏移量
- ptr减去member偏移量，即得到起始地址

安全遍历：

```
#define list_for_each_safe(pos, n, head) \
    for (pos = (head)->next, n = pos->next; pos != (head); \
        pos = n, n = pos->next)
```

## 1.5 链表删除

```
static inline void list_del(struct list_head *entry)
{
    __list_del_entry(entry);
    entry->next = LIST_POISON1;
    entry->prev = LIST_POISON2;
}

static inline void __list_del_entry(struct list_head *enry)
{
    if (!__list_del_entry_valid(entry)
        return;
    __list_del(entry->prev, entry->next);
}

static inline void __list_del(struct list_head *prev, struct list_head *next)
{
    next->prev = prev;
    prev->next = next;
}
```
- list_del来删除一个结点，被删除的节点的两个指针指向两个固定的位置
- LIST_POISON1和LIST_POISON2是内核空间的两个地址
- 删除结点之前，遍历链表应该用安全遍历，因为删除后若指向的结点不存在则会出现问题

## 二、链表的应用
编写一个内核模块，来创建、增加、删除和遍历一个双向链表：

```
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/list.h>

MODULE_LICENSE("GPL v2");

#define N 10	//number of listNode
struct numlist {
	int num;
	struct list_head list;
};

struct numlist numhead;	//head list

static int __init doublelist_init(void)
{
	// initialize the list node
	struct numlist *listnode;
	struct list_head *pos;
	struct numlist *p;
	int i;

	printk("dounlelist is starting...\n");
	INIT_LIST_HEAD(&numhead.list);

	// create n nodes, and add to list
	for (i = 0; i < N; i++) {
		listnode = (struct numlist *)kmalloc(sizeof(struct numlist), GFP_KERNEL);
		listnode->num = i + 1;
		list_add_tail(&listnode->list, &numhead.list);
		printk("Node %d has added to the doublelist ...\n", i + 1);
	}

	// traversal the list
	i = 1;
	list_for_each(pos, &numhead.list) {
		p = list_entry(pos, struct numlist, list);
		printk("Node %d's data: %d\n", i, p->num);
		i++;
	}

	return 0;
}
module_init(doublelist_init);


static void __exit doublelist_exit(void)
{
	struct list_head *pos, *n;
	struct numlist *p;
	int i;

	//delete n nodes
	i = 1;
	list_for_each_safe(pos, n, &numhead.list) {
		list_del(pos);
		p = list_entry(pos, struct numlist, list);
		kfree(p);
		printk("Node %d has removed from the doublelist...\n", i++);
	}
	printk("doublelist is exiting..\n");
}
module_exit(doublelist_exit);

```



执行结果(日志里）:
```
[ 4390.441568] dounlelist is starting...
[ 4390.441569] Node 1 has added to the doublelist ...
[ 4390.441570] Node 2 has added to the doublelist ...
[ 4390.441570] Node 3 has added to the doublelist ...
[ 4390.441571] Node 4 has added to the doublelist ...
[ 4390.441571] Node 5 has added to the doublelist ...
[ 4390.441572] Node 6 has added to the doublelist ...
[ 4390.441572] Node 7 has added to the doublelist ...
[ 4390.441573] Node 8 has added to the doublelist ...
[ 4390.441573] Node 9 has added to the doublelist ...
[ 4390.441574] Node 10 has added to the doublelist ...
[ 4390.441575] Node 1's data: 1
[ 4390.441575] Node 2's data: 2
[ 4390.441576] Node 3's data: 3
[ 4390.441576] Node 4's data: 4
[ 4390.441577] Node 5's data: 5
[ 4390.441577] Node 6's data: 6
[ 4390.441578] Node 7's data: 7
[ 4390.441578] Node 8's data: 8
[ 4390.441579] Node 9's data: 9
[ 4390.441579] Node 10's data: 10
[ 4403.753440] Node 1 has removed from the doublelist...
[ 4403.753441] Node 2 has removed from the doublelist...
[ 4403.753442] Node 3 has removed from the doublelist...
[ 4403.753442] Node 4 has removed from the doublelist...
[ 4403.753443] Node 5 has removed from the doublelist...
[ 4403.753443] Node 6 has removed from the doublelist...
[ 4403.753444] Node 7 has removed from the doublelist...
[ 4403.753444] Node 8 has removed from the doublelist...
[ 4403.753445] Node 9 has removed from the doublelist...
[ 4403.753445] Node 10 has removed from the doublelist...
[ 4403.753446] doublelist is exiting..
```