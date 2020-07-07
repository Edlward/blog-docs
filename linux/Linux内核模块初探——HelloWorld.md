本文通过学习宋宝华老师的《Linux设备驱动开发详解》第四章而写的学习笔记
## 一、为什么要有内核模块

因为Linux内核架构庞大，组件很多，如果我们把所有需要功能都编译到Linux内核中，就会导致内核很大，并且当我们要在现有内核中添加或删除功能时都要重新编译内核。

Linux使用了模块（Module）这一种机制，模块不用编译进内核映像，模块可以被加载和卸载，如果被加载就和内核中其他部分一样。

## 二、内核模块的程序结构
1. 模块加载函数
        
    当通过insmod或modprobe命令加载内核模块时，模块的加载函数会自动被内核执行，完成本模块的相关初始化工作
2. 模块卸载函数
    
    当通过rmmod命令卸载某模块时，模块的卸载函数会自动被内核执行，完成与模块卸载函数相反的功能
3. 模块许可证明
        
    许可证（LICENSE)声明描述内核模块的许可权限，如果不声明LICENSE,模块被加载时，将会收到内核被污染（Kernel Tainted)的警告
    
    内核模块领域，可接受的LICENSE包括“GPL"、"GPL v2"、"GPL and additional rights"、"Dual BSD/GPL"、"Dual MPL/GPL"
4. 模块参数（可选）

    模块参数是模块被加载时候可以传递给它的值，它本身对应模块内部的全局变量
5. 模块导出符号（可选）

    内核模块可以导出的符号(symbol，对应于函数或变量)，若导出，其他模块则可以使用本模块中的变量或函数
6. 模块作者等信息声明（可选）

## 三、“Hello World"内核模块
hello.c:

```
#include <linux/init.h>
#include <linux/module.h>

static int __init hello_init(void)
{
	printk(KERN_INFO "Hello World enter\n");
	return 0;
}
module_init(hello_init);

static void __exit hello_exit(void)
{
	printk(KERN_INFO "Hello World exit\n");
}
module_exit(hello_exit);

MODULE_LICENSE("GPL v2");
```
- 这个Hello World内核模块只包含内核模块加载函数、卸载函数和对GPL v2许可权限的声明
- 加载函数一般以 __init 标识声明
/include/linux/init.h：
```
#define  __init  __attribute__ ((__section__ (".init.text"))) __cold
```
__init的函数如果直接编译进内核，成为内核镜像的一部分，在连接的时候都会放在.init.text这个区段内
- 模块加载函数以 "module_init(函数名)"的形式被指定。初始化成功则返回0；失败则返回错误编码
- 卸载函数一般以 __exit标识声明
- 模块卸载函数在模块卸载的时候执行，不返回任何值，以 "module_exit(函数名)"的形式指定


## 四、编译、加载、卸载
Makefile：

```
KVERS = $(shell uname -r)

obj-m += hello.o

kernel_modules:
	make -C /lib/modules/$(KVERS)/build M=$(CURDIR) modules

clean:
	make -C /lib/modules/$(KVERS)/build M=$(CURDIR) clean
```
### 4.1 编译模块：
![这里写图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/linux/img/1.png)

### 4.2 加载模块：
使用命令insmod加载hello.ko

```
sudo insmod ./hello.ko
```
加载完后使用命令lsmod可以看到hello模块已经被加载到系统中：
![这里写图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/linux/img/2.png)

### 4.3 卸载模块：
使用命令rmmod命令卸载hello模块，卸载完后再用lsmod命令查看hello模块已经不在：
![这里写图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/linux/img/3.png)

### 4.4 查看输出
我们在加载和卸载时候，在控制台并没有看到 "Hello World"的输出，这是因为我们使用的printk给的参数(KERN_INFO),使得日志级别没有控制台要求输出的最低级别高

使用下述语句查看日志：

```
cat /var/log/syslog | grep Hello
```
就可以看到我们加载和卸载hello模块的输出信息了：
![这里写图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/linux/img/4.png)