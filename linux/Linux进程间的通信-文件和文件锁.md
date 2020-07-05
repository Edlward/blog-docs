IPC（Inter-Process Communication，进程间通信）

系统对文件本身存在缓存机制，使用文件进行IPC的效率在某些多读少写的情况下并不低下
# 1.竞争条件(racing) ##
并发100个进程，约定好一个内容初值为0的文件，每一个进程打开文件读出数字，并且加一写回:

```
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>//提供write(),read(),lseek()函数及相关参数定义
#include <errno.h>
#include <fcntl.h>//提供open()函数及相关参数定义
#include <string.h>
#include <sys/file.h>
#include <wait.h>

#define COUNT 100
#define NUM 64
#define FILEPATH "./count"//文件的路径

int do_child(const char *path)//子进程执行函数
{
	int fd,ret,count;
	char buf[NUM];
	fd = open(path,O_RDWR); //成功返回0值，不成功返回-1,可读可写的方式打开path路径的文件	
	if (fd<0)//若打开失败，打印返回的错误
	{
		perror("open()");
		exit(1);//异常退出
	}
	ret=read(fd,buf,NUM);//成功返回读取字节数，错误返回-1
	if(ret<0)
	{
		perror("read()");
		exit(1);
	}
	buf[ret] = "\0";
	count=atoi(buf);//字符串转换成整型数，如果第一个非空格字符不存在或者不是数字也不是正负号则返回零，否则开始做类型转换，之后检测到非数字(包括结束符 \0) 字符时停止转换，返回整型数
	++count;
	sprintf(buf,"%d",count);//将整型变量count打印成字符串输出到buf中
	lseek(fd,0,SEEK_SET);//将读写位置移到文件开头
	ret= write(fd,buf,strlen(buf));//将buf写进文件中
	close(fd);
	exit(0);//正常退出
}

int main()
{
	pid_t pid;//pid_t实际上就是int，在/sys/types.h中定义
	int count;
	for(count=0;count<COUNT;count++)
	{
		pid=fork();//创建一个子进程进程，在父进程中，fork返回新创建子进程的进程ID；在子进程中，fork返回0；如果出现错误，fork返回一个负值；
	
		if(pid<0)
		{
			perror("fork()");
			exit(1);
		}
		if(pid==0)
		{	
			do_child(FILEPATH);//创建出来的那个新进程执行任务
		}
	}	
	for(count=0;count<COUNT;count++)
		wait(NULL);//等待所有进程退出	
}
```
运行情况：

```
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ echo 0 > count
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
0
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./racing
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
35zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ echo 0 > count
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./racing
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
46zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ echo 0 > count
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./racing
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
57zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ 
```
   理想状态下，文件最后的数字应该是100,因为有100个进程进行了读数，加一，写回操作，实际上每次执行的情况都不一样，都没有达到预期理想结果，造成这一现象的原因是————竞争条件
   
  最开始文件内容是0,假设此时同时打开了多个进程，多个进程同时打开了内容为0的文件，每个进程读到的数都是0,都给0加1并且写1回到文件。每次100个进程执行顺序可能不一样，每次结果也可能不一样，但一定少于产生的实际进程数。
  
把多个执行过程(进程或线程)中访问同一个共享资源，而这些共享资源又无法被多个执行过程存取的程序片段，叫做临界区代码。

通过对临界区代码加锁，解决竞争条件问题:

```
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>//提供write(),read(),lseek()函数及相关参数定义
#include <errno.h>
#include <fcntl.h>//提供open(),close()函数及相关参数定义
#include <string.h>
#include <sys/file.h>//提供flock()函数及相关参数定义
#include <wait.h>

#define COUNT 100
#define NUM 64
#define FILEPATH "./count"//文件的路径

int do_child(const char *path)//子进程执行函数
{
	int fd,ret,count;
	char buf[NUM];
	fd = open(path,O_RDWR); //成功返回0值，不成功返回-1,可读可写的方式打开path路径的文件	
	if (fd<0)//若打开失败，打印返回的错误
	{
		perror("open()");
		exit(1);//异常退出
	}
	ret=flock(fd,LOCK_EX);//LOCK_EX 建立互斥锁定;返回0表示成功，若有错误则返回-1，错误代码存于errno
	if(ret==-1)
	{	
		perror("flock()");
		exit(1);
	}
	ret=read(fd,buf,NUM);//成功返回读取字节数，错误返回-1
	if(ret<0)
	{
		perror("read()");
		exit(1);
	}
	buf[ret] = "\0";
	count=atoi(buf);//字符串转换成整型数，如果第一个非空格字符不存在或者不是数字也不是正负号则返回零，否则开始做类型转换，之后检测到非数字(包括结束符 \0) 字符时停止转换，返回整型数
	++count;
	sprintf(buf,"%d",count);//将整型变量count打印成字符串输出到buf中
	lseek(fd,0,SEEK_SET);//将读写位置移到文件开头
	ret= write(fd,buf,strlen(buf));//将buf写进文件中
	ret=flock(fd,LOCK_UN);//解除锁定
	if(ret==-1)
	{
		perror("flock()");
		exit(1);
	}
	close(fd);
	exit(0);//正常退出
}

int main()
{
	pid_t pid;//pid_t实际上就是int，在/sys/types.h中定义
	int count;：
	for(count=0;count<COUNT;count++)
	{
		pid=fork();//创建一个子进程进程，在父进程中，fork返回新创建子进程的进程ID；在子进程中，fork返回0；如果出现错误，fork返回一个负值；
	
		if(pid<0)
		{
			perror("fork()");
			exit(1);
		}
		if(pid==0)
		{	
			do_child(FILEPATH);//创建出来的那个新进程执行任务
		}
	}	
	for(count=0;count<COUNT;count++)
		wait(NULL);//等待所有进程退出	
}
```
执行情况：

```
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ echo 0 > count
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
0
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./racingn
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ cat count
100zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ 
```
# 2.flock和lockf ##
Linux的文件锁主要有两种：flock和lockf

flock只能对整个文件加锁;
lockf是fcntl系统调用的一个封装，实现了更细粒度的文件锁————记录锁，可以对文件的部分字节上锁;

flock的语义是针对文件的锁;
lockf是针对文件描述符(fd)的锁


:

文件锁程序：

```
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <wait.h>

#define  PATH "./lock"

int main()
{
        int fd;
        pid_t pid;

        fd=open(PATH,O_RDWR|O_CREAT|O_TRUNC,0644);//O_TRUNC:若文件存在并且以可写的方
式打开时，此旗标会令文件的长度清0,存于文件的资料消失
        if (fd<0)
        {
                perror("open()");
                exit(1);
        }

        if(flock(fd,LOCK_EX)<0)//使用flock对其加互斥，或者if(lockf(fd,F_LOCK,0)<0)使用lockf对其加互斥锁

	{		
		perror("flock()");
        	exit(1);
        }
        printf("%d: locked!\n",getpid());//打印表示加锁成功

        pid=fork();
        if(pid<0)
        {
                perror("fork()");
                exit(1);
        }

        if(pid == 0)
        {
                fd=open(PATH,O_RDWR|O_CREAT|O_TRUNC,0644);
                if(fd<0)
                {
                        perror("open()");
                        exit(1);
                }
                if(flock(fd,LOCK_EX)<0)//在子进程中使用flock对同一个文件加互斥锁
                {
                        perror("flock()");
                        exit(1);
		}
                printf("%d: locked!\n",getpid());//打印加锁成功
                exit(0);
        }
        wait(NULL);
        unlink(PATH);//删除指定文件
        exit(0);
}
```
运行情况:

```
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./flock
3544: locked!
```
子进程flock/lockf的时候阻塞


两种锁之间互不影响，比如以下例子
----------------

```
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <fcntl.h>

#define PATH "./lock"
int main()
{
        int fd;
        pid_t  pid;
        if(open("PATH",O_RDWR|O_CREAT|O_TRUNC,0644)<0)
        {
                perror("open()");
                exit(1);
        }
        if(flock(fd, LOCK_EX)<0)
        {
                perror("flock()");
                exit(1);
        }
        printf("%d: Locked with flock\n",getpid());

        if(lockf(fd, F_LOCK, 0)<0)
        {
                perror("lockf()");
                exit(1);
        }
        printf("%d: Locked with lockf\n", getpid());
        exit(0);
}
```
执行情况如下：

```
zach@zach-i16:~/文档/note/Linux/进程通信/2.文件和文件锁$ ./lockf_and_flock
4162: Locked with flock
4162: Locked with lockf
```
# 3.标准IO库文件锁##

```
#include <stdio.h>

void flockfile(FILE *filehandle);
int ftrylockfile(FILE *filehandle);
void funlockfile(FILE *filehandle);
```
stdio库中实现的文件锁与flock或lockf有本质区别,标准IO的锁在多进程环境中使用是有问题的.

件锁只能处理一个进程中的多个线程之间共享的FILE 的进行文件操作.

多个线程必须同时操作一个用fopen打开的FILE 变量，如果内部自己使用fopen重新打开文件，那么返回的FILE *地址不同，起不到线程的互斥作用

##4.小结
本次Linux进程间通信的学习是基于以下文章和书籍

> [穷佐罗的Linux书-Linux的进程间通信-文件和文件锁](http://liwei.life/2016/07/31/file_and_filelock/)
> 《UNIX环境高级编程》
> [ linux中fcntl()、lockf、flock的区别](http://blog.chinaunix.net/uid-28541347-id-5678998.html)