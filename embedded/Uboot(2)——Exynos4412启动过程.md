
# 一、 iROM
首先arm的pc指针从0x0地址开始执行，打开4412手册可以看到0x0000_0000地址存放着iROM：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/4.png)

所以4412上电后会到iROM中去执行，iROM是4412出厂时就固化的一段程序，这段程序提供了执行arm代码的基本环境，并且从SD/MMC,eMMC4.3,eMMC4.4或NAND中下载BL1代码并检查下载的BL1的完整性

iROM的执行流程：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/5.png)
- Disable watchdog：关闭看门狗(可以不用喂狗操作，默认uboot程序没有很复杂不需要看门狗)
- Disable IRQ's and MMU：关闭中断和MMU（因为需要对IRQ和MMU进行配置才使用，所以先关闭）
- Disable D-cache, Enable I-cache：关闭数据cache，使能指令cache
- Flush TLB's and Invalidate caches：刷新TLB并失效caches
- Make CORE1 idle：使其他核进入空闲模式，只留下CPU0
- Deep-stop or AFTR：这里进行分支判断是否是待机唤醒，如果是唤醒则直接跳转到BL1中，否则继续
- Initalize Stack(IRQ, SVC)：设置IRQ和SVC模式的栈空间(通过片内的ram来设置栈地址，0x0202_0100)
- Initialize ZI/RW：初始化iROM程序中的一些变量(RW(read/write)是程序中的已初始化变量；ZI(zero)是程序中的未初始化的变量)
- Register the function pointers：导出部分核心函数(应该是可以在BL1程序中使用)
- Get the reset status：获取复位状态
- Set the clock divider & Set PLLs：设置时钟分频

    可以从下图中看到4412SOC中有一个大小为256KB的片内ram，上电就可以使用不需要初始化：
  
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/6.png)
    
    内存描述表中可以看到这个片内ram的起始地址是0x0202_0000 ~ 0x0206_0000
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/7.png)
    
    可以从下图看出更具体的片内内存情况：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/8.png)
    - 在地址0x0202_0000 ~ 0x0202_0100内放着Product_ID, iRom_Version, Function_ptr的信息
    - 上面iROM中设置栈地址即在0x0202_0100 ~ 0x0202_0800（1.75KB）
    - 在iROM中初始化的变量地址放在0x0202_0800 ~ 0x0202_1400（3KB）
    - 不去看0x0202_1400上面的部分，这里是Exynos4212的图，4412的BL1是15KB
- Get bootmode (OM pin):获得OM Pin的值，根据值选择从不同外设启动：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/9.png)
- Fail download...:进行BL1完整性等检查，如果检查都通过则进入BL1中执行    

    根据下图BL1的结构组成图可以看到，BL1包括了header+Encrypted BL1 Binary+signature：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/10.png)

    Check Sum和Verify BL1的操作就是通过在BL1的Header里放着checksum value，iROM通过计算校验和并和Header里的checksum value进行比较。
    
    Decrypt BL1是通过密钥来解密镜像，使用 RSA（Rivest-Shamir-Adleman）公钥加密算法；AES（高级加密标准）块加密算法


**总结：iROM从出厂开始固化在SOC上，iROM设置相关程序运行环境（关闭看门狗，关闭中断和MMU，失效数据cache，使能指令cache，设置IRQ和SVC模式的栈，设置时钟，初始化相关变量），并根据OM引脚确定启动设备，将BL1从设备中读出并存入片内ram，最后进入BL1中执行。**


# 二、 BL1 & BL2
## 2.1 BL1

在上一篇文章[U-Boot(1)——编译分析](http://39.106.181.170:8080/article?Path=Uboot%281%29%e2%80%94%e2%80%94%e7%bc%96%e8%af%91%e5%88%86%e6%9e%90)中用来编译U-Boot的编译脚本build_uboot.sh中最后生成u-boot-iTOP-4412.bin时通过把多个二进制文件组合起来得到(包括生成的u-boot.bin),执行语句：
```
cat E4412_N.bl1.SCP2G.bin bl2.bin all00_padding.bin u-boot.bin tzsw_SMDK4412_SCP_2GB.bin
```
可以从三星给的文档中找到对应的信息：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/11.png)
- bootloader共由4个部分组成：
    - BL1 == E4412_N.bl1.SCP2G.bin(15KB)
    - BL2 == bl2.bin(14kB) + all00_padding.bin(2kB)
    - U-boot == u-boot.bin(328KB)
    - TZSW == tzsw_SMDK4412_SCP_2GB.bin(156KB)

由下面的Exynos4412的片内内存映射图可以看到大小为15KB的BL1程序被放在0x0202_1400 ~ 0x0202_4c00地址中运行：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/12.png)

BL1也是由三星厂家提供，只有二进制文件，没有源码，下图是BL1的执行流程图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/13.png)
- Initialize IRQ and SVC_stack：初始化IRQ和SVC栈
- Lowpwr-Audio wakeup：是否是这个Lowpwr-Audio唤醒操作，如果是就直接跳转到BL2中去执行，否则继续执行
- Boot device：判断从哪个设备上启动的(SDMMC,eMMC4.3等等)
- Sleep wakeup：是否是sleep唤醒操作，如果是就直接跳转到BL2，否则继续执行
- Secure boot & Verify：进行安全性检查和判断完整性，这里根据下图看到BL2其实包含了BL2 Binary+Checksum+Signature+Padding，根据Signature去进行安全检查，根据Checksum去判断完整性：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/14.png)

## 2.2 BL2
在BL1中对BL2进行检查成功后跳转到BL2执行，具体执行流程图如下：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/15.png)
相比较BL1而言，BL2多出了执行了SET CLOCK's(设置时钟)和Initialize DRAM(初始化DRAM)的工作，转到之后的OS中去执行

