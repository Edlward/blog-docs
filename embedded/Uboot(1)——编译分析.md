硬件选用讯为ITOP-4412开发板，板栽三星的Exynos4412芯片
# 一、编译入手
根据从官网上下载来的U-boot源码中README文件中可以得到信息，如果要使用开发板board/<board_name>，应该先执行"make <board_name>_config"命令进行配置，随后执行"make all"生成编译后文件

所以可以得出编译的主要两步：
1. make <board_name>_config进行配置
2. make all得到编译后目标文件

以这两步为核心来看《ITOP-4412开发板之精英版使用手册》中编译uboot生成镜像的方法：

```
./build_uboot.sh SCP_1GDDR
```

在三星原厂手册《SEC_Exynos4x12_[SSCR][TC4]ICS_Installation_Guide_RTM1.0.2》中看到对应的方式：
> [1]: enter the directory TC4_uboot
> [2]: execute the command below for u-boot compile
> For TC4Plus board:
> ./build_uboot.sh tc4_plus

因此，可以看出build_uboot.sh这个脚本是三星提供的编译uboot镜像的脚本


# 二、 build_uboot.sh
在三星提供的TC4的uboot源码中的build_uboot.sh脚本中写着大概的执行步骤，在讯为提供的build_uboot.sh中又加入了对应的SCP和POP以及Android和Linux文件系统的相关配置代码

执行 **./build_uboot.sh SCP_1GDDR**的步骤：


![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/1.png)

通过build_uboot.sh这一脚本完成：
1. "make <board/name>_config"配置编译
2. "make all"编译uboot镜像
3. 将多个二进制文件合并成最终uboot镜像(这是由于4412启动的分层机制，由固定在芯片中的出厂程序iROM引导BL1、BL2、Uboot，这里在之后文章会详细分析)

## itop_4412_android_config_scp_1GDDR

根据刚才对build_uboot.sh执行步骤进行的分析，会执行指令
```
make itop_4412_android_config_scp_1GDDR
```
在Makefile中对应语句为：
```
itop_4412_android_config_scp_1GDDR:		unconfig
	@$(MKCONFIG) $(@:_config=) arm arm_cortexa9 smdkc210 samsung s5pc210 SCP_1GDDR
```
- unconfig的规则，可以看到是删除了许多的配置文件：
    ```
    unconfig:
    	@rm -f $(obj)include/config.h $(obj)include/config.mk \
    		$(obj)board/*/config.tmp $(obj)board/*/*/config.tmp \
    		$(obj)include/autoconf.mk $(obj)include/autoconf.mk.dep
    ```
- $(MKCONFIG)...，是执行mkconfig... ：
```
SRCTREE		:= $(CURDIR)
MKCONFIG	:= $(SRCTREE)/mkconfig
```
- $(@:_config=)，等于把目标文件名称的_config替换成“”，即itop_4412_android_scp_1GDDR

# 三、 mkconfig
在build_uboot.sh中执行：
```
make itop_4412_android_config_scp_1GDDR
```
转到Makefile中实际执行：

```
mkconfig itop_4412_android_scp_1GDDR arm arm_cortexa9 smdkc210 samsung s5pc210 SCP_1GDDR
```
这条命令在mkconfig执行步骤为：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/2.png)

# 四、 Makefile
之前分析的在build_uboot.sh中执行：
```
mkconfig itop_4412_android_scp_1GDDR arm arm_cortexa9 smdkc210 samsung s5pc210 SCP_1GDDR
```
之后执行：

```
make -j$CPU_JOB_NUM
```
接下来分析一下Makefile的执行步骤：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/3.png)


