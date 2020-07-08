# 一、引子
在上一篇文章 [Uboot(2)——Exynos4412启动过程](http://39.106.181.170:8080/article?Path=Uboot%282%29%e2%80%94%e2%80%94Exynos4412%e5%90%af%e5%8a%a8%e8%bf%87%e7%a8%8b)中可以看到，BL2程序流程图中有SET CLOCK's这一步骤， 并且在U-Boot源码的 **board/samsung/smdkc210/lowlevel_init_SCP.S**的文件中也有着对时钟初始化的相关代码(因为BL2中已经初始化，所以不一定会执行)

这篇文章通过U-Boot中对时钟进行初始化的源码和三星提供的4412的datasheet来学习Exynos 4412的时钟体系结构和时钟的相关操作

# 二、Exynos 4412的时钟体系结构

在Exynos 4412的datasheet里第七章“时钟管理单元(Clock Management Unit)”详细介绍了4412的时钟体系结构


## 2.1 时钟域

首先可以看一下Exynos 4412 SCP的总框图：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/16.png)
- 从总框图可以看到4412内不仅有着多核处理器，而且有着音频视频接口、GPS、内存管理模块等等片上资源
- 而这些不同的模块工作频率也大多不同，所以一个时钟显然不方便去满足所有的工作频率，所以在4412中采用了多个时钟域

下图是Exynos 4412 SCP的时钟域图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/17.png)
- CPU_BLK：包含Cortex-A9 MPCore处理器，L2 cache控制器和CoreSight；CMU_CPU(即CPU模块内的时钟管理单元)用来给CPU block中部件产生时钟
- DMC_BLK：包含内存控制器(DMC)、安全子系统(SSS),中断控制器(GIC)；CMU_DMC为这些部件产生时钟
- LEFTBUS_BLK and RIGHTBUS_BLK：全局数据总线，用来在DRAM和子功能模块之间传播数据
- function blocks：包括G3D, MFC, LCD0, ISP, CAM, 
TV, FSYS, MFC, GPS, MAUDIO, PERIL, 和PERIR；这些功能模块的时钟由CMU_TOP产生

## 2.2 时钟源
在4412的datasheet中可找到

> The top-level clocks in Exynos 4412 SCP are:
> - Clocks from clock pads, namely, XRTCXTI, XXTI, and XUSBXTI.
> - Clocks from CMUs
> - Clocks from USB PHY
> - Clocks from HDMI_PHY
> - Clocks from GPIO pads

从第一个"clocks from clock pads..."可以看到顶层的时钟包括着三个时钟引脚，分别是:
- XRCXTI：接32.768KHz的晶振，用于RTC
- XXTI：接12MHz ~ 50MHz的晶振，用于测试，可以不接但要接GND
- XUSBXTI：需要接24MHz的晶振，向系统提供时钟输入源

打开讯为提供的Exynos 4412的原理图，可以看到XUSBXTI接了24MHz的晶振，XXTI接地：


![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/18.png)


## 2.3 PLL（锁相环）
刚才可以看到XUSBXTI接了24MHz的晶振作为系统时钟源的输入，但是4412CPU的频率可以达到1.4GHz，所以肯定需要相应的部件把24MHz的频率提升到1.4GHz

PLL（锁相环）使用一个外部晶振作为输入，可以对外部晶振所产生的频率进行倍频或分频操作

4412一共有4个PLL：
- APLL：用于给CPU_BLK提供时钟；也可以作为MPLL的补充，给DMC_BLK、LEFTBUS_BLK、RIGHTBUS_BLK和CMU_TOP提供时钟；使用FINPLL的输入产生22~1400MHz的时钟频率
- MPLL用于给DMC_BLK、LEFTBUS_BLK、RIGHTBUS_BLK和CMU_TOP提供时钟：使用FINPLL的输入产生22~1400MHz的时钟频率
- EPLL：主要给音频模块提供时钟；使用FINPLL的输入产生22~1400MHz的时钟频率；给音频子系统产生一个192MHz的时钟
- VPLL：使用FINPLL或SCLK_HDMI24M的输入产生22~1400MHz的时钟频率；给视频系统提供54MHz的时钟或给G3D（3D图形加速器）提供时钟

# 三、U-Boot源码&时钟初始化
在**board/samsung/smdkc210/lowlevel_init_SCP.S**的文件中，标号system_clock_init_scp中对应代码是时钟初始化的部分

首先执行：

```
push	{lr}
ldr	r0, =ELFIN_CLOCK_BASE	@0x1003_0000
```
- 因为是用指令bl来跳转到这个标号(bl system_clock_init_scp)，所以进这个标号先把lr指令先压栈
- 把时钟寄存器基地址加载到r0，基址地址为0x1003_0000

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/19.png)

接下选择CMU_CPU的时钟源:
```
@ CMU_CPU MUX / DIV
	ldr	r1, =0x0
	ldr	r2, =CLK_SRC_CPU_OFFSET
	str	r1, [r0, r2]

	ldr r2, =CLK_MUX_STAT_CPU_OFFSET
	ldr r3, =0x01110001
	bl wait_mux_state

```
- 首先根据CLK_SRC_CPU_OFFSET的值为0x14200，得带其实是把CLK_SRC_CPU的值设为0，根据时钟图可以看到，把这个寄存器内容写为0其实是选择外部输入的24MHz作为CMU_CPU的时钟源（MUX起到选择器的作用，用来在两个输入中选择一个输出）
    ```
    MUX_APLL_SEL = 0 = FINPLL
    MUX_CORE_SEL = 0 = MOUTAPLL
    剩下两个MUX保持复位状态0
    ```
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/20.png)
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/21.png)
- 再根据CLK_MUX_STAT_CPU_OFFSET的值为0x14400，查看CLK_MUX_STAT_CPU寄存器的值是否为0x01110001（可以看到这个寄存器是只读的）
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/22.png)
    如果是正确把CLK_SRC_CPU寄存器的值设置为0，则有：
    ```
    MPLL_USER_SEL_C = FINMPLL =1
    HPM_SEL = MOUTAPLL = 1
    CORE_SEL = MOUTAPLL =1
    ARLL_SEL = FINPLL = 1
    此寄存器值为0x0111_0001
    ```
- wait_mux_state做的工作就是把以r0为基址r2为偏移地址的寄存器中的值和r3的值比较，如果不相等则一直循环等待：
    ```
    wait_mux_state:
    	ldr r1, [r0, r2]
    	cmp r1, r3
    	bne wait_mux_state
    	mov pc, lr
    ```
随后设置了CMU_DMC的相关分频比
```
	ldr	r1, =CLK_DIV_DMC0_VAL
	ldr	r2, =CLK_DIV_DMC0_OFFSET
	str	r1, [r0, r2]
	ldr	r1, =CLK_DIV_DMC1_VAL
	ldr	r2, =CLK_DIV_DMC1_OFFSET
	str	r1, [r0, r2]
```
- 这里为什么不设置CMU_CPU的分频比呢？是因为CMU_CPU中各部件直接使用了默认分频值（即不分频，CLK_DIV_CPU0和CLK_DIV_CPU1寄存器都默认为0）
- 此时也没有设置CMU_DMC的时钟源选择，根据下图可以看到，在CMU_DMC的时钟域这些部件可以选择APLL或MPLL作为时钟源输入
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/23.png)
- 可以根据CLK_SRC_DMC寄存器看出：按照默认时钟源选择，是选择MPLL的时钟源输入，所以不需要设置CMU_DMC时钟域里面的时钟源
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/24.png)
- 根据CLK_DIV_DMC0_OFFSET的值为0x10500，设置CLK_DIV_DMC0寄存器的值为CLK_DIV_DMC0_VAL=0x1311_1113
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/25.png)
    ```
    #define CORE_TIMERS_RATIO	0x1
    #define COPY2_RATIO		0x3
    #define DMCP_RATIO		0x1
    #define DMCD_RATIO		0x1
    #define DMC_RATIO		0x1
    #define DPHY_RATIO		0x1
    #define ACP_PCLK_RATIO		0x1
    #define ACP_RATIO		0x3
    #define CLK_DIV_DMC0_VAL	((CORE_TIMERS_RATIO << 28) \
    				| (COPY2_RATIO << 24)   \
    				| (DMCP_RATIO << 20)	\
    				| (DMCD_RATIO << 16)	\
    				| (DMC_RATIO << 12)	\
    				| (DPHY_RATIO << 8)	\
    				| (ACP_PCLK_RATIO << 4)	\
    				| (ACP_RATIO))
    ```
- 根据CLK_DIV_DMC1_OFFSET的值为0x10504，设置CLK_DIV_DMC1寄存器的值为CLK_DIV_DMC1_VAL=0x0101_0100
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/26.png)
    ```
    #define DPM_RATIO	        0x1
    #define DVSEM_RATIO	        0x1
    #define PWI_RATIO	        0x1
    #define CLK_DIV_DMC1_VAL	((DPM_RATIO << 24) \
    				| (DVSEM_RATIO << 16) \
    				| (PWI_RATIO << 8))
    ```

接下来分别对 CMU_TOP、CMU_LEFTBUS、CMU_RIGHTBUS的时钟源和分频比进行了设置：

```
@ CMU_TOP MUX / DIV
	ldr	r1, =0x0
	ldr	r2, =CLK_SRC_TOP0_OFFSET
	str	r1, [r0, r2]

	ldr r2, =CLK_MUX_STAT_TOP_OFFSET
	ldr r3, =0x11111111
	bl wait_mux_state

	ldr	r1, =0x0
	ldr	r2, =CLK_SRC_TOP1_OFFSET
	str	r1, [r0, r2]

	ldr r2, =CLK_MUX_STAT_TOP1_OFFSET
	ldr r3, =0x01111110
	bl wait_mux_state

	ldr	r1, =CLK_DIV_TOP_VAL
	ldr	r2, =CLK_DIV_TOP_OFFSET
	str	r1, [r0, r2]

@ CMU_LEFTBUS MUX / DIV
	ldr	r1, =0x10
	ldr	r2, =CLK_SRC_LEFTBUS_OFFSET
	str	r1, [r0, r2]

	ldr r2, =CLK_MUX_STAT_LEFTBUS_OFFSET
	ldr r3, =0x00000021
	bl wait_mux_state

	ldr	r1, =CLK_DIV_LEFRBUS_VAL
	ldr	r2, =CLK_DIV_LEFTBUS_OFFSET
	str	r1, [r0, r2]

@ CMU_RIGHTBUS MUX / DIV
	ldr	r1, =0x10
	ldr	r2, =CLK_SRC_RIGHTBUS_OFFSET
	str	r1, [r0, r2]

	ldr r2, =CLK_MUX_STAT_RIGHTBUS_OFFSET
	ldr r3, =0x00000021
	bl wait_mux_state

	ldr	r1, =CLK_DIV_RIGHTBUS_VAL
	ldr	r2, =CLK_DIV_RIGHTBUS_OFFSET
	str	r1, [r0, r2]
```
- 由于根CMU_CPU的 MUX/DIV操作都很类似，故省略

接下来设置Lock Time：配置完PLL锁相环后会有一段时钟频率为0的空挡，这段时间CPU不工作，这段时间称为Lock Time
```
@ Set PLL locktime
	ldr	r1, =APLL_LOCK_VAL
	ldr	r2, =APLL_LOCK_OFFSET
	str	r1, [r0, r2]

	ldr	r1, =MPLL_LOCK_VAL
	ldr	r2, =MPLL_LOCK_OFFSET
	str	r1, [r0, r2]

	ldr	r1, =EPLL_LOCK_VAL
	ldr	r2, =EPLL_LOCK_OFFSET
	str	r1, [r0, r2]

	ldr	r1, =VPLL_LOCK_VAL
	ldr	r2, =VPLL_LOCK_OFFSET
	str	r1, [r0, r2]

	ldr	r1, =CLK_DIV_CPU0_VAL
	ldr	r2, =CLK_DIV_CPU0_OFFSET
	str	r1, [r0, r2]
	ldr	r1, =CLK_DIV_CPU1_VAL
	ldr	r2, =CLK_DIV_CPU1_OFFSET
	str	r1, [r0, r2]
```
- 可以找到APLL_LOCK_VAL和APLL_LOCK_OFFSET的值：
    ```
    /* APLL_LOCK		*/
    #define APLL_LOCK_VAL	0x000002F1
    #define APLL_LOCK_OFFSET		0x14000		
    ```
    所以是向APLL_LOCK寄存器中写入0x0000_02F1
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/27.png)
    寄存器描述最大PLL的lock time是22.5us，此时FIN是24MHz，PDIV是2，PLL_LOCKTIM=540=0x21C，这里写入的时间应该是比最大时间大就行，所以写入0x2F1>0x21C
- MPLL、EPLL和VPLL的Lock Time设置跟APLL类似，不再赘述
- 随后设置分频系数，分别向CLK_DIV_CPU0和CLK_DIV_CPU1两个寄存器中输入0x0114_3730和0x4

接下来设置锁相环:
```
@ Set APLL
	ldr	r1, =APLL_CON1_VAL
	ldr	r2, =APLL_CON1_OFFSET
	str	r1, [r0, r2]
	ldr	r1, =APLL_CON0_VAL
	ldr	r2, =APLL_CON0_OFFSET
	str	r1, [r0, r2]

```
- APLL,MPLL,EPLL,VPLL设置方法类似，所以这里只看APLL即可
- 向APLL_CON1写入0x0080_3800，可以看第22位BYPASS，如果置1则APLL输出直接是FIN的输入，所以设置为0输出为倍频后的时钟：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/28.png)
- 向APLL_CON0写入APLL_CON0_VAL，先来看一下APLL_CON0寄存器的描述：
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/29.png)
    31bit用来使能和失效APLL，如果要使用肯定是置1；16 ~25bit，8 ~13bit，0 ~2bit分别代表着M,P,S三个值，而这三个值用来配置具体输出频率(根据公式)
    
    ```
    FOUT = M * FIN / (P * 2^S)
    ```
    下表是一些推荐的配置(可以验证下上面的公式）即可：
   ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/30.png)
    U-boot源码中传入参数是M=200，P=6，S=1，得到FOUT = 400
- APLL_CON0的29bit用来读出PLL的锁定状态，如果是1表示已锁定(即PLL输出稳定)，所以在设定完APLL后跳转到wai_pll_lock标号循环等待是否稳定：
    ```
    ldr r2, =APLL_CON0_OFFSET
    	bl wait_pll_lock
    
    wait_pll_lock:
    	ldr r1, [r0, r2]
    	tst r1, #(1<<29)
    	beq wait_pll_lock
    	mov pc, lr
    ```

在设置好PLL之后，此时设置MUX都选择PLL输出的时钟频率为输入，以CMU_CPU选择为例(CMU_DMC,CMU_TOP等操作相似)：
- 还是之前看到过的时钟图，这回要选择从APLL输出的系统时钟频率，所以:
    ```
    MUX_APLL_SEL = 1 = MOUTAPLLFOUT
    MUX_CORE_SEL = 0 = MOUTAPLL
    ```
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/31.png)
    ![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/embedded/img/32.png)

### 时钟初始化总结：
1. 对各时钟域的时钟源进行选择，并且设置分频比（不能同时使用PLL时钟源和设置PLL，所以先保证没有使用PLL时钟源）
2. 设置Lock Time（向对应PLL的LOCK寄存器写入值大于cpu暂停时间即可）
3. 设置锁相环（设置PLL的M,P,S值；设置其他参数，使能）
4. 重新对时钟域的时钟源进行选择，选择PLL输出的时钟(设置MUX)