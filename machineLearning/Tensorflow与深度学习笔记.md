
本文知识来源于《TensorFlow实战Goolgle深度学习框架》的三四章
# 一、计算图——Tensorflow的计算模型
## 1.1 计算图的概念
TensorFlow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系

- 两个向量相加样例的计算图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/1.png)

## 1.2 计算图的使用
Tensorflow程序一般可以分为两个阶段：
1. 第一个阶段需要定义计算图中所有的计算
2. 第二个阶段为执行阶段

定义阶段的样例：

```
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
```


- 在TensorFlow程序中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图
- 除了使用默认的计算图，TensorFlow支持通过tf.Graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享
- 在一个计算图中，可以通过集合(collection)来管理不同类别的资源。比如通过tf.add_to_collection函数可以将资源加入一个或多个集合中，然后通过tf.get_collection获取一个集合里面的所有资源

# 二、张量——TensorFlow的数据模型
在TensorFlow程序中，所有数据都通过张量(tensor)的形式来表示

例如：
```
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print result
##############################
# 输出：
# Tensor("add:0", shape=(2,), dtype=float32)
```
一个张量中主要保存了三个属性：名字、维度和类型

# 三、会话——TensorFlow的运行模型
会话(session)拥有并管理TensorFlow程序运行时的所有资源。

当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露的问题

使用TensorFlow中的会话来执行定义好的运算：
1. 第一种模式需要明确调用会话生成函数和关闭会话函数：

```
# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算的结果。
# 比如可以调用sess.run(result)来的到张量result的取值
sess.run(...)
# 关闭会话使得本次运行中使用到的资源可以被释放
sess.close()
```

2. TensorFlow可以通过Python的上下文管理器来使用会话：

```
# 创建一个会话，并通过Python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    # 使用这个创建好的会话来计算关心的结果
    sess.run(...)
# 不需要再调用"Session.close()"函数来关闭对话
# 当上下文退出时会话关闭和资源释放也自动完成了
```
# 四、神经网络相关
使用神经网络解决分类问题主要分为4个步骤
1. 提取问题中实体的特征向量作为神经网络的输入
2. 定义神经网络的结构，并定义如何从神经网络的输入得到输出。（神经网络的前向传播算法）
3. 通过训练数据来调整神经网络中参数的取值，这就是训练神经网络的过程
4. 使用训练好的神经网络来预测未知的数据

## 4.1 全连接网络结构的前向传播算法
神经元结构示意图：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/2.png)
- 一个神经元有多个输入和一个输出，每个神经元的输入既可以是其他神经元的输出，也可以是整个神经网络的输入
- 所谓神经网络的结构就是指的不同神经元之间的连接结构
- 如图所示，一个最简单的神经元结构的输出就是所有输入的加权和，而不同输入的权重就是神经元的参数

全连接即相邻两层之间任意两个节点之间都有连接

三层神经网络结构图：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/3.png)

全连接的神经网络前向传播算法示意图：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/4.png)

可以通过矩阵乘法得到隐藏层三个节点所组成的向量取值：

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/5.png)

类似的输出层也可以用矩阵乘法表示：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/6.png)

TensorFlow中矩阵乘法使用tf.matmul函数实现，上述矩阵乘法可以写成：

```
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
```
## 4.2 TensorFlow变量
TensorFlow变量声明函数tf.Variable：

```
#变量w为：一个2x3的矩阵，矩阵中元素是均值为0，标准差为2的随机数
w = tf.Variable(tf.random_normal([2, 3], stddev = 2))

#变量b为： 初始值全部为0且长度为3的变量
b = tf.Variable(tf.zeros[3])
```

初始化所有变量：

```
init_op = tf.initialize_all_variables()
sess.run(init_op)
```
## 4.3 通过TensorFlow训练神经网络模型

神经网络反向传播优化流程图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/7.png)

- 反向传播算法实现了一个迭代的过程。
- 在每次迭代的开始，首先需要选取一小部分训练数据，这一小部分数据叫做一个batch
- 之后，这个batch的样例会通过前向传播算法得到神经网络模型的预测结果
- 将预测答案与正确答案进行比较，计算出差距
- 最后根据预测值和真实值之间的差距，反向传播算法会相应更新神经网络参数的取值，使得在这个batch上神经网络模型的预测结果和真实答案更加接近

**placeholder机制是TensorFlow中用于提供输入数据的，相当于定义了一个位置，这个位置中的数据在程序运行时再指定**

通过placeholder实现前向传播算法的代码：

```
import tensorflow as tf

w1 = tf.Variable(tf.random_normal[2, 3], stddev = 1))
w2 = tf.Variable(tf.random_normal[3, 1], stddev = 1))

# 定义placeholder作为存放输入数据的地方
# 这里维度也不一定要定义
# 但如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape = (1, 2), name = "inout")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

print(sees.run(y, feed_dict = {x: [[0.7, 0.9]]}))
```
- feed_dict来指定x的取值
- 上述例子值计算了一个样例的前向传播结果，如果每次提供一个batch的训练样例，将输入的1x2矩阵改为nx2的矩阵，就可以得到n个样例的前向传播结果了

```
x = tf.placeholder(tf.float32, shape=(3, 2), name = "input")

print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
```

在得到一个batch的前向传播结果之后，需要定义一个损失函数来刻画当前的预测值和真实答案之间的差距。

定义一个简单的损失函数，并通过TensorFlow定义了反向传播算法：

```
# 定义损失函数来刻画预测值与真实值得差距
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
```
## 4.4 通过激活函数实现去线性化

如果将每一个神经元(也就是神经网络中的节点)的输出通过一个非线性函数，那么整个神经网络的模型也就不再是线性的了。这个非线性函数就是激活函数。

加入偏置项和激活函数的神经元结构示意图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/8.png)

常用的神经网络激活函数的函数图像：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/9.png)

TensorFlow实现三层全相连的前向传播算法，加入偏置项和激活函数：

```
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)
```
## 4.5 损失函数
损失函数被用来测试神经网络模型的效果，即用来判定预测结果和正确结果之间差距，并且通过差距来优化参数

交叉熵用来刻画两个概率分布之间的距离，给定两个概率分布p和q，通过q来表示p的交叉熵为：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/10.png)

Softmax用来把输出的数据映射成为(0, 1)的值，而这些值的累和为1。

假设有一个三分类问题，某个样例的正确答案是(1,0,0)。
某模型经过Softmax回归之后的预测答案是(0.5,0.4,0.1)，那么这个预测和正确答案之间的交叉熵为：

```math
H((1,0,0),(0.5,0.4,0.1)) = -(1*log0.5+0*log0.4+0*log0.1)≈0.3
```
如果另外一个模型的预测是(0.8,0.1,0.1)，那么这个测试值和真实值之间的交叉熵是：
```math
H((1,0,0),(0.8,0.1,0.1)) = -(1*log0.8+0*log0.1+0*log0.1)≈0.1
```
通过TensorFlow实现交叉熵：

```
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10 1.0)))
```
- y_代表正确结果，y代表预测结果
- tf.clip_by_value用来把y值限制在1e-10 ~ 1.0之间
- tf.log函数完成了对张量中所有元素依次求对数的功能
- 计算得到结果是n x m的二维矩阵，其中n为一个batch中样例的数量，m为分类的类别数量
- 根据交叉熵的公式，应该将每行中的m个结果相加得到所有样例的交叉熵，然后再对这n行取平均得到一个batch的交叉平均熵，使用tf.reduce_mean来完成这一操作：

```
v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# 输出3.5
print tf.reduce_mean(v).eval()
```
- 交叉熵与softmax回归一起封装，使用tf.nn.softmax_cross_entropy_with_logits函数得到softmax回归之后的交叉熵损失函数：

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
```

与分类问题不同，回归问题解决的是对具体数值的预测，最常用的损失函数是均方误差MSE，定义为：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/11.png)

通过TensorFlow实现均方误差损失函数：

```
mse = tf.reduce_mean(tf.square(y_ - y))
```
### 自定义损失函数
一个当预测多于真实值和预测少于真实值有不同损失系数的损失函数：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/12.png)

在TensorFlow中实现：

```
loss = tf.reduce_sum(tf.select(tf.greater(v1, v2), (v1 - v2) * a, (v2 - v1) * b))
```

## 4.6 神经网络的优化

### 4.6.1 batch
实际应用中一般采用 每次计算一小部分训练数据的损失函数，这一小部分数据被称之为一个batch。

神经网络的训练大都遵循以下过程：

```
batch_size = n

# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape = (batch_size, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (batch_size, 1), name = 'y-input')

# 定义神经网络结构和优化算法
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练神经网络
with tf.Session() as sess:
    # 参数初始化
    ...
    # 迭代的更新参数
    for i in range (STEPS):
        # 准备batch_size个训练数据。一般将所有训练数据随机打乱之后再选取可以得到更好的优化结果
        current_X, current_Y = ...
        sess.run(train_step, feed_dict = {x: current_X, y_: current_Y})
```
### 4.6.2 学习率的设置
设置学习率控制参数更新的速度


```
global_step = tf.Variable(0)

# 通过exponential_decay函数生成学习率
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase = True)

# 使用指数衰减的学习率
# 在minimize函数中传入global_steo将自动更新global_step参数，从而使得学习率也得到相应更新
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(...my loss..., global_step = global_step)
```
- 上段代码设定了初始学习率为0.1，因为指定staircase=True,所以每训练100轮后学习率乘以0.96
- tf.train.exponential_decay函数实现了指数衰减学习率，通过这个函数，可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代逐步减小学习率
- exponential_decay函数会指数级地减小学习率，实现以下代码的功能：

```
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
```
- learning_rate: 初始学习率
- decay_rate: 衰减系数
- decay_steps：衰减速度

### 4.6.3 正则化
避免过拟合问题，假设用于刻画模型使用J(0)，优化时优化J(0)+λJ(0)
- R(w)刻画的是模型的复杂程度，而λ表示模型复杂损失在总损失中的比例
- 0表示的是一个神经网络中的所有参数，它包括边上的权重w和偏置项b

一个简单的带L2正则化损失函数定义：

```
w = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed =1))
y = tf.matmul(x, w)

loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)
```
L1正则化和L2正则化的区别:

```
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    # 输出为(|1|+|-2|+|-3|+|4|) * 0.5 = 5。其中0.5为正则化项的权重
    print sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))
    # 输出为(1^2 + (-2)^2 + (-3)^2 + 4^2) / 2 * 0.5 = 7.5
    print sess.run(tf.contrib.layers.l2_regularizer(.5)(weights))
```
### 4.6.4 滑动平均模型
滑动平均模型即用来控制模型参数调整的速度，如果有异常数据进入训练不至于改变的太离谱

使用tf.train.ExponentialMovingAverage来实现滑动平均模型：
- 初始化ExponentialMovingAverage时，需要提供一个衰减率(decay)，这个衰减率用于控制模型更新的速度
- ExponentialMovingAverage对每一个变量会维护一个影子变量，这个影子变量的初始值就是相应变量的初始值，而且每次运行变量更新时，影子变量的值会更新为：
        
    shadow_variable = decay * shadow_variable + (1 - decay) * variable
- 从公式中可以看出来，decay决定了模型更新的速度，decay越大模型越趋于稳定。（实际应用中，decay一般会设成非常接近1的数，比如0.999）
- 为了使得模型在训练前期可以更新更快，ExponentialMovingAverage还提供了num_updates参数来动态设置decay大小，如果初始化时提供了num_updates参数，那么每次使用的衰减率是：

    min{decay, (1 + num_updates) / (10 + num_updates)}

- 下面代码解释ExponentialMovingAverage是如何被使用：

```
import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0
# 这里手动指定了变量的类型为tf.float32,因为所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0, dtype = tf.float32)
# 这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable = False)

# 定义一个滑动平均的类(class)
# 初始化时给定了衰减率(0.99)和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。
# 这里需要给定一个列表，每次执行这个操作时这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    
    # 通过ema.average(v1)获取滑动平均之后变量的取值
    # 在初始化之后变量v1的值和v1的滑动平均都为0
    # 输出[0.0, 0.0]
    print sess.run([v1, ema.average(v1])
    
    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值。衰减率为min{0.99, （1 + step)/(10 + step) = 0.1} = 0.1
    # 所以v1的滑动平均会被更新为0.1 * 0 + 0.9 * 5 = 4.5
    sess.run(maintain_averages_op)
    # 输出[5.0, 4.5]
    print sess.run([v1, ema.average(v1)])
    
    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值。衰减率为min{0.99, (1 + step) / (10 +  step) ≈ 0.999} = 0.99
    # 所以v1的滑动平均会被更新为0.99*4.5+0.01*10 = 4.555
    sess.run(maintain_averages_op)
    # 输出[10.0, 4.5549998]
    print sess.run([v1, ema.average(v1)])
    
    # 再次更新滑动平均值，得到的新滑动平均值为0.99*4.555+0.01*10 = 4.60945
    sess.run(maintain_averages_op)
    # 输出[10.0, 4.6094499]
    print sess.run([v1, ema.average(v1)])
```
