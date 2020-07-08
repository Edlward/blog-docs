本文是从智慧门禁系统项目工程实践中摘取出人脸识别模块来进行一个记录。
# 一、人脸识别器软件设计

## 1.1 总体流程图
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/14.png)

## 1.2 系统代码目录结构
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/15.png)

# 二、模块详细设计
人脸识别器模块为了方便后台进行调用，给出了一个接口，通过不同参数的传入可以选择训练或识别，并且根据选择的功能返回不同的返回值，代表不同的含义。人脸识别器分为训练模块，识别模块，并且有着与本地文件交互的相关工具函数集，用来读取和处理本地文件到内存中方便训练和识别模块来运行。

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/16.png)

## 2.1 训练模块(train)的实现

此模块共有两个主要部分，siamese class决定着神经网络的相关配置，将Triplet网络结构的相关参数、前向传播和损失函数等；siamese train主要为主接口face_inference.py提供调用的接口，每一次调用只遍历一轮数据并采用随机采样的三元组图像数据作为输入。训练模块在开始训练前读取log文件来判断是否为第一次训练，如果不是第一次训练则读取日志中保存的相关参数做相应处理作为本轮训练的参数。

### 2.1.1 Siamese class

Siamese网络是一种相似性度量方法，当类别数多，但每个类别的样本数量少的情 况下可用于类别的识别、分类等。Siamese网络也非常适合于进行人脸识别研究，被测者 仅需提供最少1张照片，即可正确识别出被测者。这与利用分类器来识别人脸有着很大的 不同，利用分类器进行训练的方法需要每一类中有足够多的样本才行，这其实在实际生活中是不现实的。原因如下：无法对每一个人采取足够多的样本，当类别过多，每一类别的 样本也足够多时，机器性能也就跟不上了，无法训练出合适的模型；无法对陌生人脸进行 划分，主要指不属于分类器中的类别会被划分为分类器中的类别，例如，分类器可以识别 3个人的人脸，当第4个人要识别时，他就会被误识别为这3个人中的其中一个。
Siamese 网络的主要思想是通过一个函数将输入映射到目标空间，在目标空间使用 简单的距离（欧式距离等）进行对比相似度。在训练阶段去最小化来自相同类别的一对样 本的损失函数值，最大化来自不同类别的一堆样本的损失函数值。Siamese网络训练时的 输入是一对图片，这一对图片是X1 ,X2标签是y。当X1, X2是同类时，y为0，当X1, X2不同类时，y为1。Siamese网络的损失函数为L=(1 - y)LG (EW(X1, X2 )) + yLI (EW(X1, X2))，其中EW(X1, X2)=||G W(X1) - GW(X2 )||。GW (X)就是神经网络中的参数，其中LG是只计算相同类别对图片的损失函数，LI是只计算不相同类别对图片的损失函数。
训练模块通过使用Siamese网络的改进版——Triplet网络，使用三元组（Anchor, Positive, Negative)来作为网络结构的输入。通过训练一个参数共享的网络(VGG16，网络结构如图3-4所示),得到三个元素的特征表，记为f(x_a), f(x_p), f(x_n)。通过学习，尽可能让x_a与x_n之间的距离和x_a与x_p之间距离之间有一个最小的间隔t，即||f(x_a) - f(x_p)|| + t < ||f(x_a) - f(x_n)||。损失函数为：L = {||f(x_a) - f(x_p)|| - ||f(x_a) - f(x_n)|| + t}_+


VGG16网络结构图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/17.png)

Triplet网络结构图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/18.png)

### 2.1.2 Siamese train
为了方便后台服务器的调用，只将训练一轮的流程封装起来，如果要不停的训练则只需在后台服务程序中循环调用即可，为了达成这一目的，每进行一轮训练则需保存当前相关参数。训练时每128组数据为一个batch，循环每个人为一轮。

训练流程图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/19.png)


## 2.2 识别模块(recognition)的实现

识别模块封装好识别函数以供人脸识别接口调用，传入需要识别照片的系统路径，返回不同的返回值。根据训练日志提取出训练总轮数，判定选择哪种识别方式进行识别。
使用训练好的模型需要加载模型，将要识别照片与证件照库中每个人的照片分别传入已经训练好的网络当中做一次正向传递操作，并且计算两张图片最终的距离，根据提前设定好的阈值来判定是否识别为本照片库中人脸，取得距离值最小的匹配人脸返回人名。
使用第三方API选择百度人脸识别接口，通过Python的requests库向接口发送POST请求，利用jason库来对接口传回的jason格式数据进行解析，并使用os库与本地文件进行交互，成功则返回人名，失败则根据具体失败原因返回不同的值。

识别总框图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/20.png)


识别系统流程图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/21.png)

## 2.3 工具集(tools)的实现
为了使代码得结构更加鲜明，增加代码复用率，封装了几个和本地文件交互并且生成目标数据的函数。

主要工具函数图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/22.png)

# 三、测试
## 3.1 测试训练模块
运行训练模块的代码文件，查看是否能够成功运行，打印每一batch训练后的结果：时间，步数，损失值，总损失值；并且打开(创建)日志文件并写入：时间，步数，总损失值和学习率。

训练运行图：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/23.png)

## 3.2 测试识别模块
运行识别模块代码文件，通过传入待识别照片，经过识别模块进行识别传出识别出的结果，若成功则返回人名拼音，若失败则返回相关错误值

识别测试图（马赛克是自己截图时打的）：
![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/24.png)

## 3.3 集成测试

这一步是我们项目工程集成测试，需要前端和后台交互

通过向后台提供统一的调用接口，使得后台可以通过统一的接口来调用我的人脸识别器进行训练和识别，后台Java来调用Python文件的统一接口来完成识别流程。

![在这里插入图片描述](http://39.106.181.170:8080/getImage?path=/root/code/go/kjblog/resources/blog-docs/machineLearning/img/25.png)

# 四、代码
## 4.1 人脸识别器总接口
face_inference.py
```
"""
功能：人脸识别器总接口
作者：zach_zhang

调用格式: python face_inference.py [options] [argument]
    options: r
        argument: 图片路径
        返回值： 字符串：最有可能的人名 2：请对齐人脸 1：错误(网络问题） 0：与数据库中人脸不匹配
    options: t
        argument: 训练集文件夹路径  ./test/
        返回值 1：成功
"""
import sys

from recognition import recognition_inference, third_api
from train import inference, tools, train

if __name__ == "__main__":
    parameterList = sys.argv
    if(len(parameterList) < 3):
        print (0)
    
    elif(parameterList[1] == 'r'):
        face_path = parameterList[2]
        ret = recognition_inference.recognize(face_path)
        print (ret)
        
    elif (parameterList[1] == 't'):
        train_data_path = parameterList[2]
        ret = train.siamese_train(train_data_path)
        print (ret)
    else:
        print (0)
```
## 4.2 Triplet网络结构类：

```
"""
功能：神经网络的配置

作者: zach_zhang
"""

import tensorflow as tf

# 输入图像大小为96*96*3
# INPUT_NODE = 27648
IMAGE_SIZE = 96
NUM_CHANNELS = 3

CONV_SIZE = 3

CONV1_DEEP = 64
CONV2_DEEP = 64

CONV3_DEEP = 128
CONV4_DEEP = 128

CONV5_DEEP = 256
CONV6_DEEP = 256
CONV7_DEEP = 256

CONV8_DEEP = 512
CONV9_DEEP = 512
CONV10_DEEP = 512

CONV11_DEEP = 512
CONV12_DEEP = 512
CONV13_DEEP = 512

FC1_SIZE = 4096
FC2_SIZE = 4096
FC3_SIZE = 1000

'''
功能：定义了神经网络的前向传播，损失函数
'''    
class siamese:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        self.x2 = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        self.x3 = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        with tf.variable_scope("siamese") as scope:
            #x1_input = tf.reshape(self.x1, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            self.out1 = self.network(self.x1)
            
            scope.reuse_variables()
            #x2_input = tf.reshape(self.x2, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            self.out2 = self.network(self.x2)
            
            scope.reuse_variables()
            #x3_input = tf.reshape(self.x3, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            self.out3 = self.network(self.x3)
            
        
        self.loss = self.Triplet_loss()
        self.look_like = self.cal_distance()
    

    def network(self, x):
        with tf.variable_scope('conv1'):
            # 第一层卷积，输入96*96*3 输出96*96*64
            kernel_shape = [CONV_SIZE, CONV_SIZE, NUM_CHANNELS, CONV1_DEEP]
            bias_shape = [CONV1_DEEP]
            conv1 = self.cnn_layer(x, kernel_shape , bias_shape)
        
        with tf.variable_scope('conv2'):
            # 第二层卷积，输入96*96*64 输出48*48*64
            conv2 = self.cnn_layer(conv1, [CONV_SIZE, CONV_SIZE, CONV1_DEEP, CONV2_DEEP], [CONV2_DEEP])
            pool1 = self.pool_layer(conv2)
        
        with tf.variable_scope('conv3'):
            # 第三层卷积，输入48*48*64 输出48*48*128
            conv3 = self.cnn_layer(pool1, [CONV_SIZE, CONV_SIZE, CONV2_DEEP, CONV3_DEEP], [CONV3_DEEP])
            
        with tf.variable_scope('conv4'):
            # 第四层卷积，输入48*48*128 输出24*24*128
            conv4 = self.cnn_layer(conv3, [CONV_SIZE, CONV_SIZE, CONV3_DEEP, CONV4_DEEP], [CONV4_DEEP])
            pool2 = self.pool_layer(conv4)
            
        with tf.variable_scope('conv5'):
            # 第五层卷积，输入24*24*128 输出24*24*256
            conv5 = self.cnn_layer(pool2, [CONV_SIZE, CONV_SIZE, CONV4_DEEP, CONV5_DEEP], [CONV5_DEEP])
        with tf.variable_scope('conv6'):
            # 第六层卷积，输入24*24*256 输出24*24*256
            conv6 = self.cnn_layer(conv5, [CONV_SIZE, CONV_SIZE, CONV5_DEEP, CONV6_DEEP], [CONV6_DEEP])
        with tf.variable_scope('conv7'):
            # 第七层卷积，输入24*24*256 输出12*12*256
            conv7 = self.cnn_layer(conv6, [CONV_SIZE, CONV_SIZE, CONV6_DEEP, CONV7_DEEP], [CONV7_DEEP])
            pool3 = self.pool_layer(conv7)
            
        with tf.variable_scope('conv8'):
            # 第八层卷积，输入12*12*256 输出12*12*512
            conv8 = self.cnn_layer(pool3, [CONV_SIZE, CONV_SIZE, CONV7_DEEP, CONV8_DEEP], [CONV8_DEEP])
        with tf.variable_scope('conv9'):
            # 第九层卷积，输入12*12*512 输出12*12*512
            conv9 = self.cnn_layer(conv8, [CONV_SIZE, CONV_SIZE, CONV8_DEEP, CONV9_DEEP], [CONV9_DEEP])
        with tf.variable_scope('conv10'):
            # 第十层卷积，输入12*12*512 输出6*6*512
            conv10 = self.cnn_layer(conv9, [CONV_SIZE, CONV_SIZE, CONV9_DEEP, CONV10_DEEP], [CONV10_DEEP])
            pool4 = self.pool_layer(conv10)
            
        with tf.variable_scope('conv11'):
            # 第十一层卷积，输入6*6*512 输出6*6*512
            conv11 = self.cnn_layer(pool4, [CONV_SIZE, CONV_SIZE, CONV10_DEEP, CONV11_DEEP], [CONV11_DEEP])
        with tf.variable_scope('conv12'):
            # 第十二层卷积，输入6*6*512 输出6*6*512
            conv12 = self.cnn_layer(conv11, [CONV_SIZE, CONV_SIZE, CONV11_DEEP, CONV12_DEEP], [CONV12_DEEP])
        with tf.variable_scope('conv13'):
            # 第十三层卷积，输入6*6*512 输出3*3*512
            conv13 = self.cnn_layer(conv12, [CONV_SIZE, CONV_SIZE, CONV12_DEEP, CONV13_DEEP], [CONV13_DEEP])
            pool5 = self.pool_layer(conv13)
        
        # pool4_shape[0]是batch数目，nodes为长宽和深度的乘积
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        reshaped = tf.reshape(pool5, [-1, nodes])
        
        
        with tf.variable_scope('fc1'):
            # 全连接层1，输入3*3*512 输出4096 [实际(batch, 3*3*512) * (3*3*512, 4096) = (batch, 4096)]
            fc1 = self.full_layer(reshaped, [nodes, FC1_SIZE], [FC1_SIZE])
        with tf.variable_scope('fc2'):
            # 全连接层2，输入4096 输出1024
            fc2 = self.full_layer(fc1, [FC1_SIZE, FC2_SIZE], [FC2_SIZE])
        with tf.variable_scope('fc3'):
            # 全连接层3，输入1024 输出128
            fc3 = self.full_layer(fc2, [FC2_SIZE, FC3_SIZE], [FC3_SIZE])
            
        return fc3
    
    def cnn_layer(self, input_x, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", shape = kernel_shape, initializer = tf.truncated_normal_initializer(stddev = 0.04))
        biases = tf.get_variable("biases", shape = bias_shape, initializer = tf.constant_initializer(0.01))
        
        conv = tf.nn.conv2d(input_x, weights, strides = [1, 1, 1, 1], padding = 'SAME')
        return tf.nn.relu(tf.nn.bias_add(conv, biases))
    
    def pool_layer(self, input_x):
        pool = tf.nn.max_pool(input_x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        drop = tf.nn.dropout(pool, 1.0)
        return drop
    
    def full_layer(self, input_x, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", shape = kernel_shape, initializer = tf.truncated_normal_initializer(stddev = 0.04))
        biases = tf.get_variable("biases", shape = bias_shape, initializer = tf.constant_initializer(0.01))
        
        fc = tf.nn.relu(tf.matmul(input_x, weights) + biases)
        drop = tf.nn.dropout(fc, 1.0)
        return drop
        
    def Triplet_loss(self):
        margin = 5.0
        main_out = self.out1
        pos_out = self.out2
        neg_out = self.out3
        
        d_pos = tf.reduce_sum(tf.square(main_out - pos_out), 1, name = "d_pos")
        d_neg = tf.reduce_sum(tf.square(main_out - neg_out), 1, name = "d_neg")
        
        losses = tf.maximum(0., margin + d_pos - d_neg, name = "losses")
        loss = tf.reduce_mean(losses, name = "loss")
        return loss
    
    def cal_distance(self):
        anchor_output = self.out1  # shape [None, 128]
        positive_output = self.out2  # shape [None, 128]
        d_look = tf.reduce_sum(tf.square(anchor_output - positive_output), 1, name="d_look")
        distance = tf.reduce_mean(d_look, name="distance")
        return distance
```
## 4.3 用来训练一轮的代码

```
"""
功能：开始神经网络训练

作者：zach_zhang
"""
import os
import csv
import tensorflow as tf
import numpy as np
from time import ctime

from train import inference, tools
BATCH_SIZE = 128

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'
NEW = False
file_name = './log.csv'
def siamese_train(face_data_path):
    # 读取文件夹，返回标签数和图片路径数组
    max_num, face_path = tools.get_image_path(face_data_path)
    tf.reset_default_graph()
    siamese = inference.siamese()
    if os.path.exists(file_name) and os.path.getsize(file_name):
        with open(file_name,'r') as csv_r:
            lines=csv_r.readlines()
            opt = lines[-1].split(',')
            
            step = int(opt[2])
            loss_sum = float(opt[4])
            learn_rate = float(opt[6]) * 0.99
    else:
        loss_sum = 0
        step = 0
        learn_rate = 1e-6
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(siamese.loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_file = MODEL_SAVE_PATH + MODEL_NAME
        tf.initialize_all_variables().run()
        if not NEW:
            saver.restore(sess, model_file)
        
        #loss_sum = 0
        
        
        # 生成训练数据
        train_data = tools.generate_train_data(face_path, max_num)
        
        # 开始训练一遍
        loss_step = 0
        losses = 0
        image_x1 = []
        image_x2 = []
        image_x3 = []
        # 从train_data[0]~train_data[5748] 每128为一个BATCH
        for step_i in range(0, max_num-1):
            # 每128（BATCH）传入神经网络中训练一遍
            if step_i > 1 and step_i % BATCH_SIZE == 0:
                loss_step += 1
                train_anc = np.array(image_x1)
                train_pos = np.array(image_x2)
                train_neg = np.array(image_x3)
        
                _, loss_v = sess.run([train_step, siamese.loss], feed_dict = {
                        siamese.x1: train_anc,
                        siamese.x2: train_pos,
                        siamese.x3: train_neg})
                losses = losses + loss_v
                #print('time %s, %d: loss %.4f  losses %.4f' % (ctime(), loss_step, loss_v, losses))
                image_x1.clear()
                image_x2.clear()
                image_x3.clear()
                
            # 没到128数量前，先攒够一个BATCH，把图像数据读出
            x1 = train_data[step_i][0]
            x2 = train_data[step_i][1]
            x3 = train_data[step_i][2]
            image_x1.append(tools.get_image_array(x1))
            image_x2.append(tools.get_image_array(x2))
            image_x3.append(tools.get_image_array(x3))
        
        loss_sum += losses
        with open(file_name, 'a+', newline = '') as csv_w:
            csv_print_writer = csv.writer(csv_w, dialect='excel')
            csv_print_writer.writerow([ctime(), 'step:', step, '  loss_sum:', loss_sum, '  rate:', learn_rate])
        saver.save(sess, model_file)    
        return 1
```
## 4.4 人脸检索识别

```
"""
功能：通过训练好的CNN模型来实现人脸检索

作者: zach_zhang
"""
import tensorflow as tf
import cv2
import numpy as np

from train import inference, tools
from before import align_dlib

model = './model/model.ckpt'
face_path = './my_image_data/'
PREDICTOR_PATH = './before_train/shape_predictor_68_face_landmarks.dat'

def recognize_cnn(face_input):
    detector = align_dlib.AlignDlib(PREDICTOR_PATH)
    img_bgr = cv2.imread(face_input)  # 从文件读取bgr图片
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转为RGB图片
    face_align = detector.align(96, img_rgb)
    if face_align is None:
        return 0
    else:
        face_align = cv2.cvtColor(face_align, cv2.COLOR_RGB2BGR) #转为BGR
    test_data = np.array(face_align).astype('float32') / 255.0
    
    siamese = inference.siamese()
    max_num, names, face_array = tools.get_triplet_data(face_path)
    face = [[] for n in range(max_num)]
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model)
        for i in range(max_num):
            for j, img in enumerate(face_array[i]):
                res = sess.run(siamese.look_like, feed_dict = {
                        siamese.x1 : [test_data],
                        siamese.x2 : [img]})
                face[i].append(res)
        new_list = []
        for n in range(max_num):
            face[n].sort()
            new_list.append(face[n][0])
        min_data = min(new_list)
        if min_data < 0.3: #阈值为0.3
            face_id = new_list.index(min_data)
            name = names[face_id]
        else:
            name = None
        return name
```
## 4.5 百度人脸识别API的使用

```
"""
功能：通过第三方API接口实现人脸检索

作者: zach_zhang
"""

import requests
import base64
import json
import sys


# 这里使用自己的ak和sk
AK = "xxxxxxxx"
SK = "xxxxxxxx"

host = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"\
       "&client_id=%s&client_secret=%s" % (AK, SK)

url_match = "https://aip.baidubce.com/rest/2.0/face/v3/match"
url_search = "https://aip.baidubce.com/rest/2.0/face/v3/search"



'''
功能：获取access_token
输入：无
输出：access_token的值
'''
def GetToken():
    request=requests.get(host)
    js = request.json()
    return js['access_token']


'''
功能：将两张图片中的人脸进行对比
输入：人脸图片1，人脸图片2（路径）， access_token的值
输出：1：可能是同一个人 0：可能不是同一人
'''
def FaceRecg(face1, face2, token):
    f = open(face1, 'rb')
    # 参数images：图像base64编码
    img1 = base64.b64encode(f.read())
    # 二进制方式打开图文件
    f = open(face2, 'rb')
    # 参数images：图像base64编码
    
    img2 = base64.b64encode(f.read())

    #print (str(img1, 'utf-8'))
    
    #images = str(img1, 'utf-8') + ',' + str(img2, 'utf-8')
    #print (images)
    
    payload = json.dumps([{'image': str(img1, 'utf-8'), "image_type": "BASE64"},{'image': str(img2, 'utf-8'), "image_type": "BASE64"}])
    #print (payload)

    request_url = url_match + "?access_token=" + token
    
    #headers = {'Content-Type' : 'application/json'}
    
    request = requests.post(request_url, data = payload)
    
    
    if request.status_code == 200:
        js = request.json()
        #print(request.text)
        if (js['result']['score'] > 80):
            #print(js['result']['score'].type)
            return 1
        else:
            return 0


'''
功能：将图片中的人脸与数据库中人脸进行检索
输入：人脸图片1（路径）， access_token的值
输出：字符串：最有可能的人名 2：请对齐人脸 1：错误(网络问题） 0：与数据库中人脸不匹配
'''
def FaceSearch(face, token):
    f = open(face, 'rb')

    img = base64.b64encode(f.read())
    payload = json.dumps({'image':str(img, 'utf-8'), "image_type":"BASE64", "group_id_list":"owner"})
    #payload = json.dumps({'image':face, "image_type":"BASE64", "group_id_list":"owner"})
    request_url = url_search + "?access_token=" + token
    
    #headers = {'Content-Type' : 'application/json'}
    
    request = requests.post(request_url, data = payload)
    #print(request.text)
    if request.status_code == 200:
        js = request.json()
        #print(js['result']['user_list'])
        #print(js['result']['user_list'][0]['score'])
        if js['error_code'] != 0:
            return 2
        else:
            if (js['result']['user_list'][0]['score'] > 80):
                return js['result']['user_list'][0]['user_id']
            else:
                return 0
    else:
        return 1
```

## 4.6 工具函数集

```
"""
功能：多工具函数

作者：zach_zhang
"""
import os
import random
import numpy as np
import cv2

'''
功能：获取一个文件夹里文件夹数和图片路径
输入：文件夹路径
输出：文件夹数，图片路径（二维数组）
'''
def get_image_path(path):
    max_num = len(os.listdir(path))
    face_array = []
    id_array=[]
    for filename in os.listdir(path):
        for img_name in os.listdir(path + filename):
            if img_name.endswith('.jpg'):
                path_name = path + filename + '/' + img_name
                id_array.append(path_name)
        face_array.append(id_array)
        id_array=[]
    return max_num, face_array

'''
功能：从图片中随机生成三元组(a, pos, neg)
输入：图片路径(二维数组)，文件夹数(二维数组的一维数目)
输出：三元组列表
'''
def generate_train_data(image_path, num):
    temp_data = []
    train_data = []
    third_num = None
    for i in range(num):
        temp_data.clear()
        # 获取第i个文件夹下图片路径并打乱
        temp_path = image_path[i]
        random.shuffle(temp_path)
        
        # 如果文件夹下只有一张图片，取两次 作为a和pos
        if len(temp_path) == 1:
            temp_data.append(str(temp_path[0]))
            temp_data.append(str(temp_path[0]))
        elif len(temp_path) >= 2:
            temp_data.append(str(temp_path[0]))
            temp_data.append(str(temp_path[-1]))
        else:
            continue
        
        # 循环找与前两张不属于同一文件夹的第三张图片
        flag_find_third = True
        while flag_find_third:
            third_num = random.randint(0, num - 1)
            if third_num != i and len(image_path[third_num]) != 0:
                flag_find_third = False
        # 找到要取图片的文件夹，并随机打乱,取一张作为neg
        temp_path_third = image_path[third_num]
        random.shuffle(temp_path_third)
        temp_data.append(str(temp_path_third[0]))
        # 将三元组加入train_data里
        train_data.append(tuple(temp_data))
    random.shuffle(train_data)
    return train_data
 
'''
功能：读取图片转化为矩阵格式,并归一化处理
输入：图片路径
输出：一个array
'''       
def get_image_array(path):
    img  = cv2.imread(path)
    img = np.array(img)
    return img.astype('float32') / 255.0
    
      
'''
功能：获取一个文件夹里文件数，图片文件名称，图片数据
输入：文件夹路径
输出：文件数，文件名称，图片数据
'''            
def get_triplet_data(path):
    #print(11111)
    
    names = []
    num = len(os.listdir(path))             # 获取人脸标签数
    face_array = [[] for n in range(num)]   # 初始化二维数组
    for i, filename in enumerate(os.listdir(path)):
        for img_name in os.listdir(path+filename):
            if img_name.endswith('.jpg'):
                path_name = path+filename + '/' + img_name
                #print(path_name)
                img = cv2.imread(path_name)
                img = np.array(img)
                face_array[i].append(img.astype('float32') / 255.0)
        names.append(filename)
            
    return num, names, face_array
```
