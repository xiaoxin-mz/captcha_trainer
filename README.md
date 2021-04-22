使用说明：
https://www.freebuf.com/articles/web/195469.html


# 1. 项目介绍

基于深度学习的图片验证码的解决方案 - 该项目能够秒杀字符粘连重叠/透视变形/模糊/噪声等各种干扰情况，足以解决市面上绝大多数复杂的[验证码场景](#jump)，目前也被用于其他OCR场景。 
<div align=center>
<img src="https://raw.githubusercontent.com/kerlomz/captcha_trainer/master/resource/logo.png" style="zoom:70%;" />
</div>

<div align=center>
<a href="https://github.com/kerlomz/captcha_trainer/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</div>

该项目基于 TensorFlow 1.14 开发，旨在帮助中小企业或个人用户快速构建图像分类模型并投入生产环境使用，降低技术应用门槛。

面向算法工程师：提供了可拓展的结构支持，允许通过源码灵活方便的添加自己设计的网络结构及其他组件。

面向零基础用户：有需求？但是不会编程？时间就是金钱，学习成本太高我想白嫖。它简直为你而生！

面向需求频繁者：同样的类型的需求一天10个，它的复用程度让您无需一份代码一个需求，一套服务全部搞定。



> **编译版下载地址：** https://github.com/kerlomz/captcha_trainer/releases/

------

其使用的网络结构主要包含三部分，从下至上依次为：
<div align=center>
<img src="https://raw.githubusercontent.com/kerlomz/captcha_trainer/master/resource/net_structure.png" style="zoom:80%;" />
</div>
>  输入OP:  **input** ， 输出OP:   **dense_decoded** 



卷积层：从输入图像中提取特征序列;

循环层，预测从卷积层获取的特征序列的标签（真实值）分布;

转录层，把从循环层获取的标签分布通过去重整合等操作转换成最终的识别结果;



## 1. 可视化模型配置

为每个图像分类任务创建一个独立的项目，每个项目拥有完全独立的管理空间，方便多任务切换和管理。**全程无需修改一行代码**，根据模板生成模型配置，生成的配置文件可直接用模型部署服务。

本项目对应的部署服务支持同时部署多套模型，模型支持热拔插，版本迭代等，业务层可拓展颜色提取，算术运算等常见的验证码解决方案。详情可以移步：https://github.com/kerlomz/captcha_platform 
<div align=center>
<img src="https://raw.githubusercontent.com/kerlomz/captcha_trainer/master/resource/main.png" style="zoom:80%;" />
</div>

## 2. 特性

1. 目前支持Windows平台的GPU编译版，无需安装环境，0基础建模。
2. 项目化管理，适合容易被任务优先级安排的程序员们，同一份代码，不同的项目，随心切换，互不干扰。
3. 新增样本集无需重新打包，可直接增量添加新的样本集，每个训练任务支持加载多个TFRecords文件。
4. 解除循环层依赖的必须性，支持CNN5/ResNet50/DenseNet+CrossEntropy的怀旧组合模式。
5. 提供智能建议性配置功能，选择样本源路径时可根据样本特性自动推荐字符集，设置尺寸，标签数等。
6. 支持不定宽[-1, HEIGHT]的网络输入，在样本尺寸多样的场景下自动按比例缩放。
7. 支持训练中的数据增广，如：指定范围的二值化/模糊/旋转/椒盐噪声等。



## 3. 模板参数介绍

```yaml
# - requirement.txt  -  GPU: tensorflow-gpu, CPU: tensorflow
# - If you use the GPU version, you need to install some additional applications.
# MemoryUsage: 显存占用率，推荐0.6-0.8之间
System:
  MemoryUsage: {MemoryUsage}
  Version: 2

# CNNNetwork: [CNN5, ResNet50, DenseNet] 
# RecurrentNetwork: [CuDNNBiLSTM, CuDNNLSTM, CuDNNGRU, BiLSTM, LSTM, GRU, BiGRU, NoRecurrent]
# - 推荐配置为 不定长问题：CNN5+GRU ，定长：CNN5/DenseNet/ResNet50
# UnitsNum: RNN层的单元数 [16, 64, 128, 256, 512] 
# - 神经网络在隐层中使用大量神经元，就是做升维，将纠缠在一起的特征或概念分开。
# Optimizer: 优化器算法 [AdaBound, Adam, Momentum]
# OutputLayer: [LossFunction, Decoder]
# - LossFunction: 损失函数 [CTC, CrossEntropy] 
# - Decoder: 解码器 [CTC, CrossEntropy] 
NeuralNet:
  CNNNetwork: {CNNNetwork}
  RecurrentNetwork: {RecurrentNetwork}
  UnitsNum: {UnitsNum}
  Optimizer: {Optimizer}
  OutputLayer:
    LossFunction: {LossFunction}
    Decoder: {Decoder}


# ModelName: 模型名/项目名，同时也对应编译后的pb模型文件名
# ModelField: 模型处理的数据类型，目前只支持图像 [Image, Text]
# ModelScene: 模型处理的场景类型，目前只支持分类场景 [Classification]
# - 目前只支持 “图像分类” 这一种场景.
Model:
  ModelName: {ModelName}
  ModelField: {ModelField}
  ModelScene: {ModelScene}

# FieldParam 分为 Image, Text 两种，不同数据类型时可配置的参数不同，目前只提供 Image 一种。
# ModelField 为 Image 时:
# - Category: 提供默认的内置解决方案:
# -- [ALPHANUMERIC（含大小写英文数字）, ALPHANUMERIC_LOWER（小写英文数字）, 
# -- ALPHANUMERIC_UPPER（大写英文数字）,NUMERIC（数字）, ALPHABET_LOWER（小写字母）, 
# -- ALPHABET_UPPER（大写字母）, ALPHABET（大小写字母）, 
# -- ALPHANUMERIC_CHS_3500_LOWER（小写字母数字混合中文常用3500）]
# - 或者可以自定义指定分类集如下（中文亦可）:
# -- ['Cat', 'Lion', 'Tiger', 'Fish', 'BigCat']
# - Resize: 重置尺寸，对应网络的输入： [ImageWidth, ImageHeight/-1, ImageChannel]
# - ImageChannel: 图像通道，3为原图，1为灰度 [1, 3]
# - 为了配合部署服务根据图片尺寸自动选择对应的模型，由此诞生以下参数（ImageWidth，ImageHeight）:
# -- ImageWidth: 图片宽度.
# -- ImageHeight: 图片高度.
# - MaxLabelNum: 该参数在使用CTC损失函数时将被忽略，仅用于使用交叉熵作为损失函数/标签数固定时使用
# ModelField 为 Text 时:
# - 该类型暂时不支持
FieldParam:
  Category: {Category}
  Resize: {Resize}
  ImageChannel: {ImageChannel}
  ImageWidth: {ImageWidth}
  ImageHeight: {ImageHeight}
  MaxLabelNum: {MaxLabelNum}
  OutputSplit: {OutputSplit}


# 该配置应用于数据源的标签获取.
# LabelFrom: 标签来源，目前只支持 从文件名提取 [FileName, XML, LMDB]
# ExtractRegex: 正则提取规则，对应于 从文件名提取 方案 FileName:
# - 默认匹配形如 apple_20181010121212.jpg 的文件.
# - 默认正则为 .*?(?=_.*\.)
# LabelSplit: 该规则仅用于 从文件名提取 方案:
# - 文件名中的分割符形如: cat&big cat&lion_20181010121212.png，那么分隔符为 &
# - The Default is null.
Label:
  LabelFrom: {LabelFrom}
  ExtractRegex: {ExtractRegex}
  LabelSplit: {LabelSplit}


# DatasetPath: [Training/Validation], 打包为TFRecords格式的训练集/验证集的本地绝对路径。
# SourcePath:  [Training/Validation], 未打包的训练集/验证集源文件夹的本地绝对路径。
# ValidationSetNum: 验证集数目，仅当未配置验证集源文件夹时用于系统随机抽样用作验证集使用。
# - 该选项用于懒人训练模式，当样本极度不均衡时建议手动设定合理的验证集。
# SavedSteps: 当 Session.run() 被执行一次为一步（1.x版本），保存训练过程的步数，默认为100。
# ValidationSteps: 用于计算准确率，验证模型的步数，默认为每500步验证一次。
# EndAcc: 结束训练的条件之准确率 [EndAcc*100]% 到达该条件时结束任务并编译模型。
# EndCost: 结束训练的条件之Cost值 EndCost 到达该条件时结束任务并编译模型。
# EndEpochs: 结束训练的条件之样本训练轮数 Epoch 到达该条件时结束任务并编译模型。
# BatchSize: 批次大小，每一步用于训练的样本数量，不宜过大或过小，建议64。
# ValidationBatchSize: 验证集批次大小，每个验证准确率步时，用于验证的样本数量。
# LearningRate: 学习
