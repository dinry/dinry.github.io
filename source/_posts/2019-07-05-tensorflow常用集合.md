---
layout: post/
title: temsorflow常用集合(colection)
date: 2019-07-05 10:55:07
tags: tensorflow
---


tensorflow 用集合collection组织不同类别的对象，tf.GraphKeys中包含了所有默认集合的名称。

collection在对应的scope内提供了“零存整取”的思想：任意位置，任意层次的对象，统一提取。

tf.optimizer只优化tf.GraphKeys.TRAINABLE_VARIABLES中的变量

## 常用集合
- Variable集合：模型参数
- summary 集合：监测
- 自定义集合

# Variable
Variable被收集在tf.GraphKeys.VARIABLES的collection中
## 定义
k=tf.Variable()
![](/tensorflow常用集合/1.JPG)

# summary

Summary被收集在名为tf.GraphKeys.SUMMARIES的collection中

## define
对网络中tensor取值进行监测

调用tf.scalar_summary系列函数，会向默认的collection中添加一个operation

![](/tensorflow常用集合/2.JPG)

# 自定义
tf.add_to_collection("losses",l1)
losses=tf.get_collection('losses')
