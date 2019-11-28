---
layout: post/
title: tensorflow学习率衰减
date: 2019-08-16 11:01:00
tags: tensorflow
---

在神经网络的训练过程中，学习率(learning rate)控制着参数的更新速度，tf.train类下面的五种不同的学习速率的衰减方法。

* tf.train.exponential_decay
* tf.train.inverse_time_decay
* tf.train.natural_exp_decay
* tf.train.piecewise_constant
* tf.train.polynomial_decay

1. 首先使用较大学习率(目的：为快速得到一个比较优的解);
2. 然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);

```js
tf.train.exponential_decay(
    learning_rate,初始学习率
    global_step,当前迭代次数
    decay_steps,衰减速度（在迭代到该次数时学习率衰减为earning_rate * decay_rate）
    decay_rate,学习率衰减系数，通常介于0-1之间。
    staircase=False,(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
    name=None
)
```
