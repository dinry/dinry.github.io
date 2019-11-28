---
layout: post/
title: tf.multinomial
date: 2019-08-03 21:31:58
tags: tensorflow
categories: deep learning
---
明明按概率，亲测却非常随机

tf.multinomial(logits, num_samples, seed=None, name=None)

从multinomial分布中采样，样本个数是num_samples，每个样本被采样的概率由logits给出

#### parametrs:
* logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log probabilities for all classes.2维量，shape是 [batch_size, num_classes]，每一行都是关于种类的未归一化的对数概率

* num_samples: 0-D. Number of independent samples to draw for each row slice.标量，表示采样的个数，更重要的是，它限制了返回张量中元素的范围{：0，1，2，…，num_samples-1 }

```js
import tensorflow as tf
samples = tf.multinomial(tf.log([[10., 10., 10.]]), 5)
with tf.Session() as sess:
	sess.run(samples)

# 运行结果：array([[2, 1, 2, 2, 0]])
```
