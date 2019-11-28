---
layout: post/
title: 'paper:AdamOptimizer'
date: 2019-07-12 20:20:39
tags: tensorflow
categories: deep learning
---

# paper：Adam: A Method for Stochastic Optimization

论文链接：![](https://arxiv.org/abs/1412.6980)

![](/paper-AdamOptimizer/1.JPG)

如上算法所述，在确定了参数 $\alpha$,$\beta_1$,$\beta_2$和随机目标函数 $f(\theta)$ 之后，我们需要初始化参数向量、一阶矩向量、二阶矩向量和时间步。然后当参数 $\theta$ 没有收敛时，循环迭代地更新各个部分。即时间步 t 加 1、更新目标函数在该时间步上对参数 $\theta$ 所求的梯度、更新偏差的一阶矩估计和二阶原始矩估计，再计算偏差修正的一阶矩估计和偏差修正的二阶矩估计，然后再用以上计算出来的值更新模型的参数 $\theta$。

# 算法
