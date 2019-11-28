---
layout: post/
title: 'tensorflow深度学习(2):tf.nn.top_k()'
date: 2019-07-05 11:42:07
tags: tensorflow
---

# introduction

def top_k(input, k=1, sorted=True, name=None)

Finds values and indices of the k largest entries for the last dimension.

If the input is a vector (rank=1), finds the k largest entries in the vector and outputs their values and indices as vectors.Thus values[j] is the j-th largest entry in input, and its index is indices[j].

For matrices (resp. higher rank input), computes the top k entries in each row (resp. vector along the last dimension).Thus, values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

# parameters

![](/tensorflow深度学习-2-tf-nn-top-k/1.JPG)

# code

![](/tensorflow深度学习-2-tf-nn-top-k/2.JPG)
