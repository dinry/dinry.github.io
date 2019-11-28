---
layout: word
title: Word Embedding
date: 2019-09-19 18:33:25
tags: NLP
categories: NLP
---

# dimension reduction
![](/embedding/1.JPG)

# word Embedding
* Machine learn the meaning of words from reading a lot of documents without supervision.
* Generating Word Vector is Unsupervised
* A word can be understood by its context.
![](/embedding/2.JPG)

# How to exploit the context?
* count based: If two words $w_i$ and $w_j$ frequently co-occur, $V(w_i)$ and $V(w_j)$ would be close to each other.(Glove Vector)

$V(w_i) \cdot V(w_j) \to N_{i,j}$, where number of times $w_i$ and $w_j$ in the same document.
* prediction based: predict next word based on previous words.

![](/embedding/3.JPG)

* take out he input of the neurons in the first layer.
* use it to represent a word w
* word vector. word embedding feature: V(w)
具有相同上下文的单词具有相近的分布
![](/embedding/4.JPG)
![](/embedding/5.JPG)
![](/embedding/6.JPG)
如何让两个weight一样？一样有什么好处？
* Given the same initialization
* ![](/embedding/7.JPG)
* cross entropy: ![](/embedding/8.JPG)
## two class:
* Cbow
* skip-gram
![](/embedding/9.JPG)
结构信息：结构，包含关系等
![](/embedding/10.JPG)
![](/embedding/11.JPG)
# document Embedding
![](/embedding/12.JPG)
