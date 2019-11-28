---
layout: post/
title: RippleNet  Propagating User Preferences on the Knowledge Graph for Recommender Systems
date: 2019-10-30 10:41:20
tags: KG+rec
categories:
  recommender systems
---
# introduction
CF-->sparsity, the cold start-->side information-->knowledge graph

## KG's strength

* KG introduces semantic relatedness among items, which can help find their latent connections and improve the precision of recommended items;
* KG consists of relations with various types, which is helpful for extending a user’s interests reasonably and increasing the diversity of recommended items;
* KG connects a user’s historical records and
the recommended ones, thereby bringing explainability to recommender
systems.

## KG's categories:
1. Embedding-based methods: DKN, CKE, SHINE------Embedding-based methods show high flexibility in utilizing KG
to assist recommender systems, but the adopted KGE algorithms
in these methods are usually more suitable for in-graph applications
such as link prediction than for recommendation, thus
the learned entity embeddings are less intuitive and effective to
characterize inter-item relations.
2. Path-based methods: explore
the various patterns of connections among items in KG to
provide additional guidance for recommendations. PER, Meta-Graph
Based Recommendation.------Path-based
methods make use of KG in a more natural and intuitive way, but
they rely heavily on manually designed meta-paths, which is hard
to optimize in practice. Another concern is that it is impossible
to design hand-crafted meta-paths in certain scenarios (e.g., news
recommendation) where entities and relations are not within one
domain.

## RippleNet(CTR prediction)
The major difference
between RippleNet and existing literature is that RippleNet combines
the advantages of the above mentioned two types of methods:
(1) RippleNet incorporates the KGE methods into recommendation naturally by preference propagation; (2) RippleNet can automatically
discover possible paths from an item in a user’s history to a
candidate item, without any sort of hand-crafted design.

## contribution
* To the best of our knowledge, this is the first work to combine
embedding-based and path-based methods in KG-aware
recommendation.
* We propose RippleNet, an end-to-end framework utilizing
KG to assist recommender systems. RippleNet automatically
discovers users’ hierarchical potential interests by iteratively
propagating users’ preferences in the KG.
* We conduct experiments on three real-world recommendation
scenarios, and the results prove the efficacy

![](/ripplenet/1.JPG)
