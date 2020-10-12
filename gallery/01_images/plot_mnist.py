#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
"""
MNIST Dataset
=============


This example shows how to download/load/import MNIST
"""


import numpy_datasets as nds
import matplotlib.pyplot as plt

mnist = nds.images.mnist.load()

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, 1 + i)
    plt.imshow(mnist["train_set/images"][i, 0], aspect="auto", cmap="Greys")
    plt.xticks([])
    plt.yticks([])
    plt.title(str(mnist["train_set/labels"][i]))

plt.tight_layout()
