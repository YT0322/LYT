#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
"""
import numpy as np


def kMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    pass
    idx = np.random.randint(0, X.shape[0], size=K)
    centroids = X[idx, :]
    return centroids
