#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
"""

import numpy as np


def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this
    # 提示ravel()函数将数据拉伸为一维数组
    for i in range(K):
        # Find indices of examples assigned to centroid i
        assigned_examples = np.where(idx == i)[0]

        # Compute the mean of the assigned examples
        centroids[i] = np.mean(X[assigned_examples], axis=0)

    # ======================END ======================
    return centroids
