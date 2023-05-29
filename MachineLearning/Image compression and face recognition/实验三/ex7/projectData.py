#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: Compute the projection of the data using only the top K
%               eigenvectors in U (first K columns).
%               For the i-th example X(i,:), the projection on to the k-th
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
"""
import numpy as np


def projectData(X, U, K):
    # 找K个最能代表原数据的向量
    Z = np.zeros((X.shape[0], K))
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this
    for i in range(X.shape[0]):
        x = X[i, :]  # 检索第i个示例
        projection = x @ U[:, :K]  # 投影到前K个特征向量上
        Z[i, :] = projection
    # ======================END ======================
    return Z
