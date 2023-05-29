#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不是作业
"""


def featureNormalize(X):
    """
    求特征缩放
    :param X:
    :return: 每列平均值 标准差
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma
    return X, mu, sigma
