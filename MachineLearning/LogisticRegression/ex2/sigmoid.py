#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sigmoid函数
:Time:  2020/7/28 11:10
:Author:  lenjon
"""
import numpy as np


def sigmoid(z):
    g = np.zeros(z.shape)
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this
    z_ravel=z.ravel() # 按行展开——避免多次循环

    lenth=len(z_ravel)
    g=g.ravel()
    for ii in range(lenth):
        if z_ravel[ii]>=0:
            g[ii] = 1./(1. + np.exp(-z_ravel[ii]))
        else:
            g[ii] = np.exp(z_ravel[ii])/(1.+np.exp(z_ravel[ii]))
    # ========== END ==========
    return g.reshape(z.shape)
