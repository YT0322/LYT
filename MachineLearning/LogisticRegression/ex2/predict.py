#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters.
%               You should set p to a vector of 0's and 1's
:Time:  2020/7/28 13:40
:Author:  lenjon
"""
import numpy as np
from costFunction import costFunction

from sigmoid import sigmoid


def predict(theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this

    p = sigmoid(X @ theta)

    #大于0.5 value=1；反之 value=0
    shape_p = p.shape
    p = p.ravel()
    len_p = len(p)
    for ii in range(len_p):
        if p[ii] >= 0.5:
            p[ii] = 1
        else:
            p[ii] = 0
    p = p.reshape(shape_p)

    # ========== END ==========
    return p
