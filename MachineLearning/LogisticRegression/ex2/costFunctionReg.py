#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
:Time:  2020/7/28 14:41
:Author:  lenjon
"""
import numpy as np
from sigmoid import sigmoid

from costFunction import costFunction


def costFunctionReg(theta, X, y, _lambda):
    theta = theta.reshape(1, -1)
    m = X.shape[0]
    J = 0
    grad = np.zeros(theta.shape)
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this
    # print(theta.shape)
    h = sigmoid(np.dot(X, theta.T))


    J = (-1 / m) * np.sum((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))) + (_lambda/(2*m)) * np.sum(theta[:,1:] ** 2)


    grad_ = (1 / m) *np.sum (np.dot((h - y).T, X[:,0]))

    grad = (1 / m) * (np.dot((h - y).T, X)) + (_lambda/m) * theta

    grad[:,0]=grad_
    # ========== END ==========
    return J, grad
