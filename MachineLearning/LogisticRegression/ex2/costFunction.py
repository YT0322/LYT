#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
"""
import numpy as np
import math

from sigmoid import sigmoid


def costFunction(theta, X, y):
    # Initialize
    theta = theta.reshape(1, -1)
    m = X.shape[0]
    J = 0
    grad = np.zeros(theta.shape)
    pass
    # ====================== YOUR CODE HERE ======================
    # You should change this
    h = sigmoid(np.dot(X, theta.T))
    J = (-1 / m) * np.sum((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))))

    grad = (1 / m) *(np.dot((h - y).T,X))

    # ========== END ==========
    return J, grad
