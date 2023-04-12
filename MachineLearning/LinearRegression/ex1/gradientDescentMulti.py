from computeCostMulti import computeCostMulti
import numpy as np


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    同gradientDescent
    """
    theta = theta.reshape(1, -1)
    J_history = np.zeros(num_iters)
    m = X.shape[0]

    for i in range(num_iters):
        pass
        # ================ YOUR CODE HERE ================
        # 迭代更新theta，计算不同theta时对应的代价函数
        # 提示：在调试时，在这里打印成本函数（computeCostMulti）和梯度的值非常有用。
        # 利用梯度下降公式更新theta
        # 利用computeCostMulti函数计算J_history

        # 利用梯度下降公式更新theta
        h = np.dot(X, theta.T)
        # print(h.shape)#(97,1)


        # 这里前面要转置 theta1数据维度决定了  (1,97)x(97,2) =(1,2)
        theta = theta - alpha / m * np.dot((h - y).T, X)

        # 利用computeCost函数计算J_history
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history
