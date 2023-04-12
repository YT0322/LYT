import numpy as np


def computeCostMulti(X, y, theta):
    """
    同computeCost
    """
    theta = theta.reshape(1, -1)
    # print("theta shape",theta.shape)
    m = X.shape[0]
    #你需要正确的返回变量J
    J = 0
    pass
    # ================ YOUR CODE HERE ================
    # 根据代价函数公式编写代码，将结果存储在J中
    # 提示：python中a@b表示a叉乘b

    # 将矩阵θ转置 点乘 样本矩阵x
    h = np.dot(X, theta.T)

    #计算方差
    sumSqu = (h-y).T @ (h-y)
    J = 1/(2*m) * np.sum(sumSqu)


    return J