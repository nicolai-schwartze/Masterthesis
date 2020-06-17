# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:06:10 2020

@author: Nicolai
"""

from IKernelBase import IKernelBase
import numpy as np

class KernelGauss(IKernelBase):
    """
    Implementation of abstract class IKernelBase where Gaussian Kernels are used. 
    
    .. math:: 
        
        w e^{-y ((x_0 - c_0)^2 + (x_1 - c_1)^2)}
    
    Atributes
    ---------
    _kernel_type: string
        descriptive string of the kernel formula
    _kernel_size: int
        number of parameters to be optimised in one kernel
    
    Methods
    -------
    kernel_type(): string
        getter for the attribute kernel_type
    kernel_size(): int
        getter of the attribute kerel_size
    solution(kernels, x): float
        linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0(kernels, x): float
        derivative with respect to x0 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x1(kernels, x): float
        derivative with respect to x1 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0_x0(kernels, x): float
        second derivative with respect to x0 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x1_x1(kernels, x): float
        second derivative with respect to x1 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0_x1(kernels, x): float
        second derivative with respect to x0,x1 of linear combination of kernels specified in arguemnt, evaluated at x
    """
    
    def __init__(self):
        self._kernel_type = "Gauss Kernel: sum_{i}^{N}(w_i*e^(-y_i*((x_0 - c_0_i)^2 + (x_1 - c_1_i)^2)))"
        self._kernel_size = 4
        
    @property
    def kernel_type(self):
        return self._kernel_type
    
    @property
    def kernel_size(self):
        return self._kernel_size
    
    def solution(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR): 
            exponent = -1*kernels[i][1]*((x[0] - kernels[i][2])**2 \
            +(x[1] - kernels[i][3])**2)
            result += kernels[i][0] * np.e**exponent
            
        return result
    
    def solution_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            ci = np.array([kernels[i][2], kernels[i][3]])
            wi = kernels[i][0]
            yi = kernels[i][1]
            result += \
            wi*yi*(x[1] - ci[1])* \
            np.e**(-1*yi * np.linalg.norm(x - ci)**2)
        return -2*result
    
    def solution_x0(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            ci = np.array([kernels[i][2], kernels[i][3]])
            wi = kernels[i][0]
            yi = kernels[i][1]
            result += \
            wi*yi*(x[0] - ci[0])* \
            np.e**(-1*yi * np.linalg.norm(x - ci)**2)
        return -2*result
    
    def solution_x0_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            ci = np.array([kernels[i][2], kernels[i][3]])
            wi = kernels[i][0]
            yi = kernels[i][1]
            result += \
            wi*4*yi*yi*((x[0] - kernels[i][2])*(x[1] - kernels[i][3])) * \
            np.e**(-1*yi * np.linalg.norm(x - ci)**2)
        return result
    
    
    def solution_x0_x0(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            ci = np.array([kernels[i][2], kernels[i][3]])
            wi = kernels[i][0]
            yi = kernels[i][1]
            result += \
            wi*yi*(4*yi*(x[0] - kernels[i][2])**2 - 2) * \
            np.e**(-1*yi * np.linalg.norm(x - ci)**2)
        return result
    
    
    def solution_x1_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            ci = np.array([kernels[i][2], kernels[i][3]])
            wi = kernels[i][0]
            yi = kernels[i][1]
            result += \
            wi*yi*(4*yi*(x[1] - kernels[i][3])**2 - 2) * \
            np.e**(-1*yi * np.linalg.norm(x - ci)**2)
        return result
    
    
    
    
if __name__ == "__main__":
    
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    print("start test")
    
    kg = KernelGauss()
    print(kg.kernel_type)
    
    candidate_1 = np.array([1,0.01,2,3,-1,20,2,-2])
    candidate_1_reshaped = candidate_1.reshape((2,4))
    
    # show solution
    print("show solution")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.savefig("aliasing_error.pdf")
    plt.show()
    
    # show derivative with respect to x0
    print("show derivative to x0")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution_x0(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    # show derivative with respect to x1
    print("show derivative to x1")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([-kg.solution_x1(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    # show derivative with respect to x0 x0
    print("show derivative to x0 x0")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution_x0_x0(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    print("show derivative to x0 x1")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution_x0_x1(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    # show derivative with respect to x0 x1
    print("show derivative to x1 x1")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution_x1_x1(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    print("overflow test")
    candidate_o = np.array([[69.13155327, -59.50487635, 63.13495401, 72.31468988],
                            [12.9604027 , -76.7379638 , 55.64266812, 91.56222343],
                            [83.12853572, -60.83721539,  3.36485524, 51.36506458],
                            [79.46589204, -16.83238165, 13.40452466, 78.59279995],
                            [80.61433144, -45.23737621,  9.77667237, 93.48153471]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution(candidate_o, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    print("max(zs0) = {}".format(max(zs0)))
    print("min(zs0) = {}".format(min(zs0)))
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    print("underflow test")
    candidate_u = np.array([[69.13155327, 59.50487635, 63.13495401, 72.31468988],
                            [12.9604027 , 76.7379638 , 55.64266812, 91.56222343],
                            [83.12853572, 60.83721539,  3.36485524, 51.36506458],
                            [79.46589204, 16.83238165, 13.40452466, 78.59279995],
                            [80.61433144, 45.23737621,  9.77667237, 93.48153471]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kg.solution(candidate_u, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    print("max(zs0) = {}".format(max(zs0)))
    print("min(zs0) = {}".format(min(zs0)))
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    print("finised test")
    