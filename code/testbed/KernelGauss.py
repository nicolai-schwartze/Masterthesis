# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:06:10 2020

@author: Nicolai
"""

from IKernelBase import IKernelBase
import numpy as np

class KernelGauss(IKernelBase):
    
    
    def __init__(self):
        self._kernel_type = "Gauss Kernel: sum_{i}^{N}(w_i*e^(-y_i*((x_0 - c_1)^2 + (x_1 - c_1)^2))"
        
    @property
    def kernel_type(self):
        return self._kernel_type
    
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
    
    candidate_1 = np.array([1,0.2,0.5,0.5])
    candidate_1_reshaped = candidate_1.reshape((1,4))
    
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
    
    
    print("finised test")
    