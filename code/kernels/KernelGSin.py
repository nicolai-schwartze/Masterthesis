# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:21:10 2020

@author: Nicolai
"""

from IKernelBase import IKernelBase
import numpy as np

class KernelGSin(IKernelBase):
    """
    Implementation of abstract class IKernelBase where GSin Kernels are used. 
    
    .. math:: 
        
        w e^{-y ((x_0 - c_0)^2 + (x_1 - c_1)^2)} sin(f ((x_0 - c_0)^2 + (x_1 - c_1)^2)-p)
    
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
        self._kernel_type = "GSin Kernel: sum_{i}^{N}(w_i*e^(-y_i*((x_0 - c_0_i)^2 + (x_1 - c_1_i)^2)))sin(f_i*((x_0 - c_0_i)^2 + (x_1 - c_1_i)^2)-p_i)"
        self._kernel_size = 6
        
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
            r = (x[0] - kernels[i][2])**2 +(x[1] - kernels[i][3])**2
            exp = -1*kernels[i][1]*(r)
            s = np.sin(kernels[i][4]*r - kernels[i][5])
            result += kernels[i][0] * np.e**exp * s
            
        return result
    
    def solution_x0(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            w = kernels[i][0]
            g = kernels[i][1]
            c0= kernels[i][2]
            c1= kernels[i][3]
            f = kernels[i][4]
            p = kernels[i][5]
            r2= (x[0] - c0)**2 + (x[1] - c1)**2
            exp=(x[0] - c0)*np.e**(-g*(r2))
            result += 2*w*g*exp*np.sin(p - f*(r2)) + 2*w*f*exp*np.cos(p - f*(r2))

        return result
    
    def solution_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        for i in range(kernelNR):
            w = kernels[i][0]
            g = kernels[i][1]
            c0= kernels[i][2]
            c1= kernels[i][3]
            f = kernels[i][4]
            p = kernels[i][5]
            r2= (x[0] - c0)**2 + (x[1] - c1)**2
            exp=(x[1] - c1)*np.e**(-g*(r2))
            result += 2*w*g*exp*np.sin(p - f*(r2)) + 2*w*f*exp*np.cos(p - f*(r2))
            
        return result
    
    def solution_x0_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        y = x[1]
        x = x[0]
        for i in range(kernelNR):
            w = kernels[i][0]
            g = kernels[i][1]
            c0= kernels[i][2]
            c1= kernels[i][3]
            f = kernels[i][4]
            p = kernels[i][5]
            r2= (c0 - x)**2 + (c1 - y)**2
            result += 4*w*(c0 - x)*(c1 - y)*np.e**(-g*(r2))*((f**2 - g**2)* \
                           np.sin(p - f*(r2)) - 2*f*g*np.cos(p - f*(r2)))
        return result
    
    
    def solution_x0_x0(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        y = x[1]
        x = x[0]
        for i in range(kernelNR):
            w = kernels[i][0]
            g = kernels[i][1]
            c0= kernels[i][2]
            c1= kernels[i][3]
            f = kernels[i][4]
            p = kernels[i][5]
            r2= (c0 - x)**2 + (c1 - y)**2
            result += 2*w*np.e**(-g*(r2))*((2*c0**2*(f**2 - g**2) + 4*c0*x*(g**2 - f**2) \
                                     + 2*f**2*x**2 - 2*g**2*x**2 + g)*np.sin(p - f*(r2)) + \
                                 f*(-4*c0**2*g + 8*c0*g*x - 4*g*x**2 + 1)*np.cos(p - f*(r2)))
        return result
    
    
    def solution_x1_x1(self, kernels, x):
        kernelNR, dim = kernels.shape
        result = 0.0
        y = x[1]
        x = x[0]
        for i in range(kernelNR):
            w = kernels[i][0]
            g = kernels[i][1]
            c0= kernels[i][2]
            c1= kernels[i][3]
            f = kernels[i][4]
            p = kernels[i][5]
            r2= (c1 - y)**2 + (c0 - x)**2
            result += 2*w*np.e**(-g*(r2))*((2*c1**2*(f**2 - g**2) + 4*c1*y*(g**2 - f**2) \
                                     + 2*f**2*y**2 - 2*g**2*y**2 + g)*np.sin(p - f*(r2)) + \
                                 f*(-4*c1**2*g + 8*c1*g*y - 4*g*y**2 + 1)*np.cos(p - f*(r2)))
        return result
    
    
    
    
if __name__ == "__main__":
    
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    print("start test")
    
    kgs = KernelGSin()
    print(kgs.kernel_type)
    
    candidate_1 = np.array([1,1,0,0,1,0])
    candidate_1_reshaped = candidate_1.reshape((1,6))
    
    # show solution
    print("show solution")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3, 3.1, 0.005)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kgs.solution(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.savefig("gsk.pdf")
    plt.show()
    
    # show derivative with respect to x0
    print("show derivative to x0")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kgs.solution_x0(candidate_1_reshaped, \
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
    
    zs0 = np.array([kgs.solution_x1(candidate_1_reshaped, \
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
    
    zs0 = np.array([kgs.solution_x0_x0(candidate_1_reshaped, \
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
    
    zs0 = np.array([kgs.solution_x0_x1(candidate_1_reshaped, \
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
    
    zs0 = np.array([kgs.solution_x1_x1(candidate_1_reshaped, \
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    print("overflow test")
    candidate_o = np.array([[69.13155327, -59.50487635, 63.13495401, 72.31468988, 24.31468981, 4.8112859],
                            [12.9604027 , -76.7379638 , 55.64266812, 91.56222343, 14.31468982, 8.4634546],
                            [83.12853572, -60.83721539,  3.36485524, 51.36506458, 65.31468983, 2.4894392],
                            [79.46589204, -16.83238165, 13.40452466, 78.59279995, 34.31628983, 5.8237846],
                            [80.61433144, -45.23737621,  9.77667237, 93.48153471, 31.31864989, 3.4890687]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kgs.solution(candidate_o, \
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
    candidate_u = np.array([[69.13155327, 59.50487635, 63.13495401, 72.31468988, 24.31468981, 4.8112859],
                            [12.9604027 , 76.7379638 , 55.64266812, 91.56222343, 14.31468982, 8.4634546],
                            [83.12853572, 60.83721539,  3.36485524, 51.36506458, 65.31468983, 2.4894392],
                            [79.46589204, 16.83238165, 13.40452466, 78.59279995, 34.31628983, 5.8237846],
                            [80.61433144, 45.23737621,  9.77667237, 93.48153471, 31.31864989, 3.4890687]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-5, 5.1, 0.1)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kgs.solution(candidate_u, \
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
    