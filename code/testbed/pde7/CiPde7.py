# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:18:57 2020

@author: Nicolai
"""

import sys
import os
importpath = os.path.abspath(__file__) + "../../../"
sys.path.append(importpath)
from CiPdeBase import CiPdeBase

import numpy as np

class CiPde7(CiPdeBase):
    """
    **Implementation of PDE7 of the testbed:** 
        
    .. math:: 
        - \Delta u(\mathbf{x}) = 0.24 x^{-1.4}
        
        \Omega: \mathbf{x} \in [0,1]
        
        u(\mathbf{x})|_{\partial \Omega} = u(\mathbf{x})
    
    **with the solution:** 
        
    .. math:: 
        u(\mathbf{x}) = x^{0.6}
        
    Attributes
    ----------
    sol_kernel: np.array
        n by 4 dimensional array for gauss kerenl\n
        n by 6 dimensional array for gsin kernel \n
        n is the number of kernels\n
        4/6 are the parameters of the kernel\
        [wi, yi, c1i, c2i, fi, pi]
        
    Methods
    -------
    solve()
        tries to solve the pde by calling the opt_algo.opt function on the 
        objective function fitness_function
        the time to execute this function is stored in _exec_time
        the memory used for this process is stored in _mem_consumption
        
    Examples
    --------
    >>> import numpy as np
    >>> import sys
    >>> sys.path.append("../")
    >>> sys.path.append("../../opt_algo")
    >>> sys.path.append("../../kernels")
    >>> import OptAlgoMemeticJADE as oaMemJade
    >>> import KernelGauss as gk
    >>> initialPop = 1*np.random.rand(80,40)
    >>> max_iter = 10**4
    >>> min_err = 10**(-200)
    >>> mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_iter, min_err) 
    >>> gkernel = gk.KernelGauss()
    >>> cipde1 = CiPde1(mJade, gkernel)
    >>> pos = np.array([0.5, 0.5])
    >>> cipde1.exact(pos)
        1.0
    >>> cipde1.solve()
    >>> cipde1.approx(pos)
        0.0551194418735029
    >>> cipde1.normL2()
        0.471414887516362
    >>> cipde1.exec_time
        11886.15
    >>> cipde1.mem_consumption
        180473856
    >>> cipde1.sol_kernel 
        array([[ 0.18489863,  0.80257658,  2.73320428,  1.4806761 ],
               [ 0.03794604,  0.4414469 , -0.21652954, -0.29846278],
               [ 0.05160915,  3.60778814,  0.46935849,  0.49860103],
               [ 0.64412968,  0.12194749,  4.12595979, -0.80986777],
               [ 0.22469726,  0.91309473,  0.85266394,  0.3984538 ],
               [ 0.56367422,  0.52322707, -0.56848659, -0.25252624],
               [-0.21006541,  0.45991996,  1.03983463,  0.12103236],
               [-0.31711126,  0.12152396,  1.64290433,  1.46009719],
               [-0.64298078,  0.39706295, -0.76325605,  0.01840455],
               [ 0.31188508,  0.42570347, -0.16890337,  1.48988639]])
    """
    
    def __init__(self, opt_algo, kernel, nb, nc):
        # initialisation from CiPdeBase
        super().__init__(opt_algo, kernel, nb, nc)
        
        # descriptive string
        self._pde_string = "-laplacian(u(x)) = 0.24*x^{-1.4}"
        
        # user-defined inner weighting factor
        self._kappa = 1
        
        # inner weights for collocation points
        self._xi = []
        for xi in self._nc:
            self._xi.append(self.nc_weight(xi))
        
        # weighting factor on bounday points
        self._phi = []
        for xb in self._nb:
            self._phi.append(100)
        
        # boundary for integration in L2 Norm
        self._lx = 0.0
        self._ux = 1.0
        self._ly = lambda x: 0.0
        self._uy = lambda x: 1.0
        
    def exact(self, x): 
        y = x[1]
        x = x[0]
        return x**0.6 + y*0
    
    def fitness_func(self, kernels): 
        
        dimension = kernels.shape[0]
        numberOfKernels = dimension/self.kernel.kernel_size
        # candidate solution coding: 
        # [wi, yi, c1i, c2i] where i is the number of kernels
        kernels = kernels.reshape((int(numberOfKernels), self.kernel.kernel_size))
        inner_sum = 0.0
        for i in range(len(self._nc)):
            x = self._nc[i][0]
            y = self._nc[i][1]
            u_x0_x0 = self.kernel.solution_x0_x0(kernels, np.array([x,y]))
            u_x1_x1 = self.kernel.solution_x1_x1(kernels, np.array([x,y]))
            f = 0.24*(x**(-1.4))
            inner_sum += self._xi[i]*(-u_x0_x0 - u_x1_x1 - f)**2 
        
        boarder_sum = 0.0
        for i in range(len(self._nb)):
            x = self._nb[i][0]
            y = self._nb[i][1]
            boarder_sum += self._phi[i]*(self.exact(np.array([x, y])) - self.kernel.solution(kernels, np.array([x, y])))**2
        
        return (boarder_sum + inner_sum)/(len(self._nb) + len(self._nc))
        


if __name__ == "__main__":
    
    import sys
    sys.path.append("../")
    sys.path.append("../../opt_algo")
    sys.path.append("../../kernels")
    import OptAlgoMemeticJADE as oaMemJade
    import KernelGauss as gk
    
    initialPop = 1*np.random.rand(40,20)
    max_iter = 5*10**2
    min_err = 10**(-200)
    mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_iter, min_err)
    
    gkernel = gk.KernelGauss()
    
    # collocation points
    nc = []
    omega = np.arange(0.1, 1.0, 0.1)
    for x0 in omega:
        for x1 in omega:
            nc.append((x0, x1))
        
    # boundary points
    nb = []
    nby = np.hstack((np.zeros(10), np.arange(0.0, 1.0, 0.1), np.ones(10), np.arange(1.0, 0.0, -0.1)))
    nbx = np.hstack((np.arange(0.0, 1.0, 0.1), np.ones(10), np.arange(1.0, 0.0, -0.1), np.zeros(10)))
    for i in range(40):
        nb.append((nbx[i], nby[i]))
    
    cipde7 = CiPde7(mJade, gkernel, nb, nc)
    
    print(cipde7.pde_string)
    
    try:
        cipde7.exact(np.array([0.0,0.0]))
    except:
        print("Î error message above")
    
    try:
        cipde7.approx(np.array([0.0,0.0]))
    except:
        print("Î error message above")
    
    cipde7.solve()
    
    print("-------------------------------------")
    
    print("exact(0.5, 0.5) = {}".format(cipde7.exact(np.array([0.5,0.5]))))
    print("approx(0.5, 0.5) = {}".format(cipde7.approx(np.array([0.5,0.5]))))
    print("L2 norm to the real solution {}".format(cipde7.normL2()))
    print("solving took {} sec".format(cipde7.exec_time))
    print("solving uses {} Mb".format(cipde7.mem_consumption/1000000))
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0.0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([cipde7.exact(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0.0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([cipde7.approx(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    