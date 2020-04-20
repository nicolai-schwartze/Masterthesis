# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:57:32 2020

@author: Nicolai
"""

import sys
sys.path.append("../")
from FemPdeBase import FemPdeBase
import numpy as np

# import from ngsolve
import ngsolve as ngs
from netgen.geom2d import unit_square
import time
import psutil

class FemPde2(FemPdeBase):
    """
    **Implementation of PDE2 of the testbed:** 
        
    .. math:: 
        - \Delta u(\mathbf{x}) = 2^{40}y^{10}(1-y)^{10}[90x^8(1-x)^{10} 
        
        - 200x^9(1-x)^9 + 90x^{10}(1-x)^8] 
        
        -2x^{10}(1-x)^{10}[90y^8(1-y)^{10} 
        
        - 200y^9(1-y)^9 + 90y^{10}(1-y)^8]
        
        \Omega: \mathbf{x} \in [0,1]
        
        u(\mathbf{x})|_{\partial \Omega} = 0
    
    **with the solution:** 
        
    .. math:: 
        u(\mathbf{x}) = 2^{40}x^{10}(1-x)^{10}y^{10}(1-y)^{10}
        
        
    Attributes
    ----------
    max_nodf: int
        the maximum number of degrees of freedom that can be created in the 
        adaptive mesh refinement, standard value is 50000
        
    Methods
    -------
    solve()
        solves the pde by calling ngsolve, provides: static condensation, 
        adaptive mesh refinement, parallelisation (where possible), sets the 
        internal variables for evaluating the exact solution and calculating 
        the distance between exact and approx solution
        also sets execution time and memory consumption
        
    Examples
    --------
    >>> import numpy as np
    >>> fempde2 = FemPde2(True)
    >>> pos = np.array([0.5, 0.5])
    >>> fempde2.exact(pos)
    >>> x -> numpy.ndarray with shape (2,) 
        _mesh -> ngs.comp.Mesh 
        _ngs_ex -> ngs.fem.CoefficientFunction 
        -> try to call solve() first
    >>> fempde2.solve()
    >>> fempde2.exact(pos)
        1.0
    >>> fempde2.approx(pos)
        0.999998924259486
    >>> fempde2.normL2()
        5.853102150391562e-07
    >>> fempde2.exec_time
        3.830256175994873
    >>> fempde2.mem_consumption
        76705792
    
    
    """
    
    def __init__(self, show_gui, max_ndof=50000):
        super().__init__(show_gui)
        
        # init protected
        self._pde_string = "-laplacian(u(x)) = -(2^40*y^10*(1-y)^10*(90*x^8*(1-x)^10 - 200*x^9*(1-x)^9 + 90*x^10*(1-x)^8)) -(2*x^10*(1-x)^10*(90*y^8*(1-y)^10 - 200*y^9*(1-y)^9 + 90*y^10*(1-y)^8))"
        self._ngs_ex = (2**(4*10))*(ngs.x**10)*((1-ngs.x)**10)*(ngs.y**10)*((1-ngs.y)**10)
        
        # init public
        self.max_ndof = max_ndof
        
        
    
    def solve(self): 
        # measure how much memory is used until here
        process = psutil.Process()
        memstart = process.memory_info().rss
        
        # starts timer
        tstart = time.time()
        if self.show_gui:
            import netgen.gui
        
        # create mesh with initial size 0.1
        self._mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
        
        #create finite element space
        self._fes = ngs.H1(self._mesh, order=2, dirichlet=".*", autoupdate=True)
        
        # test and trail function
        u = self._fes.TrialFunction()
        v = self._fes.TestFunction()
        
        # create bilinear form and enable static condensation
        self._a = ngs.BilinearForm(self._fes, condense=True)
        self._a += ngs.grad(u)*ngs.grad(v)*ngs.dx
    
        # creat linear functional and apply RHS
        self._f = ngs.LinearForm(self._fes)
        self._f += ( \
        -(2**40*ngs.y**10*(1-ngs.y)**10*(90*ngs.x**8*(1-ngs.x)**10 - 200*ngs.x**9*(1-ngs.x)**9 + 90*ngs.x**10*(1-ngs.x)**8)) \
        -(2**40*ngs.x**10*(1-ngs.x)**10*(90*ngs.y**8*(1-ngs.y)**10 - 200*ngs.y**9*(1-ngs.y)**9 + 90*ngs.y**10*(1-ngs.y)**8)) )*v*ngs.dx
        
        # preconditioner: multigrid - what prerequisits must the problem have? 
        self._c = ngs.Preconditioner(self._a,"multigrid")
        
        # create grid function that holds the solution and set the boundary to 0
        self._gfu = ngs.GridFunction(self._fes, autoupdate=True)  # solution 
        self._g = 0.0
        self._gfu.Set(self._g, definedon=self._mesh.Boundaries(".*"))
        
        # draw grid function in gui
        if self.show_gui:
            ngs.Draw(self._gfu)
        
        # create Hcurl space for flux calculation and estimate error
        self._space_flux = ngs.HDiv(self._mesh, order=2, autoupdate=True)
        self._gf_flux = ngs.GridFunction(self._space_flux, "flux", autoupdate=True)
        
        # TaskManager starts threads that (standard thread nr is numer of cores)
        with ngs.TaskManager():
            # this is the adaptive loop
            while self._fes.ndof < self.max_ndof:
                self._solveStep()
                self._estimateError()
                self._mesh.Refine()
        
        # since the adaptive loop stopped with a mesh refinement, the gfu must be 
        # calculated one last time
        self._solveStep()
        if self.show_gui:
            ngs.Draw(self._gfu)
            
        # set measured exectution time
        self._exec_time = time.time() - tstart
        
        # set measured used memory
        process = psutil.Process()
        memstop = process.memory_info().rss - memstart
        self._mem_consumption = memstop
        


if __name__ == "__main__":
    
    fempde2 = FemPde2(True)
    print(fempde2.pde_string)
    
    try:
        fempde2.exact(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    try:
        fempde2.approx(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    fempde2.solve()
    
    print("-------------------------------------")
    
    print("exact(0.5, 0.5) = {}".format(fempde2.exact(np.array([0.5,0.5]))))
    print("approx(0.5, 0.5) = {}".format(fempde2.approx(np.array([0.5,0.5]))))
    print("L2 norm to the real solution {}".format(fempde2.normL2()))
    print("solving took {} sec".format(fempde2.exec_time))
    print("solving uses {} Mb".format(fempde2.mem_consumption/1000000))
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde2.exact(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde2.approx(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    