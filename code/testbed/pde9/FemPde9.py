# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:57:37 2020

@author: Nicolai
"""

import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(importpath)
from FemPdeBase import FemPdeBase
import numpy as np

# import from ngsolve
import ngsolve as ngs
from netgen.geom2d import unit_square

import time
import psutil
import gc

class FemPde9(FemPdeBase):
    """
    **Implementation of PDE9 of the testbed:** 
        
    .. math:: 
        -\Delta u(\mathbf{x}) = 
        
        (20\sqrt{2}(x^2 + y^2 -2x^2y - 2xy^2 + 4xy - x - y))/(400((x+y)/(\sqrt{2})-0.8)^2+1) 
        
        + (16000(1-x)x(1-y)y((x+y)(\sqrt{2})-0.8))/((400((x+y)/(\sqrt{2})-0.8)^2+1)^2) 
        
        + tan^{-1}(20((x+y)/(\sqrt{2})-0.8))(2(1-y)y + 2(1-x)x)
        
        \Omega: \mathbf{x} \in [0,1]
        
        u(\mathbf{x})|_{\partial \Omega} = u(\mathbf{x})
    
    **with the solution:** 
        
    .. math:: 
        u(\mathbf{x}) = tan^{-1}(20((x + y)/\sqrt{2} -0.8))x(1-x)y(1-y)
        
        
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
        self._pde_string = """-laplacian(u(x)) = (20\sqrt{2}(x^2 + y^2 -2x^2y - 2xy^2 + 4xy - x - y))/(400((x+y)/(\sqrt{2})-0.8)^2+1) + 
                   (16000(1-x)x(1-y)y((x+y)(\sqrt{2})-0.8))/((400((x+y)/(\sqrt{2})-0.8)^2+1)^2) + 
                   tan^{-1}(20((x+y)/(\sqrt{2})-0.8))(2(1-y)y + 2(1-x)x)"""
        self._ngs_ex = ngs.atan(20*((ngs.x + ngs.y)/(2**(1/2)) -0.8))*ngs.x*(1-ngs.x)*ngs.y*(1-ngs.y)
        
        # init public
        self.max_ndof = max_ndof
        
        
    
    def solve(self): 
        
        # disable garbage collector 
        # --------------------------------------------------------------------#
        gc.disable()
        while(gc.isenabled()):
            time.sleep(0.1)
        # --------------------------------------------------------------------#
        
        # measure how much memory is used until here
        process = psutil.Process()
        memstart = process.memory_info().vms
        
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
        self._a += (ngs.grad(u)*ngs.grad(v))*ngs.dx
        
        # creat linear functional and apply RHS
        self._f = ngs.LinearForm(self._fes)
        self._f += ((20*ngs.sqrt(2)*(ngs.x**2 + ngs.y**2 -2*ngs.x**2*ngs.y - 2*ngs.x*ngs.y**2 + 4*ngs.x*ngs.y - ngs.x - ngs.y))/(400*((ngs.x+ngs.y)/(ngs.sqrt(2))-0.8)**2+1) + \
                    (16000*(1-ngs.x)*ngs.x*(1-ngs.y)*ngs.y*((ngs.x+ngs.y)/(ngs.sqrt(2))-0.8))/((400*((ngs.x+ngs.y)/(ngs.sqrt(2))-0.8)**2+1)**2) + \
                    ngs.atan(20*((ngs.x+ngs.y)/(ngs.sqrt(2))-0.8))*(2*(1-ngs.y)*ngs.y + 2*(1-ngs.x)*ngs.x))*v*ngs.dx
        
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
        memstop = process.memory_info().vms - memstart
        self._mem_consumption = memstop
        
        # enable garbage collector 
        # --------------------------------------------------------------------#
        gc.enable()
        gc.collect()
        # --------------------------------------------------------------------#
        


if __name__ == "__main__":
    
    fempde9 = FemPde9(True, max_ndof=100000)
    print(fempde9.pde_string)
    
    try:
        fempde9.exact(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    try:
        fempde9.approx(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    fempde9.solve()
    
    print("-------------------------------------")
    
    print("exact(0.5, 0.5) = {}".format(fempde9.exact(np.array([0.5,0.5]))))
    print("approx(0.5, 0.5) = {}".format(fempde9.approx(np.array([0.5,0.5]))))
    print("L2 norm to the real solution {}".format(fempde9.normL2()))
    print("solving took {} sec".format(fempde9.exec_time))
    print("solving uses {} Mb".format(fempde9.mem_consumption/1000000))
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde9.exact(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    fig.tight_layout()
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0, X1)")
    plt.show()
    fig.savefig("sol_pde_9.pdf", bbox_inches='tight')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde9.approx(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()
    