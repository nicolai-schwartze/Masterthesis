# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:57:32 2020

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
import netgen.geom2d as geom2d

import time
import psutil
import gc

class FemPde0B(FemPdeBase):
    """
    **Implementation of PDE0A of the testbed:** 
        
    .. math:: 
        \Delta u(\mathbf{x}) = 2 e^{-2   (x^2 + y^2)} (2   sin(-2   (x^2 + y^2)) + 2   (1-8   x^2) cos(-2   (x^2 + y^2))) + 
        
        2 e^{-2   (x^2 + y^2)} (2   sin(-2   (x^2 + y^2)) + 2   (1-8   y^2) cos(-2   (x^2 + y^2))) + 
        
        2 e^{-1   (x^2 + y^2)} (1   sin(-1   (x^2 + y^2)) + 1   (1-4   x^2) cos(-1   (x^2 + y^2))) + 
        
        2 e^{-1   (x^2 + y^2)} (1   sin(-1   (x^2 + y^2)) + 1   (1-4   y^2) cos(-1   (x^2 + y^2))) + 
        
        2 e^{-0.1 (x^2 + y^2)} (0.1 sin(-0.1 (x^2 + y^2)) + 0.1 (1-0.4 x^2) cos(-0.1 (x^2 + y^2))) + 
        
        2 e^{-0.1 (x^2 + y^2)} (0.1 sin(-0.1 (x^2 + y^2)) + 0.1 (1-0.4 y^2) cos(-0.1 (x^2 + y^2))) 
                                 
                                 
        
        \Omega: \mathbf{x} \in [-2,2]
        
        u(\mathbf{x})|_{\partial \Omega} = u(\mathbf{x})
        
    **with the solution:** 
        
    .. math:: 
        u(\mathbf{x}) = e^{-2 (x^2 + y^2)} sin(2  (x^2 + y^2)) + 
        
                        e^{-1 (x^2 + y^2)} sin(1  (x^2 + y^2)) + 
                        
                        e^{-0.1 (x^2 + y^2)} sin(0.1(x^2 + y^2))
        
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
    >>> fempde0A = FemPde0A(True)
    >>> pos = np.array([0.0, 0.0])
    >>> fempde0A.exact(pos)
    >>> x -> numpy.ndarray with shape (2,) 
        _mesh -> ngs.comp.Mesh 
        _ngs_ex -> ngs.fem.CoefficientFunction 
        -> try to call solve() first
    >>> fempde0A.solve()
    >>> fempde0A.exact(pos)
        2.009915008706665
    >>> fempde0A.approx(pos)
        2.009914779748446
    >>> fempde0A.normL2()
        2.9670770746774782e-05
    >>> fempde0A.exec_time
        6.375466346740723
    >>> fempde0A.mem_consumption
        159.055872 Mb
    
    
    """
    
    def __init__(self, show_gui, max_ndof=50000):
        super().__init__(show_gui)
        
        # init protected
        self._pde_string = """laplacian(u(x)) = 2*np.exp(-2  (x^2 + y^2))*(2 * np.sin(-2 (x^2 + y^2)) + 2 (1-8 * x^2) np.cos(-2 * (x^2 + y^2))) + \
                   2*np.exp(-2  (x^2 + y^2))*(2 * np.sin(-2  (x^2 + y^2)) + 2  (1-8 * y^2) np.cos(-2 * (x^2 + y^2))) + \
                   2*np.exp(-1  (x^2 + y^2))*(1 * np.sin(-1  (x^2 + y^2)) + 1  (1-4 * x^2) np.cos(-1 * (x^2 + y^2))) + \
                   2*np.exp(-1  (x^2 + y^2))*(1 * np.sin(-1  (x^2 + y^2)) + 1  (1-4 * y^2) np.cos(-1 * (x^2 + y^2))) + \
                   2*np.exp(-0.1(x^2 + y^2))*(0.1*np.sin(-0.1(x^2 + y^2)) + 0.1(1-0.4*x^2) np.cos(-0.1*(x^2 + y^2))) + \
                   2*np.exp(-0.1(x^2 + y^2))*(0.1*np.sin(-0.1(x^2 + y^2)) + 0.1(1-0.4*y^2) np.cos(-0.1*(x^2 + y^2)))  """
        
        self._ngs_ex = ngs.exp(-2  * ((ngs.x)**2 + (ngs.y)**2))*ngs.sin(2  * ((ngs.x)**2 + (ngs.y)**2)) + \
                       ngs.exp(-1  * ((ngs.x)**2 + (ngs.y)**2))*ngs.sin(1  * ((ngs.x)**2 + (ngs.y)**2)) + \
                       ngs.exp(-0.1* ((ngs.x)**2 + (ngs.y)**2))*ngs.sin(0.1* ((ngs.x)**2 + (ngs.y)**2))
        
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
        geo = geom2d.SplineGeometry()
        p1 = geo.AppendPoint (-2,-2)
        p2 = geo.AppendPoint (2,-2)
        p3 = geo.AppendPoint (2,2)
        p4 = geo.AppendPoint (-2,2)
        geo.Append (["line", p1, p2])
        geo.Append (["line", p2, p3])
        geo.Append (["line", p3, p4])
        geo.Append (["line", p4, p1])
        self._mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))
        
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
        self._f += - ( \
                     2*ngs.exp(-2 * (ngs.x**2 + ngs.y**2))*(2 * ngs.sin(-2 * (ngs.x**2 + ngs.y**2)) + 2 * (1-8 * ngs.x**2)*ngs.cos(-2 * (ngs.x**2 + ngs.y**2))) + \
                     2*ngs.exp(-2 * (ngs.x**2 + ngs.y**2))*(2 * ngs.sin(-2 * (ngs.x**2 + ngs.y**2)) + 2 * (1-8 * ngs.y**2)*ngs.cos(-2 * (ngs.x**2 + ngs.y**2))) + \
                     2*ngs.exp(-1 * (ngs.x**2 + ngs.y**2))*(1 * ngs.sin(-1 * (ngs.x**2 + ngs.y**2)) + 1 * (1-4 * ngs.x**2)*ngs.cos(-1 * (ngs.x**2 + ngs.y**2))) + \
                     2*ngs.exp(-1 * (ngs.x**2 + ngs.y**2))*(1 * ngs.sin(-1 * (ngs.x**2 + ngs.y**2)) + 1 * (1-4 * ngs.y**2)*ngs.cos(-1 * (ngs.x**2 + ngs.y**2))) + \
                     2*ngs.exp(-0.1*(ngs.x**2 + ngs.y**2))*(0.1*ngs.sin(-0.1*(ngs.x**2 + ngs.y**2)) + 0.1*(1-0.4*ngs.x**2)*ngs.cos(-0.1*(ngs.x**2 + ngs.y**2))) + \
                     2*ngs.exp(-0.1*(ngs.x**2 + ngs.y**2))*(0.1*ngs.sin(-0.1*(ngs.x**2 + ngs.y**2)) + 0.1*(1-0.4*ngs.y**2)*ngs.cos(-0.1*(ngs.x**2 + ngs.y**2)))
                     )*v*ngs.dx
        
        # preconditioner: multigrid - what prerequisits must the problem have? 
        self._c = ngs.Preconditioner(self._a,"multigrid")
        
        # create grid function that holds the solution and set the boundary to 0
        self._gfu = ngs.GridFunction(self._fes, autoupdate=True)  # solution 
        self._g = self._ngs_ex
        self._gfu.Set(self._g, definedon=self._mesh.Boundaries(".*"))
        
        # draw grid function in gui
        if self.show_gui:
            ngs.Draw(self._gfu)
        
        # create Hcurl space for flux calculation and estimate error
        self._space_flux = ngs.HDiv(self._mesh, order=2, autoupdate=True)
        self._gf_flux = ngs.GridFunction(self._space_flux, "flux", autoupdate=True)
        
        # TaskManager starts threads that (standard thread nr is numer of cores)
        #with ngs.TaskManager():
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
    
    fempde0B = FemPde0B(True)
    print(fempde0B.pde_string)
    
    try:
        fempde0B.exact(np.array([0.0,0.0]))
    except:
        print("Î error message above")
    
    try:
        fempde0B.approx(np.array([0.0,0.0]))
    except:
        print("Î error message above")
    
    fempde0B.solve()
    
    print("-------------------------------------")
    
    print("exact(0.0, 0.0) = {}".format(fempde0B.exact(np.array([0.0,0.0]))))
    print("approx(0.0, 0.0) = {}".format(fempde0B.approx(np.array([0.0,0.0]))))
    print("L2 norm to the real solution {}".format(fempde0B.normL2()))
    print("solving took {} sec".format(fempde0B.exec_time))
    print("solving uses {} Mb".format(fempde0B.mem_consumption/1000000))
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-2.0, 2.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde0B.exact(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    fig.tight_layout()
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0, X1)")
    
    plt.show()
    fig.savefig("sol_pde_0b.pdf", bbox_inches='tight')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-2.0, 2.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([fempde0B.approx(\
    np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.show()