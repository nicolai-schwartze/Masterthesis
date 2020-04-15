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

class FemPde1(FemPdeBase):
    """
    Implementation of PDE1 of the testbed: 
        
    .. math:: 
        - \Delta u(\mathbf{x}) = 2\pi^2 sin(\pi x_0) sin(\pi x_1) 
        
        \Omega: \mathbf{x} \in [0,1]
        
        u(\mathbf{x})|_{\partial \Omega} = 0
        
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
    >>> fempde1 = FemPde1(True)
    >>> pos = np.array([0.5, 0.5])
    >>> fempde1.exact(pos)
    >>> x -> numpy.ndarray with shape (2,) 
        _mesh -> ngs.comp.Mesh 
        _ngs_ex -> ngs.fem.CoefficientFunction 
        -> try to call solve() first
    >>> fempde1.solve()
    >>> fempde1.exact(pos)
        1.0
    >>> fempde1.approx(pos)
        0.9999999839860894
    >>> fempde1.normL2()
        3.4717202708948315e-07
    >>> fempde1.exec_time
        7.630256175994873
    >>> fempde1.mem_consumption
        166629376
    
    
    """
    
    def __init__(self, show_gui, max_ndof=50000):
        super().__init__(show_gui)
        
        # init protected
        self._pde_string = "-laplacian(u(x)) = 2*(pi**2)*sin(pi*x)*sin(pi*y)"
        self._ngs_ex = ngs.sin(np.pi*ngs.x)*ngs.sin(np.pi*ngs.y)
        
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
        self._fes = ngs.H1(self._mesh, order=2, dirichlet=".*")
        
        # test and trail function
        u = self._fes.TrialFunction()
        v = self._fes.TestFunction()
        
        # create bilinear form and enable static condensation
        self._a = ngs.BilinearForm(self._fes, condense=True)
        self._a += ngs.grad(u)*ngs.grad(v)*ngs.dx
    
        # creat linear functional and apply RHS
        self._f = ngs.LinearForm(self._fes)
        self._f += 2*(np.pi**2)*ngs.sin(np.pi*ngs.x)*ngs.sin(np.pi*ngs.y)*v*ngs.dx
        
        # preconditioner: multigrid - what prerequisits must the problem have? 
        self._c = ngs.Preconditioner(self._a,"multigrid")
        
        # create grid function that holds the solution and set the boundary to 0
        self._gfu = ngs.GridFunction(self._fes)  # solution 
        self._gfu.Set(0.0, definedon=self._mesh.Boundaries(".*"))
        
        # draw grid function in gui
        if self.show_gui:
            ngs.Draw(self._gfu)
        
        # create Hcurl space for flux calculation and estimate error
        self._space_flux = ngs.HDiv(self._mesh, order=2)
        self._gf_flux = ngs.GridFunction(self._space_flux, "flux")
        
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
    
    fempde1 = FemPde1(True)
    print(fempde1.pde_string)
    
    try:
        fempde1.exact(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    try:
        fempde1.approx(np.array([0.5,0.5]))
    except:
        print("Î error message above")
    
    fempde1.solve()
    
    print(fempde1.exact(np.array([0.5,0.5])))
    print(fempde1.approx(np.array([0.5,0.5])))
    print(fempde1.normL2())
    print("solving took {} sec".format(fempde1.exec_time))
    print("solving uses {} bytes".format(fempde1.mem_consumption))
    