# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:57:32 2020

@author: Nicolai
"""

from ITestbenchBase import ITestbenchBase
import numpy as np

# import from ngsolve
import ngsolve as ngs
from netgen.geom2d import unit_square

class FemPdeBase(ITestbenchBase):
    """
    Abstract class that provides implements the the ITestbenchBase interface. 
    It further provides functionality that is used by the FEM solver. 
    
    Attributes
    ----------
    show_gui: bool
        display the graphical user interface or not - has an impact on the 
        execution time and the memory consumption
    
    Methods
    -------
    pde_string()
        getter for returning the _pde_string that holds a short description of the problem
    exec_time()
        getter for returning the execution time taken for solving the probem
    mem_consumption()
        getter for returning the memory consumption of the solver
    _solveStep()
        gradually refines the FEM mesh and calculates a new solution
    _estimateError
        estimates the error of an element and marks the worst 25% elements for refinement
    normL2()
        returns the distance between the exact and the approximate solution
    solve()
        not implemented, must be overridden by the child class that is specific to a pde
    
    """
    
    def __init__(self, show_gui):
        super().__init__()
        # init public atributes 
        self.show_gui = show_gui
        
        # init protected atributes
        self._ngs_ex = None
        self._fes = None
        self._gfu = None
        self._a = None
        self._f = None
        self._space_flux = None
        self._gf_flux = None
        self._mesh = None
        
    @property
    def pde_string(self):
        return self._pde_string
    
    @property
    def exec_time(self):
        return self._exec_time
    
    @property
    def mem_consumption(self):
        return self._mem_consumption
    
    def exact(self, x): 
        if isinstance(x, np.ndarray) and \
        isinstance(self._ngs_ex, ngs.fem.CoefficientFunction) and \
        isinstance(self._mesh, ngs.comp.Mesh):
            point = self._mesh(x[0], x[1])
            return self._ngs_ex(point)
        else: 
            raise TypeError("""
           x -> numpy.ndarray with shape (2,) 
           _mesh -> ngs.comp.Mesh 
           _ngs_ex -> ngs.fem.CoefficientFunction 
           -> try to call solve() first""")
            return None
    
    def approx(self, x): 
        if isinstance(x, np.ndarray) and \
        isinstance(self._gfu, ngs.fem.CoefficientFunction) and \
        isinstance(self._mesh, ngs.comp.Mesh):
            point = self._mesh(x[0], x[1])
            return self._gfu(point)
        else: 
            raise TypeError("""
           x -> numpy.ndarray with shape (2,) 
           _mesh -> ngs.comp.Mesh 
           _gfu -> ngs.fem.CoefficientFunction 
           -> try to call solve() first""")
            return None
    
    def normL2(self): 
        try: 
            return ngs.sqrt(ngs.Integrate((self._gfu-self._ngs_ex)*(self._gfu-self._ngs_ex), self._mesh))
        except TypeError:
            print("FEM approximate solution not ready - try to call solve() first")
    
    # protected method that calculates one step of the grid refinement
    def _solveStep(self):
        # update the finite element space
        self._fes.Update()
        # update the grid function (= solution)
        self._gfu.Update()
        if self.show_gui:
            ngs.Draw(self._gfu)
        # assamble the bilinear form
        self._a.Assemble()
        # assamble the linear functional
        self._f.Assemble()
        # solve the boundary value problem iteratively
        # if the solution does not converge, this is the first place to look
        ngs.solvers.BVP(bf=self._a, lf=self._f, gf=self._gfu, pre=self._c)
        if self.show_gui:
            ngs.Redraw(blocking=True)
            
    
    def _estimateError(self):
        # update the flux space (Hcurl)
        self._space_flux.Update()
        # update the gradient in the Hcurls space
        self._gf_flux.Update()
        # compute the flux
        flux = ngs.grad(self._gfu)
        # interpolate the gradient in the Hcurls space
        self._gf_flux.Set(flux) 
            
        # compute estimator:
        err = (flux-self._gf_flux)*(flux-self._gf_flux)
        eta2 = ngs.Integrate(err, self._mesh, ngs.VOL, element_wise=True)
        # set max error for calculating the worst 25% 
        maxerr = max(eta2)
            
        # mark for refinement:
        for el in self._mesh.Elements():
            # mark worst 25% elements to refinement
            self._mesh.SetRefinementFlag(el, eta2[el.nr] > 0.25*maxerr)
    
    # actually not implemented, implementation is done in inheriting class
    def solve(self): 
        pass


if __name__ == "__main__":
    
    fempdeb = FemPdeBase(True)
    
    print("1: " + str(fempdeb.exact))
    
    fempdeb.solve()
    
    print("2: " + str(fempdeb.exact))
    
    
    
    
    