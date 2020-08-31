# Masterthesis

Computational Intelligence Methods for Solving Partial Differential Equations

For earning a Master of Science in Mechatronics from Vorarlberg University of Applied Science

## Abstract

This master thesis investigates a Computational Intelligence-based method for solving PDEs. 
The proposed strategy formulates the residual of a PDE as a fitness function. The solution is approximated by a finite sum of Gauss kernels. 
An appropriate optimisation technique, in this case JADE, is deployed that searches for the best fitting parameters for these kernels. 
This field is fairly young, a comprehensive literature research reveals several past papers that investigate similar techniques. 
To evaluate the performance of the solver, a comprehensive testbed is defined. It consists of 11 different Poisson equations. 
The solving time, the memory consumption and the approximation quality are compared to the state of the art open-source Finite Element solver NGSolve. 
The first experiment tests a serial JADE. The results are not as good as comparable work in the literature. 
Further, a strange behaviour is observed, where the fitness and the quality do not match. 
The second experiment implements a parallel JADE, which allows to make use of parallel hardware. 
This significantly speeds up the solving time. The third experiment implements a parallel JADE with adaptive kernels. 
It starts with one kernel and introduce more kernels along the solving process. 
A significant improvement is observed on one PDE, that is purposely built to be solvable. 
On all other testbed PDEs the quality-difference is not conclusive. The last experiment investigates the discrepancy between the fitness and the quality. 
Therefore, a new kernel is defined. This kernel inherits all features of the Gauss kernel and extends it with a sine function. 
As a result, the observed inconsistency between fitness and quality is mitigated. The thesis closes with a proposal for further investigations. 
The concepts here should be reconsidered by using better performing optimisation algorithms from the literature, like CMA-ES. 
Beyond that, an adaptive scheme for the collocation points could be tested. Finally, the fitness function should be further examined.






## NGSolve GUI Prerequisites

for a more detailed description look at 
https://ngsolve.org/docu/latest/install/usejupyter.html


* python version 3.7 with virtual environment
```
    > conda create -n ngsolve anaconda python=3.7 
```
* change virtual envorinment to ngsolve:
```
    > conda activate ngsolve 
```
* ipykernel version 4.10 in newly created virtual environment
```
    > conda install ipykernel==4.10.0 
```    
* install ngsolve as recommended by https://ngsolve.org/downloads
```
    > conda config --add channels conda-forge 
    
    > conda config --add channels ngsolve 
    
    > conda install ngsolve 
```    
* jupyter notebook extensions for ngsolve
```
    > jupyter nbextension install --user --py ngsolve
```    
* run ngsolve gui-examples under ./code/examples/ngsolve_gui.ipynb
