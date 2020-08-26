# Masterthesis

Computational Intelligence Method for Solving Partial Differential Equations

For earning a Master of Science in Mechatronics from Vorarlberg University of Applied Science

## Abstract

This master-thesis describes and tests a differential equation solver that is based on the heuristic black-box optimisation algorithm JADE. 
The solver works by reformulating the residual of the differential equation into a fitness function. 
The underlying solution function of the differential equation is approximated by a finite sum of Radial Basis Functions. 
The performance of the solver is evaluated on a testbed that consists of 11 two-dimensional Poisson equations. 
The solving-time, the memory usage and the accuracy are compared to the open-source Finite Element Solver NGSolve as well as similar work in the current literature. 
To optimise the solving time and speed up the experiments, JADE is parallelised. 
This work introduces two new strategies: First, the number of kernels is adjusted to the differential equation during the solving process. 
Furthermore, a new kernel is formed and tested. These concepts are checked for significant improvements. The structure of the fitness function is investigated. 
A fundamental symmetry is observed that could be used to develop better suited optimisation algorithms. 






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
