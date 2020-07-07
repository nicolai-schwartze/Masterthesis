import numpy as np
import os
import zipfile
import xml.etree.ElementTree as ET
from shutil import copyfile
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__)) 
sys.path.append(importpath)

import bigjson

from scipy.stats import wilcoxon

# save an experiment object to the file path
def saveExpObject(obj, filename):
    """saves a CiPdeN object as a JSON file to the harddrive
    
    Parameters
    ----------
    obj : CiPdeN 
        object on which the .solve() method has been called
        
    filename : str
        includes path and filename

    Returns
    -------
    bool
        true if successful \n
        false if error occured

    """
    try:
        obj_dict = {"pde": str(type(obj)),\
                    "kernel_type": obj.kernel.kernel_type, \
                    "opt_algo": str(type(obj.opt_algo)),\
                    "exec_time": obj.exec_time,\
                    "mem_consumption": obj.mem_consumption,\
                    "normL2": obj.normL2(),\
                    "sol_kernel": obj.sol_kernel.tolist(),\
                    "pop_history": [o.tolist() for o in obj.pop_history],\
                    "fit_history": [o.tolist() for o in obj.fit_history],\
                    "f_history": obj.f_history,\
                    "cr_history": obj.cr_history}
        
        with open(filename, 'w') as json_file:
            json.dump(obj_dict, json_file)
            
        return True
    except Exception as e:
        print(str(e))
        return False
        
    


# load an experiment object from file
def loadExpObject(filename):
    """loads a CiPdeN object from a JSON file
    
    Parameters
    ----------
    filename : str
        includes path and filename

    Returns
    -------
    dict
        returns a dict if it worked, 
        else return None

    """
    try:
        with open(filename, 'r') as json_file:
            obj_dict = json.load(json_file)
            obj_dict["sol_kernel"] = np.array(obj_dict["sol_kernel"])
            obj_dict["pop_history"] = [np.array(o) for o in obj_dict["pop_history"]]
            obj_dict["fit_history"] = [np.array(o) for o in obj_dict["fit_history"]]
        return obj_dict
    except Exception as e:
        print(str(e))
        return None
        
    

# load an large experiment file and ingore the generation data
# expecially useful for large result files
def loadExpObjectFast(filename):
    """loads a CiPdeN object from a JSON file
        irnores generation data, expect the first and the last
    
    Parameters
    ----------
    filename : str
        includes path and filename

    Returns
    -------
    dict
        returns a dict if it worked, 
        else return None

    """
    try:
        with open(filename, 'rb') as f:
            result = bigjson.load(f)
            obj_dict = dict()
            obj_dict["pde"] = result["pde"]
            obj_dict["kernel_type"] = result["kernel_type"]
            obj_dict["opt_algo"] = result["opt_algo"]
            obj_dict["exec_time"] = result["exec_time"]
            obj_dict["mem_consumption"] = result["mem_consumption"]
            obj_dict["normL2"] = result["normL2"]
            obj_dict["sol_kernel"] = np.array(result["sol_kernel"].to_python())
        return obj_dict
    except Exception as e:
        print(str(e))
        return None
    

# draw gauss kernel to geogebra
def drawGSinKernel(parameter, ggb):
    """Draws a 2-dimensional GSin kernel function of the form
    
    .. math::
        approx(\mathbf{x})=\sum_{i=1}^{\\infty}\omega_{i}
        e^{\gamma_{i}||\mathbf{x}-\mathbf{c}_{i}||^2} sin(f ||\mathbf{x}-\mathbf{c}_{i}||^2 - p)
    
    to a GeoGebra file. 
    
    Parameters
    ----------
    parameter : numpy array
        must be of size (n,) where n % 6 = 0 \n
        [wi, yi, c0i, c1i, fi, pi] are parameters of the kernel
    ggb : str
        saves .ggb file to path

    Returns
    -------
    bool
        true if successful \n
        false if error occured

    Examples
    --------
    >>> parameter = np.array([3, 1.5, 1, 1, 0.5, 0])
    >>> ggb = "../testbed/pde0/pde0.ggb"
    >>> drawGSinKernel(parameter, ggb)
    ../testbed/pde0/pde0.ggb does not exist
    copied template to ../testbed/pde0/pde0.ggb
    False
    >>> drawGSinKernel(parameter, ggb)
    D:/Nicolai/Desktop/asdf.ggb exists
    renaming to zip
    extracting zip file
    removing zip
    parsing geogebra.xml
    modifying geogebra.xml
    saving modified xml
    creating zip archive
    renaming to ggb
    True
    
    """
    try:
        dimension = parameter.shape[0]
        numberOfKernels = dimension/6
        # candidate solution coding: 
        # [wi, yi, c0i, c1i] where i is the number of kernels which is adaptive
        kernels = parameter.reshape((int(numberOfKernels), 6))
        kernels = kernels.T
        w = kernels[0]
        y = kernels[1]
        c0 = kernels[2]
        c1 = kernels[3]
        f = kernels[4]
        p = kernels[5]
        
        # split filename from filepath
        ggb_path, ggb_name = os.path.split(ggb)
        if ggb_path == "":
            ggb_path = "."
            
        if not os.path.isfile(ggb):
            print(ggb + " does not exist")
            module_dir, _ = os.path.split(__file__)
            temp_path = module_dir + "/gsin_template.ggb"
            copyfile(temp_path, ggb)
            print("copied template to " + ggb)
            
        if os.path.isfile(ggb):
            print(ggb + " exists") 
            # convert to zip
            
            print("renaming to zip")
            zip_file = os.path.splitext(ggb)[0] + ".zip"
            os.rename(ggb, zip_file)
            
            # extract zip
            print("extracting zip file")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(ggb_path)
            zip_ref.close()
            
            # remove zip file
            print("removing zip")
            os.remove(zip_file)
            
            # open xml 
            print("parsing geogebra.xml")
            tree = ET.parse(ggb_path + "/geogebra.xml")
            root = tree.getroot()
            
            # write w, y, c0, c1
            print("modifying geogebra.xml")
            for element in root.findall('construction')[0].iter('expression'):
                if element.attrib['label'] == 'w_{approx}':
                    element.attrib['exp'] = nparrayToggblist(w)
                if element.attrib['label'] == 'y_{approx}':
                    element.attrib['exp'] = nparrayToggblist(y)
                if element.attrib['label'] == 'c0_{approx}':
                    element.attrib['exp'] = nparrayToggblist(c0)
                if element.attrib['label'] == 'c1_{approx}':
                    element.attrib['exp'] = nparrayToggblist(c1)
                if element.attrib['label'] == 'f_{approx}':
                    element.attrib['exp'] = nparrayToggblist(f)
                if element.attrib['label'] == 'p_{approx}':
                    element.attrib['exp'] = nparrayToggblist(p)
            # adapt length of sum
            for element in root.findall('construction')[0].iter('command'):
                for subelement in element:
                    for cmd in subelement.iter('input'):
                        if 'approx' in cmd.attrib['a0']:
                            cmd.attrib['a3'] = str(numberOfKernels)
            
            # write modified xml to file
            print("saving modified xml")
            tree.write(ggb_path + "/geogebra.xml")
            
            # zip files
            print("creating zip archive")
            zipArchiveName = os.path.splitext(ggb)[0] + ".zip"
            zipObj = zipfile.ZipFile(zipArchiveName, 'w')
            # Add multiple files to the zip
            zipObj.write(ggb_path + "/geogebra.xml", "/geogebra.xml")
            zipObj.write(ggb_path + "/geogebra_defaults2d.xml", "/geogebra_defaults2d.xml")
            zipObj.write(ggb_path + "/geogebra_defaults3d.xml", "/geogebra_defaults3d.xml")
            zipObj.write(ggb_path + "/geogebra_javascript.js", "/geogebra_javascript.js")
            zipObj.write(ggb_path + "/geogebra_thumbnail.png", "/geogebra_thumbnail.png")
            zipObj.close()
            
            # removing unzipped files
            os.remove(ggb_path + "/geogebra.xml")
            os.remove(ggb_path + "/geogebra_defaults2d.xml")
            os.remove(ggb_path + "/geogebra_defaults3d.xml")
            os.remove(ggb_path + "/geogebra_javascript.js")
            os.remove(ggb_path + "/geogebra_thumbnail.png")
            
            # convert to ggb
            print("renaming to ggb")
            zip_file = os.path.splitext(ggb)[0] + ".zip"
            os.rename(zip_file, ggb)
        
        return True
    
    except Exception as e:
        print("")
        print("error occured: ")
        print(e)
        return False
    




# draw gauss kernel to geogebra
def drawGaussKernel(parameter, ggb):
    """Draws a 2-dimensional Gaussian kernel function of the form
    
    .. math::
        approx(\mathbf{x})=\sum_{i=1}^{\\infty}\omega_{i}
        e^{\gamma_{i}||\mathbf{x}-\mathbf{c}_{i}||^2}
    
    to a GeoGebra file. 
    
    Parameters
    ----------
    parameter : numpy array
        must be of size (n,) where n % 4 = 0 \n
        [wi, yi, c0i, c1i] are parameters of the kernel
    ggb : str
        saves .ggb file to path

    Returns
    -------
    bool
        true if successful \n
        false if error occured

    Examples
    --------
    >>> parameter = np.array([3, 1.5, 1, 1])
    >>> ggb = "../testbed/pde0/pde0.ggb"
    >>> drawGaussKernel(parameter, ggb)
    ../testbed/pde0/pde0.ggb does not exist
    copied template to ../testbed/pde0/pde0.ggb
    False
    >>> drawGaussKernel(parameter, ggb)
    D:/Nicolai/Desktop/asdf.ggb exists
    renaming to zip
    extracting zip file
    removing zip
    parsing geogebra.xml
    modifying geogebra.xml
    saving modified xml
    creating zip archive
    renaming to ggb
    True
    
    """
    try:
        dimension = parameter.shape[0]
        numberOfKernels = dimension/4
        # candidate solution coding: 
        # [wi, yi, c0i, c1i] where i is the number of kernels which is adaptive
        kernels = parameter.reshape((int(numberOfKernels), 4))
        kernels = kernels.T
        w = kernels[0]
        y = kernels[1]
        c0 = kernels[2]
        c1 = kernels[3]
        
        # split filename from filepath
        ggb_path, ggb_name = os.path.split(ggb)
        if ggb_path == "":
            ggb_path = "."
            
        if not os.path.isfile(ggb):
            print(ggb + " does not exist")
            module_dir, _ = os.path.split(__file__)
            temp_path = module_dir + "/gauss_template.ggb"
            copyfile(temp_path, ggb)
            print("copied template to " + ggb)
            
        if os.path.isfile(ggb):
            print(ggb + " exists") 
            # convert to zip
            
            print("renaming to zip")
            zip_file = os.path.splitext(ggb)[0] + ".zip"
            os.rename(ggb, zip_file)
            
            # extract zip
            print("extracting zip file")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(ggb_path)
            zip_ref.close()
            
            # remove zip file
            print("removing zip")
            os.remove(zip_file)
            
            # open xml 
            print("parsing geogebra.xml")
            tree = ET.parse(ggb_path + "/geogebra.xml")
            root = tree.getroot()
            
            # write w, y, c0, c1
            print("modifying geogebra.xml")
            for element in root.findall('construction')[0].iter('expression'):
                if element.attrib['label'] == 'w_{approx}':
                    element.attrib['exp'] = nparrayToggblist(w)
                if element.attrib['label'] == 'y_{approx}':
                    element.attrib['exp'] = nparrayToggblist(y)
                if element.attrib['label'] == 'c0_{approx}':
                    element.attrib['exp'] = nparrayToggblist(c0)
                if element.attrib['label'] == 'c1_{approx}':
                    element.attrib['exp'] = nparrayToggblist(c1)
            # adapt length of sum
            for element in root.findall('construction')[0].iter('command'):
                for subelement in element:
                    for cmd in subelement.iter('input'):
                        if 'approx' in cmd.attrib['a0']:
                            cmd.attrib['a3'] = str(numberOfKernels)
            
            # write modified xml to file
            print("saving modified xml")
            tree.write(ggb_path + "/geogebra.xml")
            
            # zip files
            print("creating zip archive")
            zipArchiveName = os.path.splitext(ggb)[0] + ".zip"
            zipObj = zipfile.ZipFile(zipArchiveName, 'w')
            # Add multiple files to the zip
            zipObj.write(ggb_path + "/geogebra.xml", "/geogebra.xml")
            zipObj.write(ggb_path + "/geogebra_defaults2d.xml", "/geogebra_defaults2d.xml")
            zipObj.write(ggb_path + "/geogebra_defaults3d.xml", "/geogebra_defaults3d.xml")
            zipObj.write(ggb_path + "/geogebra_javascript.js", "/geogebra_javascript.js")
            zipObj.write(ggb_path + "/geogebra_thumbnail.png", "/geogebra_thumbnail.png")
            zipObj.close()
            
            # removing unzipped files
            os.remove(ggb_path + "/geogebra.xml")
            os.remove(ggb_path + "/geogebra_defaults2d.xml")
            os.remove(ggb_path + "/geogebra_defaults3d.xml")
            os.remove(ggb_path + "/geogebra_javascript.js")
            os.remove(ggb_path + "/geogebra_thumbnail.png")
            
            # convert to ggb
            print("renaming to ggb")
            zip_file = os.path.splitext(ggb)[0] + ".zip"
            os.rename(zip_file, ggb)
        
        return True
    
    except Exception as e:
        print("")
        print("error occured: ")
        print(e)
        return False
    
    
pde_solution = {"<class 'CiPde0A.CiPde0A'>" : lambda x : 2*np.e**(-1.5*(x[0]**2 + x[1]**2)) + np.e**(-3*((x[0] + 1)**2 + (x[1] + 1)**2)) + np.e**(-3*((x[0] - 1)**2 + (x[1] + 1)**2)) + np.e**(-3*((x[0] + 1)**2 + (x[1] - 1)**2)) + np.e**(-3*((x[0] - 1)**2 + (x[1] - 1)**2)),
                "<class 'CiPde0B.CiPde0B'>" : lambda x : np.exp(-2  * ((x[0])**2 + (x[1])**2))*np.sin(2  * ((x[0])**2 + (x[1])**2)) + np.exp(-1  * ((x[0])**2 + (x[1])**2))*np.sin(1  * ((x[0])**2 + (x[1])**2)) + np.exp(-0.1* ((x[0])**2 + (x[1])**2))*np.sin(0.1* ((x[0])**2 + (x[1])**2)),
                "<class 'CiPde1.CiPde1'>" : lambda x : (2**(4*10))*(x[0]**10)*((1-x[0])**10)*(x[1]**10)*((1-x[1])**10),
                "<class 'CiPde2.CiPde2'>" : lambda x : (x[0] + x[1]**3)*np.e**(-x[0]),
                "<class 'CiPde3.CiPde3'>" : lambda x : x[0]**2 + x[1]**2 + x[0] + x[1] + 1,
                "<class 'CiPde4.CiPde4'>" : lambda x : np.sin(np.pi * x[0])*np.sin(np.pi * x[1]),
                "<class 'CiPde5.CiPde5'>" : lambda x : np.arctan(20*(np.sqrt((x[0] - 0.05)**2 + (x[1] - 0.05)**2) -0.7)),
                "<class 'CiPde6.CiPde6'>" : lambda x : np.e**(-1000*((x[0]-0.5)**2 + (x[1]-0.5)**2)),
                "<class 'CiPde7.CiPde7'>" : lambda x : x[0]**0.6 + x[1]*0,
                "<class 'CiPde8.CiPde8'>" : lambda x : np.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2),
                "<class 'CiPde9.CiPde9'>" : lambda x : np.arctan(20*((x[0] + x[1])/(2**(1/2)) -0.8))*x[0]*(1-x[0])*x[1]*(1-x[1]) }
    
    
    
def calcRMSE(solve_dict):
    """calculates the RMSE of a loaded dictionary
       it is fully compativle with the testbed, other PDEs might not be 
       recognised
       
       .. math:: 
               RMSE^2 \cdot (nc + nb) = \sum_{x_c}^{nc} (u_{apx}(x_c) - u_{ext}(x_c))^2 + \sum_{x_b}^{nb} (u_{apx}(x_b) - u_{ext}(x_b))^2 
       
       Parameters
       ----------
       solve_dict: dict
                   loaded dictionary from file either with loadExpObj or 
                   loadExpObjFast
                   
       Returns
       -------
       float
            returns the RSME value
    """
    if solve_dict["pde"] == "<class 'CiPde0A.CiPde0A'>" or solve_dict["pde"] == "<class 'CiPde0B.CiPde0B'>":
        nc = []
        omega = np.arange(-1.6, 2.0, 0.4)
        for x0 in omega:
            for x1 in omega:
                nc.append((x0, x1))
            
        # boundary points for 0A and 0B
        nb = []
        nby = np.hstack((-2*np.ones(10), np.arange(-2.0, 2.0, 0.4), 2*np.ones(10), np.arange(2.0, -2.0, -0.4)))
        nbx = np.hstack((np.arange(-2.0, 2.0, 0.4), 2*np.ones(10), np.arange(2.0, -2.0, -0.4), -2*np.ones(10)))
        for i in range(len(nby)):
            nb.append((nbx[i], nby[i]))
    
    else:
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
        
    if solve_dict["kernel_type"] == "Gauss Kernel: sum_{i}^{N}(w_i*e^(-y_i*((x_0 - c_0_i)^2 + (x_1 - c_1_i)^2)))":
        importpath = os.path.dirname(os.path.realpath(__file__)) + "/../kernels/"
        sys.path.append(importpath) 
        import KernelGauss as gak
        kernel = gak.KernelGauss()

        
    else:
        importpath = os.path.dirname(os.path.realpath(__file__)) + "/../kernels/"
        sys.path.append(importpath) 
        import KernelGSin as gsk
        kernel = gsk.KernelGSin()
    
    # partial sum for collocation points
    part_sum_c = 0.0
    for c in nc:
        part_sum_c += (kernel.solution(solve_dict["sol_kernel"], c) - pde_solution[solve_dict["pde"]](c))**2
        
    # partial sum for boundary boints
    part_sum_b = 0.0
    for b in nb:
        part_sum_b += (kernel.solution(solve_dict["sol_kernel"], b) - pde_solution[solve_dict["pde"]](b))**2
        
    return np.sqrt((part_sum_c + part_sum_b)/(len(nb) + len(nc)))
    
    



def plotABSError3D(kernel, parameter, pdeName, lD, uD, name=None):
    """Draws a 2-dimensional absolut error plot 
       on the squared domain from lD to uD
    
    Parameters
    ----------
    kernel : IKernelBase
        object that implements the IKernelBase interface
    parameter : numpy array
        parameter associated with the solution
    pdeName : string
        name of the pde class used as key to the 
        analytic solution
    lD: float
        lower boundary of the squared domain
    uD: float
        upper boundary of the squared domain
    name: string
          saves the figure as this file

    Examples
    --------
    >>> import KernelGauss as gk
    >>> gkernel = gk.KernelGauss()
    >>> param = np.array([[1,4,0.5,0.5]])
    >>> plotApprox3D(gkernel, param, 0.0, 1.0)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(lD, uD+0.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([np.abs(kernel.solution(parameter, np.array([x,y])) \
                    - pde_solution[pdeName](np.array([x,y])))
                    for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("abs(error)")
    plt.tight_layout()
    if type(name) == type(""):
        plt.savefig(name,bbox_inches='tight')
        
    plt.show()
    return None





    
def plotApprox3D(kernel, parameter, lD, uD, name=None):
    """Draws a 2-dimensional kernel summation function of the form
       in between the squared domain from lD to uD
    
    Parameters
    ----------
    kernel : IKernelBase
        object that implements the IKernelBase interface
    parameter : numpy array
        parameter associated with the solution
    lD: float
        lower boundary of the squared domain
    uD: float
        upper boundary of the squared domain
    name: string
          saves the figure as this file

    Examples
    --------
    >>> import KernelGauss as gk
    >>> gkernel = gk.KernelGauss()
    >>> param = np.array([[1,4,0.5,0.5]])
    >>> plotApprox3D(gkernel, param, 0.0, 1.0)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(lD, uD+0.01, 0.01)
    X, Y = np.meshgrid(x, y)
    
    zs0 = np.array([kernel.solution(parameter, np.array([x,y]))\
                    for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs0.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
    
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("f(X0,X1)")
    plt.tight_layout()
    if type(name) == type(""):
        plt.savefig(name,bbox_inches='tight')
        
    plt.show()
    return None
    

def statsWilcoxon(a, b, alpha=0.05):
    """calculates the wilcoxon test between the two lists of data
       at a significance level of alpha = 0.05
       and returns a string describing if mean(a) < mean(b) and median(a) < median(b)
    
    Parameters
    ----------
    a : list
        list of values ought to be smaller
    b : list
        list of values ought to be larger
    alpha: float
        significance level
    
    Returns
    -------
    string
        unsig./sig. better/worse
    
    """
    stat, p = wilcoxon(a, b)
    result = ""
    
    if p > alpha:
        result = "unsig. "
    else:
        result = "sig. "
        
    if (np.mean(a) < np.mean(b)) and (np.median(a) < np.median(b)):
        result += "better"
    elif (np.mean(a) < np.mean(b)) != (np.median(a) < np.median(b)):
        result += "undecided"
    else:
        result += "worse"
        
    return result


def plotFEDynamic(FEDynamic, name=None):
    """prints the FE Dynamic to a semilogy plot 
       for an adaptive popuation
       necessary to compensate the different dimension of the array
       
    Parameters
    ----------
    FEDynamic: list
               list where each element is the FE of the population at every gen
    name: string
          saves the figure as this file
    """
    goaldim = 0
    for i in FEDynamic:
        if i.shape[0] > goaldim:
            goaldim = i.shape[0]
    plotFEList = []
    for i in FEDynamic:
        currentdim = i.shape[0]
        plotFEList.append(np.lib.pad(i, (0,goaldim-currentdim), 'constant', constant_values=(0)))
        
    plt.tight_layout()
    plt.semilogy(plotFEList)
    if type(name) == type(""):
        plt.savefig(name,bbox_inches='tight')
        
    plt.show()
    return None
    
    
# --------------------------------------------------------------------------- #
# --------------------------- internal functions ---------------------------- #
# --------------------------------------------------------------------------- #
    
    
    
def nparrayToggblist(array):
    tempList = array.tolist()
    tempString = str(tempList)
    tempString = tempString.replace('[', '{')
    return tempString.replace(']','}')
    
    
    
if __name__ == "__main__":
    
    print("starting test")
    
    # ----------------------------------------------------------------------- #
    
    print("test draw gauss kernel to ggb")
    
    # test drawGaussKernel()
    parameter = np.array([1, 1, 1, 1, 1, 1, -1, -1])
    drawGaussKernel(parameter, "geogebra_test_gauss.ggb")
    
    print("test draw gsin kernel to ggb")
    
    # test drawGSinKernel()
    parameter = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1])
    drawGSinKernel(parameter, "geogebra_test_gsin.ggb")
    
    # ----------------------------------------------------------------------- #
    
    print("test save function")
    
    # imports needed for test
    import sys
    sys.path.append("../testbed/pde2")
    import CiPde2 as pde2
    sys.path.append("../opt_algo")
    import OptAlgoMemeticJADE as oaMemJade
    sys.path.append("../kernels")
    import KernelGauss as gk
    
    # initialisation 
    initialPop = 1*np.random.rand(40,20)
    max_iter = 1*10**2
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
    
    # creating object 
    cipde2 = pde2.CiPde2(mJade, gkernel, nb, nc)
    cipde2.solve()
    
    saveExpObject(cipde2, "./save_test_large.json")
    
    load_dict = loadExpObject("./save_test_large.json")
    
    assert cipde2.kernel.kernel_type == load_dict["kernel_type"]
    assert cipde2.exec_time == load_dict["exec_time"]
    assert cipde2.mem_consumption == load_dict["mem_consumption"]
    assert cipde2.normL2() == load_dict["normL2"]
    assert np.allclose(cipde2.sol_kernel, load_dict["sol_kernel"])
    assert np.allclose(cipde2.pop_history, load_dict["pop_history"])
    assert np.allclose(cipde2.fit_history, load_dict["fit_history"])
    assert np.allclose(cipde2.f_history, load_dict["f_history"])
    assert np.allclose(cipde2.cr_history, load_dict["cr_history"])
    
    plt.plot(cipde2.fit_history)
    plt.show()
    plt.plot(load_dict["fit_history"])
    plt.show()
    
    load_dict = loadExpObjectFast("./save_test_large.json")
    
    assert cipde2.kernel.kernel_type == load_dict["kernel_type"]
    assert cipde2.exec_time == load_dict["exec_time"]
    assert cipde2.mem_consumption == load_dict["mem_consumption"]
    assert cipde2.normL2() == load_dict["normL2"]
    assert np.allclose(cipde2.sol_kernel, load_dict["sol_kernel"])
    
    
    #-------------------------------------------------------------------------#
    
    plotApprox3D(gkernel, cipde2.sol_kernel, 0.0, 1.0)
    
    print("RMSE = " + str(calcRMSE(load_dict)))
    
    print("finished test")
    
    