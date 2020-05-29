import numpy as np
import os
import zipfile
import xml.etree.ElementTree as ET
from shutil import copyfile
import matplotlib.pyplot as plt
import json


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
    
    print("test draw gauss kernel to ggb")
    # ------------------------------------------------------------------------#
    
    # test drawGaussKernel()
    parameter = np.array([1, 1, 1, 1, 1, 1, -1, -1])
    drawGaussKernel(parameter, "geogebra_test.ggb")
    
    print("test save function")
    # ------------------------------------------------------------------------#
    
    # imports needed for test
    import sys
    sys.path.append("../testbed/pde1")
    import CiPde1 as pde1
    sys.path.append("../opt_algo")
    import OptAlgoMemeticJADE as oaMemJade
    sys.path.append("../kernels")
    import KernelGauss as gk
    
    # initialisation 
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
    
    # creating object 
    cipde1 = pde1.CiPde1(mJade, gkernel, nb, nc)
    cipde1.solve()
    
    saveExpObject(cipde1, "./save_test_large.json")
    
    load_dict = loadExpObject("./save_test_large.json")
    
    assert cipde1.kernel.kernel_type == load_dict["kernel_type"]
    assert cipde1.exec_time == load_dict["exec_time"]
    assert cipde1.mem_consumption == load_dict["mem_consumption"]
    assert np.allclose(cipde1.sol_kernel, load_dict["sol_kernel"])
    assert np.allclose(cipde1.pop_history, load_dict["pop_history"])
    assert np.allclose(cipde1.fit_history, load_dict["fit_history"])
    assert np.allclose(cipde1.f_history, load_dict["f_history"])
    assert np.allclose(cipde1.cr_history, load_dict["cr_history"])
    
    plt.plot(cipde1.fit_history)
    plt.show()
    plt.plot(load_dict["fit_history"])
    plt.show()
    
    print("finished test")
    