import numpy as np
import os
import zipfile
import xml.etree.ElementTree as ET
from shutil import copyfile

def draw_gauss_kernel(parameter, ggb_file):
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
    ggb_file : str
        saves .ggb file to path

    Returns
    -------
    bool
        true if successful \n
        false if error occured

    Examples
    --------
    >>> parameter = np.array([3, 1.5, 1, 1])
    >>> ggb_file = "../testbed/pde0/pde0.ggb"
    >>> draw_gauss_kernel(parameter, ggb_file)
    ../testbed/pde0/pde0.ggb does not exist
    copied template to ../testbed/pde0/pde0.ggb
    False
    >>> draw_gauss_kernel(parameter, ggb_file)
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
        ggb_file_path, ggb_file_name = os.path.split(ggb_file)
        
        if os.path.isfile(ggb_file):
            print(ggb_file + " exists") 
            # convert to zip
            
            print("renaming to zip")
            zip_file = os.path.splitext(ggb_file)[0] + ".zip"
            os.rename(ggb_file, zip_file)
            
            # extract zip
            print("extracting zip file")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(ggb_file_path)
            zip_ref.close()
            
            # remove zip file
            print("removing zip")
            os.remove(zip_file)
            
            # open xml 
            print("parsing geogebra.xml")
            tree = ET.parse(ggb_file_path + "/geogebra.xml")
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
            tree.write(ggb_file_path + "/geogebra.xml")
            
            # zip files
            print("creating zip archive")
            zipArchiveName = os.path.splitext(ggb_file)[0] + ".zip"
            zipObj = zipfile.ZipFile(zipArchiveName, 'w')
            # Add multiple files to the zip
            zipObj.write(ggb_file_path + "/geogebra.xml", "/geogebra.xml")
            zipObj.write(ggb_file_path + "/geogebra_defaults2d.xml", "/geogebra_defaults2d.xml")
            zipObj.write(ggb_file_path + "/geogebra_defaults3d.xml", "/geogebra_defaults3d.xml")
            zipObj.write(ggb_file_path + "/geogebra_javascript.js", "/geogebra_javascript.js")
            zipObj.write(ggb_file_path + "/geogebra_thumbnail.png", "/geogebra_thumbnail.png")
            zipObj.close()
            
            # removing unzipped files
            os.remove(ggb_file_path + "/geogebra.xml")
            os.remove(ggb_file_path + "/geogebra_defaults2d.xml")
            os.remove(ggb_file_path + "/geogebra_defaults3d.xml")
            os.remove(ggb_file_path + "/geogebra_javascript.js")
            os.remove(ggb_file_path + "/geogebra_thumbnail.png")
            
            # convert to ggb
            print("renaming to ggb")
            zip_file = os.path.splitext(ggb_file)[0] + ".zip"
            os.rename(zip_file, ggb_file)
            
        else:
            print(ggb_file + " does not exist")
            module_dir, _ = os.path.split(__file__)
            temp_path = module_dir + "/template.ggb"
            copyfile(temp_path, ggb_file)
            print("copied template to " + ggb_file)
        
        return True
    
    except:
        print("")
        print("error occured: could not save file")
        return False

def nparrayToggblist(array):
    tempList = array.tolist()
    tempString = str(tempList)
    tempString = tempString.replace('[', '{')
    return tempString.replace(']','}')
    


if __name__ == "__main__":
    
    parameter = np.array([1, 1, 1, 1, 1, 1, -1, -1])
    draw_gauss_kernel(parameter, "D:/Nicolai/Desktop/asdf.ggb")