'''
This module contains the following convenience functions:
    make_file_path
'''
import numpy as np
import astropy.units as u

def make_file_path(directory, array_kwargs, extra_string=None, ext='.dat'):
    ## Makes file path given string and array kwargs.
    s = '_'
    string_kwargs = [str(int(i)) for i in array_kwargs]
    string_kwargs = np.array(string_kwargs, dtype='U25')
    if (extra_string !=None) and (len(extra_string)>25):
        raise TypeError('Extra string must have less than 25 characters')
    if extra_string !=None:
        string_kwargs = np.insert(string_kwargs, 0, extra_string)
    kwpath = s.join(string_kwargs)
    return directory+kwpath+ext
