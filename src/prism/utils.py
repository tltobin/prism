# Last updated 4/1/2021
#     Updates to support for user-generated default parameters for faraday object from config file.
#         - Creation of _default_ class to differentiate when parameters with useable defaults are specified on call.
#         - Creation of helper function, string_to_list, to convert a string formatted like '[val1, val2, val3]' into a
#           list of some specified data type. If the data type is a float, numpy longdouble, or integer, supports
#           expressions as the individual list entries.
#         - Creation of helper function, string_to_bool, to convert strings 'true' or 'false' to the associated 
#           booleans. Not case sensitive.

from glob import glob
import numpy as np


# Small function for converting fits files to text
def txt_to_fits( f ):
    
    # Gets list of all dzero output text files in outpath
    outfiles = glob( '{0}FaradayOut_beta*dzero.txt'.format( f.outpath ) )
    
    # Shortens file names by removing outpath
    outfiles = [ fname.replace( f.outpath, '' ) for fname in outfiles ]
    
    # Extracts beta from each file name
    i1 = len( 'FaradayOut_beta' )
    i2 = len( '_dzero.txt' )
    betas = [ fname[i1:-i2] for fname in outfiles ]
    betas = [ float(beta) for beta in betas ]
    betas.sort()
    
    # Iterates through betas
    for beta in betas:
        
        # Reads in text files for beta
        f.deltas = f.readin( beta, ext='txt' )
        
        # Re-writes contents as fits file
        f.write_deltas( beta, f.deltas, ext='fits' )
   


# Simple class to denote default parameter values provided by prism
class _default_:
    def __init__(self, value):
        self.value = value


# Small function to take a string formatted as a list '[1,2,3]' and convert it to a list of a given data type
def string_to_list( mystring, astype = float ):
    
    # Assumes first character is [ and last character is ] (should be checked before handing to this function)
    in_brackets = mystring[1:-1]
    
    # Splits into list by commas
    raw_list = in_brackets.split(',')
    
    # If converting to float or int, evaluates string before converting
    if astype in [int, float, np.longdouble ]:
        out_list = [ astype( eval( x ) ) for x in raw_list ]
    # Otherwise, just convert the type
    else:
        out_list = [ astype( x ) for x in raw_list ]
    
    # Returns the list 
    return out_list


# Small function to turn string 'true' or 'false' into a bool
def string_to_bool( mystring ):
    if mystring.lower() == 'true':
        outval = True
    elif mystring.lower() == 'false':
        outval = False
    return outval