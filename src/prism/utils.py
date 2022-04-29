# Last updated 4/29/2021
# - 
# - New functions, round_up_to and round_down_to, which function like ceil and floor, but allow a non-integer precision
#   to be specified. The precision functions the same as in Python's round function.
# - New function, gen_contour_levels, will create minimal contour levels between some minimum and maximum value.
# - New function, round_to_sig_fig, will round a float to a certain number of significant figures instead of decimal
#   places. Includes option to specify whether value should be rounded only up or down.


from glob import glob
import numpy as np
from math import log10, floor, ceil


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


def round_up_to( x, precision ):
    """
    Like ceil function, but allows user to specify the precision to which the value
    is rounded. Precision functions the same way as the round function, with positive
    values indicating places to the right of the decimal, and negative values indicating
    places to the left of the decimal.
    """
    
    # Scales so that ceil function to requested precision will cut off value as integer
    xscaled = x * 10.**precision
    
    # Rounds up using ceil
    xscaled_rnd = float(ceil( xscaled ))
    
    # Rescales back to original
    x_rnd = float( xscaled_rnd )* 10.**-precision
    
    # May have acquired a floating point error, so does one final round
    round( x_rnd, precision )
    
    # Returns rounded value
    return x_rnd

def round_down_to( x, precision ):
    """
    Like floor function, but allows user to specify the precision to which the value
    is rounded. Precision functions the same way as the round function, with positive
    values indicating places to the right of the decimal, and negative values indicating
    places to the left of the decimal.
    """
    
    # Scales so that floor function to requested precision will cut off value as integer
    xscaled = x * 10.**precision
    
    # Rounds up using floor
    xscaled_rnd = float(floor( xscaled ))
    
    # Rescales back to original
    x_rnd = float( xscaled_rnd )* 10.**-precision
    
    # May have acquired a floating point error, so does one final round
    round( x_rnd, precision )
    
    # Returns rounded value
    return x_rnd
    
    
    



def gen_contour_levels( value_min, value_max, min_contours = 2 ):
    
    levels = np.array([])
    
    # Handling of determining scale of each value depends on if either is 0
    # If both are 0, just raises an error
    if value_min == 0.0 and value_max == 0.0:
        raise ValueError( 'Max and min values both zero. No contours generated.' )
    
    # If neither are zero, calculation straightforward
    elif value_min != 0.0 and value_max != 0.0:
    
        # First, finds exponential scale of each value
        exscale_min = int(floor(log10(abs( value_min ))))
        exscale_max = int(floor(log10(abs( value_max ))))
    
        # Finds how big the difference between min and max is relative to their
        #    individual scales
        exscale_diff = int(floor(log10(abs( value_max - value_min ))))
    
    # If the min is zero but the max isn't
    elif value_min == 0.0:
        
        # Calculates exponential scale of max
        exscale_max = int(floor(log10(abs( value_max ))))
        
        # scale difference is just the scale of the max
        exscale_diff = exscale_max
    
    # If the max is zero but the min isn't
    else:
        
        # Calculates exponential scale of min
        exscale_min = int(floor(log10(abs( value_min ))))
        
        # scale difference is just the scale of the min
        exscale_diff = exscale_min
        
        
    # The precision of the level steps is determined by the scale of the difference
    level_step = 10.**exscale_diff
    
    # Contour levels will go from ceiling of min to floor of max
    if value_min != 0.0:
        level_min = round_up_to( value_min, -exscale_diff )
    else:
        level_min = 0.0
    if value_max != 0.0:
        level_max = round_down_to( value_max, -exscale_diff )
    else:
        level_max = 0.0
    
    # Generates levels
    levels = np.arange( level_min, level_max + level_step, level_step )
    
    # If there aren't the minimum number of contour levels, tries again with one lower precision
    while levels.size < min_contours:
        
        # Adjusts the precision and step size
        exscale_diff -= 1
        level_step = 10.**exscale_diff
    
        # Contour levels will go from ceiling of min to floor of max
        if value_min != 0.0:
            level_min = round_up_to( value_min, -exscale_diff )
        if value_max != 0.0:
            level_max = round_down_to( value_max, -exscale_diff )
    
        # Generates levels
        levels = np.arange( level_min, level_max + level_step, level_step )
    
    # Converts levels to a list and makes sure levels rounded to desired precision
    levels = [ round( lev, -exscale_diff ) for lev in levels ]
    
    # Returns the generated levels
    return levels
        
    


def round_to_sig_fig( x, sigfigs, direction = None ):
    """
    Rounds a float (x) to a certain number of significant figures.
    
    Rounding direction can be specified.
    
    Required Parameters:
        
        x               Float
                            The number to be rounded.
        
        sigfigs         Integer
                            The number of significant figures that the value should be
                            rounded to.
    
    Optional Parameters:
        
        direction       None or String: 'up' or 'down'
                            [ Default = None ]
                            If None, performs normal rounding to nearest value.
                            If 'up', only round up, and if 'down', only rounds down.
                            Not case sensitive.
    
    Returns:
        
        outval          Float
                            The input value, x, rounded to the requested number of 
                            significant figures (in the requested direction, if any).
    """
    
    # Determines rounding precision from significant figures
    roundto = -int(floor(log10(abs(x)))) + int(sigfigs-1)
    
    # If no direction specified, uses basic rounding function
    if direction is None:
        outval = round( x, roundto )
    
    # If rounding or down, need to scale so that rounding is occurring to an integer
    else:
        direction = direction.lower()
        xscaled = x * 10.**roundto
        
        # Rounds up or down to nearest integer, depending on direction
        if direction == 'up':
            xscaled_rnd = ceil( xscaled )
        elif direction == 'down':
            xscaled_rnd = floor( xscaled )
        
        # If direction not recognized, raises error
        else:
            raise ValueError( "Value '{0}' not accepted for keyword direction. Accepted values are None, 'up', and 'down'.".format(direction) )
        
        # Once rounded, rescales back to original
        x_rnd = float( xscaled_rnd )* 10.**-roundto
        
        # May have acquired a floating point error, so does one final round, but to a higher precision
        outval = round( x_rnd, roundto+2 )
    
    # Returns the rounded value
    return outval

