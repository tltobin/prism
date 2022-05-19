# Last updated 5/19/2021
# - Debugging of contour generation by gen_contour_levels

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
    #print('level_step: {0}'.format(level_step))
    
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
    levels = np.arange( level_min, value_max, level_step )

    
    # Converts levels to a list and makes sure levels rounded to desired precision
    levels = [ round( lev, -exscale_diff ) for lev in levels ]
    
    # If last contour not included and it should be, adds
    if (len(levels) != 0 and level_max - levels[-1] == level_step):
        levels.append( level_max )
        
    # print(levels)
    # If there aren't the minimum number of contour levels, tries again with one lower precision
    while len(levels) < min_contours:
    
        # Adjusts the precision and step size if range is not symmetric about zero
        if min_contours < 2 * len(levels):
            exscale_diff -= 1
            level_step = 5.0 * 10.**exscale_diff
        elif min_contours < 5 * len(levels):
            exscale_diff -= 1
            level_step = 2.0 * 10.**exscale_diff
        else:
            exscale_diff -= 1
            level_step = 10.**exscale_diff
                
        # If not symmetric about zero, contour levels will go from ceiling of min to floor of max
        if value_min != -value_max:
            if value_min != 0.0:
                level_min = round_up_to( value_min, -exscale_diff )
            if value_max != 0.0:
                level_max = round_down_to( value_max, -exscale_diff )
    
            # Generates levels
            levels = np.arange( level_min, value_max, level_step )
    
            # Converts levels to a list and makes sure levels rounded to desired precision
            levels = [ round( lev, -exscale_diff ) for lev in levels ]
            if level_max - levels[-1] == level_step:
                levels = np.append( levels, level_max )
        
        # If range is symmetric about zero, want to make sure levels include 0
        else:
            # Creates the positive levels starting at first postitive value first
            pos_levs = np.arange( level_step, value_max, level_step )
            
            # Makes sure levels rounded to desired precision
            pos_levs = np.array([ round( lev, -exscale_diff ) for lev in pos_levs ])
            if level_max - pos_levs[-1] == level_step:
                pos_levs = np.append( pos_levs, level_max )
            
            # Creates levels array from that
            levels = -1.0 * np.array([ x for x in pos_levs[::-1] ])
            levels = np.append( levels, 0.0 )
            levels = np.hstack(( levels, pos_levs ))
            
            # Converts levels to a list
            levels = list(levels)
            
            
        #print('new level_step: {0}'.format( level_step ))
        #print(levels)
    

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

