# Last updated 4/26/2021
# Functions from other models/theories for comparison
# - Added gkk, which calculates and returns the fractional stokes q/i and EVPA for a single value of theta or an array 
#   of theta.

# Importing
from math import pi, atan, sin, sqrt
import numpy as np


def gkk( theta, units = 'degrees' ):
    """
    Calculates Fractional Stokes Q (Q/I) and EVPA from GKK method.
    
    Required Parameters:
        
        theta           Float or NumPy array of floats
                            The angle between the magnetic field and the line of sight. Units
                            correspond to those specified by optional parameter, units.
    
    Optional Parameters:
            
        units           String: 'degrees' (or 'deg' or 'd') OR 'radians' (or 'rad' or 'r')
                            [ Default = 'degrees' ]
                            The units in which the thetas values are provided.
    
    Returns:
        
        gkk_fracq       Float or NumPy array of floats
                            The fractional Stokes Q/I calculated by GKK at the requested 
                            angle(s), theta. Data type is the same as that provided for theta.
        
        gkk_evpa        Float or NumPy array of floats
                            The EVPA, in radians, calculated by GKK at the requested angle(s), 
                            theta. Data type is the same as that provided for theta.
    """
    # If theta is a list, makes it into a numpy array
    if isinstance( theta, list ):
        theta = np.array( theta )
    
    # If theta isn't a list/numpy array, make sure it's a float
    elif not isinstance( theta, np.ndarray ):
        theta = float(theta)
    
    
    # Convert theta to radians if provided in degrees
    if units.lower() in ['degrees', 'deg','d']:
        theta = theta * pi / 180.
    
    # If units weren't specified as radians or degrees, raises error
    elif units.lower() not in ['radians', 'rad', 'r']:
        err_msg = "Value provided for units not recognized. Accepted values are:\n" + \
                  "    'degrees', 'deg', 'd', 'radians', 'rad', 'r' (not case sensitive)."
        raise ValueError(err_msg)
    
    # Calculates a constant for easy reference
    at2 = atan( 1.0 / sqrt( 2.0 ) )
    
    # Splits calculation based on how theta is provided; does here if it's a numpy array
    if isinstance( theta, np.ndarray ):
        gkk_fracq = np.piecewise( theta, [ np.abs(theta) <= at2, np.abs(theta) > at2 ] , \
                                [ - 1.0, lambda x: ( 3.0*(np.sin(x))**2 - 2.0 ) / (3.0*(np.sin(x))**2) ])  
        gkk_evpa = 0.5 * np.arctan2( 0.0, gkk_fracq )
    
    # Calculates here if it's a single value
    else:
        if theta <= at2:
            gkk_fracq = -1.0
        else:
            gkk_fracq =   (  3.0*(sin(theta))**2 - 2.0  )   /   (  3.0*(sin(theta))**2  )
        gkk_evpa = 0.5 * atan2( 0.0, gkk_fracq )
    
    # Return fracq and evpa
    return gkk_fracq, gkk_evpa