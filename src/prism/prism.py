# Updates: 10/17/22 (v 1.0.0)
# - Changed R calculation from integrating over ang freq with trap. rule to summing over n.
# - Added option to get R at line center or sum over n, as desired.
# - Fixed bug switching code and comments later in code
# - Added 1/2pi factor into _maser_base_.calc_far_coeff calculation of far_coeff 
#       ( previously offset by user-side parameter selection )
#
# Updates: 11/8/22 (v 1.0.0)
# - Updated "Total $\tau$" labels in plotting functions to say $\tau_f$ instead of $\tau$ for
#   consistency with variables used in paper.
# - Renamed variable/attribute previously named 'tau0' -> 'fracLOS' for clarity. Updated here and in sample 
#   parameter file, templates/sample.par.
# - Renamed all variables containing 'beta' to equivalent with 'tauf' for consistency with paper. Updated here 
#   and in sample parameter file, templates/sample.par.
# - Since above edit changed naming convention of output deltas files, changed readin method in both maser and
#   maser_v_theta classes to look first for files with the new 'tauf' naming convention and, if those are not
#   found, look for files with the old 'beta' naming convention.
# - For maser and maser_v_theta class methods, readin, changed default file extension from ext = 'txt' to 
#   ext = 'fits'.






import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.optimize.nonlin import NoConvergence
from math import cos, sin, exp, pi, factorial, log10
import os
import configparser
from collections import OrderedDict
import matplotlib.pyplot as P
from scipy.interpolate import griddata
from itertools import cycle, islice



from .utils import _default_, string_to_list, string_to_bool, gen_contour_levels
from .const import e_charge, E0, me, c
from .prism_plot import color_sets, marker_sets, format_label_string_with_exponent, _update_label_
from .comparison import gkk


# saves original numpy print options 
orig_print = np.get_printoptions()

    

########################### Utility base class for par setting and faraday calc ############################

class _maser_base_:
    def __init__( self, parfile = None, ignore = [], \
                    omegabar = _default_( None ), fracLOS = _default_( None ), theta = _default_( None ), \
                    iquv0 = _default_( None ), W = _default_( None ), k = _default_( None ), \
                    phi = _default_( 0.0 ), n = _default_( 50 ), outpath = _default_( '' ), \
                    far_coeff = _default_( 0.0 ), etapm = _default_( 1.0 ), alphapm = _default_( 1.0 ), \
                    cloud = _default_( 1 ), iquvF = _default_( None ), \
                    taufs = _default_( None ), resume = _default_( False ), lastdelta = None, \
                    verbose = _default_( True ), ftol = _default_( 6e-10 ), filename = _default_( 'FaradayOut' ), \
                    endfill = _default_( 'zero' ), trend = _default_( False ), lastdelta2 = None):
        """
        Base class for handling parameters of maser and maser_v_theta. 
        
        Does not have 'tauf', 'tau', 'dtau' attributes or the associated method to set them, self.update_tauf.
        """
        
        # Does not include required parameters, iquvF
        
        # Section name of the parameter file, hard coded
        sect = 'MASER CLASS'
        self.sect = sect
        self.parfile = parfile
        
        # If parameter file is provided, reads
        if parfile is not None:
            self.conf = self._read_par_file_(parfile)
            
            # Checks that section is present in the parameter file
            if sect not in self.conf.sections():
                raise ValueError( 'Section {0} not found in file {1}'.format( sect, parfile ) )
                
        # If not provided, sets as dictionary with section name as None
        else:
            self.conf = { sect: None }
        
        
        
        
        
        
        ### Setting a number of defaultable parameters as attributes with possible specification in par file ###
        
        
        
        ## Parameter omegabar -- NumPy array, required, can also be set by omegabar_min, omegabar_max, d_omegabar
        if 'omegabar' not in ignore:
            self.omegabar = self._process_key_( 'omegabar', omegabar, self.conf[sect], \
                                allowed = { np.ndarray: [np.longdouble, None] }, convert = False , ignore_none = True  )
            
            # If omegabar is not set, checks for omegabar_min, omegabar_max, d_omegabar; all longdouble, conv allowed
            if self.omegabar is None:
                omegabar_min = self._process_key_( 'omegabar_min', _default_(None), self.conf[sect], \
                                                       allowed = { np.longdouble : None }, convert = True , ignore_none = True  )
                omegabar_max = self._process_key_( 'omegabar_max', _default_(None), self.conf[sect], \
                                                       allowed = { np.longdouble : None }, convert = True , ignore_none = True  )
                d_omegabar   = self._process_key_( 'd_omegabar'  , _default_(None), self.conf[sect], \
                                                       allowed = { np.longdouble: None }, convert = True , ignore_none = True  )
                if None not in (omegabar_min, omegabar_max, d_omegabar):
                    self.omegabar = np.arange( omegabar_min, omegabar_max, d_omegabar, dtype = np.longdouble )
                    self.omegabar = np.append( self.omegabar, np.longdouble(omegabar_max) )
                else:
                    print(omegabar_min, omegabar_max, d_omegabar)
                    raise ValueError( 'Keyword omegabar must be provided either on call or in parameter file either directly or\n' + \
                                        '    through the combination of (omegabar_min, omegabar_max, d_omegabar).' )
        
        
        ## Parameter fracLOS -- NumPy array, required, can also be set by taures
        if 'fracLOS' not in ignore:
            self.fracLOS = self._process_key_( 'fracLOS', fracLOS, self.conf[sect], \
                                allowed = { np.ndarray: [np.longdouble, None] }, convert = False , ignore_none = True  )
            
            # If fracLOS is not set, checks for taures; int, conv allowed
            if self.fracLOS is None:
                taures = self._process_key_( 'taures', _default_(None), self.conf[sect], \
                                                       allowed = { int: None }, convert = True , ignore_none = True  )
                if taures is not None:
                    self.fracLOS = np.linspace( 0, 1, taures, dtype = np.longdouble )
                else:
                    raise ValueError( 'Keyword fracLOS must be provided either on call or in parameter file either directly or\n' + \
                                '    through taures.' )
            
        
        ## Parameter iquv0 -- NumPy array of longdouble, length 4, required
        if 'iquv0' not in ignore:
            self.iquv0 = self._process_key_( 'iquv0', iquv0, self.conf[sect], \
                                allowed = { np.ndarray: [np.longdouble, 4] }, convert = False , ignore_none = True  )
            if self.iquv0 is None:
                raise ValueError( 'Keyword iquv0 must be provided either on call or in parameter file.' )
            
        
        ## Parameter theta -- float, required
        if 'theta' not in ignore:
            self.theta = self._process_key_( 'theta', theta, self.conf[sect], allowed = { float: None }, \
                                                                                convert = True , ignore_none = True  )
            if self.theta is None:
                raise ValueError( 'Keyword theta must be provided either on call or in parameter file.' )
            
            # If theta successfully set, sets sin(theta) and cos(theta)
            else:
                self.sintheta = sin(self.theta)
                self.costheta = cos(self.theta)
            
        
        ## Parameter W -- float, required
        if 'W' not in ignore:
            self.W = self._process_key_( 'W', W, self.conf[sect], allowed = { float: None }, \
                                                                                convert = True , ignore_none = True  )
            if self.W is None:
                raise ValueError( 'Keyword W must be provided either on call or in parameter file.' )
            
        
        ## Parameter k -- int, required
        if 'k' not in ignore:
            self.k = self._process_key_( 'k', k, self.conf[sect], allowed = { int: None }, \
                                                                                convert = True , ignore_none = True  )
            if self.k is None:
                raise ValueError( 'Keyword k must be provided either on call or in parameter file.' )
            
                
        ## Parameter phi -- float, conversion allowed, can be specified in parfile, none invalid
        if 'phi' not in ignore:
            self.phi = self._process_key_( 'phi', phi, self.conf[sect], allowed = { float: None }, \
                                                                                convert = True , ignore_none = True  )
        
            # Calculates sin and cos of 2 phi
            self.sintwophi = sin(2.*self.phi)
            self.costwophi = cos(2.*self.phi)
        
        
        ## Parameter n -- integer, conversion allowed, can be specified in parfile, none invalid
        if 'n' not in ignore:
            self.n = self._process_key_( 'n', n, self.conf[sect], allowed = { int: None }, convert = True, ignore_none = True )
        
        
        ## Parameter outpath -- string, conversion disabled, can be specified in parfile, none invalid
        #      If not length-0 string, must end with a '/'
        if 'outpath' not in ignore:
            self.outpath = self._process_key_( 'outpath', outpath, self.conf[sect], allowed = { str: None }, \
                                                                                convert = False, ignore_none = True  )
            if len( self.outpath ) > 0 and not self.outpath.endswith('/'):
                self.outpath = '{0}/'.format( self.outpath )
        
        
        ## Parameter far_coeff -- float, conversion allowed, can be specified in parfile, none invalid
        if 'far_coeff' not in ignore:
            self.far_coeff = self._process_key_( 'far_coeff', far_coeff, self.conf[sect], allowed = { float: None }, \
                                                                                convert = True , ignore_none = True  )
        
        
        ## Parameter etap and etam via etapm -- list/array of floats OR single float, conversion allowed, can be 
        #    specified in parfile, none invalid
        #  print('Starting etapm...')  # Line for debugging
        if 'etapm' not in ignore:
            etapm_temp = self._process_key_( 'etapm', etapm, self.conf[sect], \
                         allowed = OrderedDict([ (list, [float,2]), (float, None) ]), convert = True , ignore_none = True  )
            if isinstance( etapm_temp, float ):
                etapm_temp = [ etapm_temp, etapm_temp ]
            self.etap = etapm_temp[0]
            self.etam = etapm_temp[1]
        
        
        ## Parameter alpha and alpham via alphapm -- list/array of floats OR single float, conversion allowed, can be 
        #    specified in parfile, none invalid
        #  print('Starting alphapm...')  # Line for debugging
        if 'alpmapm' not in ignore:
            alphapm_temp = self._process_key_( 'alphapm', alphapm, self.conf[sect], \
                         allowed = OrderedDict([ (list, [float,2]), (float, None) ]), convert = True , ignore_none = True  )
            if isinstance( alphapm_temp, float ):
                alphapm_temp = [ alphapm_temp, alphapm_temp ]
            self.alphap = alphapm_temp[0]
            self.alpham = alphapm_temp[1]
        
        
        ## Parameter cloud -- integer, must be 1 or 2, conversion allowed, can be specified in parfile, none invalid
        if 'cloud' not in ignore:
            self.cloud = self._process_key_( 'cloud', cloud, self.conf[sect], allowed = { int: [1,2] }, \
                                                                                convert = True , ignore_none = True  )
        
        
        ## Parameter iquvF -- len-4 numpy array of floats, only set if cloud is 2
        if 'iquvF' not in ignore and 'iquv0' not in ignore and 'cloud' not in ignore:
            if self.cloud == 1:
                self.iquvF = None
            elif self.cloud == 2:
                # Uses _maser_base_ class method to resolve priorities
                self.iquvF = self._process_key_( 'iquvF', iquvF, self.conf[sect], \
                         allowed = OrderedDict([ (np.ndarray, [float,4]), (None, None) ]), convert = False , ignore_none = False  )
            
                # If value returned is NoneType, assumes that means iquvF = iquv0
                if self.iquvF is None:
                    self.iquvF = np.array( self.iquv0 )
        
        
        ## Parameter taufs -- list/array, float, or nonetype, conversion allowed, can be specified in parfile, 
        #    none is valid 
        #  print('Starting taufs...')  # Line for debugging
        if 'taufs' not in ignore:
            self.taufs = self._process_key_( 'taufs', taufs, self.conf[sect], \
                         allowed = OrderedDict([ (np.ndarray, [float,None]), (float, None), (None, None) ]), \
                         convert = True , ignore_none = False )
            
            # If taufs is single value, converts it to a length-1 numpy array 
            if isinstance( self.taufs, float ):
                self.taufs = np.array([ self.taufs ])
            
        
        ## Parameter resume -- boolean, conversion not allowed, can be specified in parfile, none invalid
        if 'resume' not in ignore:
            self.resume = self._process_key_( 'resume', resume, self.conf[sect], allowed = { bool: None }, \
                                                                                convert = False, ignore_none = True  )
        
        
        ## Parameter lastdelta -- numpy array or None; not compatible with config file
        if 'lastdelta' not in ignore:
            if lastdelta is not None and not isinstance( lastdelta, np.ndarray ):
                raise ValueError( 'Keyword {0} must be data type {1}. (Current type: {2})'.format('lastdelta', \
                                                                        'NumPy array or NoneType', lastdelta.dtype ) )
            self.lastdelta = lastdelta
        
        
        ## Parameter verbose -- boolean, conversion not allowed, can be specified in parfile, none invalid
        if 'verbose' not in ignore:
            self.verbose = self._process_key_( 'verbose', verbose, self.conf[sect], allowed = { bool: None }, \
                                                                                convert = False, ignore_none = True  )
        
        
        ## Parameter ftol -- float, conversion allowed, can be specified in parfile, none invalid
        if 'ftol' not in ignore:
            self.ftol = self._process_key_( 'ftol', ftol, self.conf[sect], allowed = { float: None }, \
                                                                                convert = True , ignore_none = True  )
        
        
        ## Parameter filename -- string, conversion disabled, can be specified in parfile, none invalid
        #    Must have length > 0
        if 'filename' not in ignore:
            self.filename = self._process_key_( 'filename', filename, self.conf[sect], allowed = { str: None }, \
                                                                                    convert = False, ignore_none = True  )
            if len( self.filename ) == 0:
                raise ValueError( 'Keyword {0} cannot be empty string'.format( 'filename' ) )
        
        
        ## Parameter endfill -- string, must be 'fit' or 'zero', conversion disabled, can be specified in parfile, 
        #    none invalid
        if 'endfill' not in ignore:
            self.endfill = self._process_key_( 'endfill', endfill, self.conf[sect], allowed = { str: ['fit','zero'] }, \
                                                                                    convert = False, ignore_none = True  )
        
        
        ## Parameter trend -- bool or string (must be 'auto' if string), conversion enabled, can be specified in parfile,
        #    none invalid
        #  print('Starting trend...')  # Line for debugging
        if 'trend' not in ignore:
            self.trend = self._process_key_( 'trend', trend, self.conf[sect], \
                         allowed = OrderedDict([ (bool, None), (str, ['auto']) ]), convert = True , ignore_none = True  )
        
        ## Parameter lastdelta2 -- numpy array or None; not compatible with config file
        if 'lastdelta2' not in ignore:
            if lastdelta2 is not None and not isinstance( lastdelta2, np.ndarray ):
                raise ValueError( 'Keyword {0} must be data type {1}. (Current type: {2})'.format('lastdelta2', \
                                                                        'NumPy array or NoneType', lastdelta2.dtype ) )
            self.lastdelta2 = lastdelta2
        
        
        # Sets boolean saying that far_coeff has not (yet) been calculated by calc_far_coeff
        self.fccalc = False
    
    def calc_far_coeff(self, ne, freq0, Gam, B, A0, P0, W, mode='cm' ):
        """
        Calculates the faraday coefficient, gamma_QU/cos(theta) given:
            ne      = electron density [ cm^-3 or m^-3 ]
            freq0   = rest frequency of the line [ Hz ]
            Gamma     = loss rate [ s^-1 ]
            B       = magnetic field strength [Gauss]
            A0      = Einstein A coefficient [ s^-1 ]
            P0      = pump rate into the 0 state [cm^-3 s^-1 or m^-3 s^-1 ]
            W       = Doppler width [Hz]
        
        Keyword mode can be set to 'cm' or 'm' to specify units of the given ne and P0 values.
        Default is 'cm'. If set to 'cm', these values will be converted to SI prior to calculation.
        
        Overwrites self.far_coeff.
        """
        
        # First checks if mode is cm
        if mode == 'cm':
            
            # If in cm mode, converts to SI
            ne = ne * 10.**6
            P0 = P0 * 10.**6
        
        # Calculates small w from big W 
        w = W * c / ( 2.*pi*freq0 )     # Width parameter in velocity space in m/s
        
        # Uses resulting values to calculate
        CONSTS = ( 8. * e_charge**3 * pi**.5 ) / ( 3. * E0 * me**2 * c**4 )
        PARS = ( ne * freq0 * Gam * w * B ) / ( A0 * P0 )
        # 1/2pi factor
        CONSTS = CONSTS / ( 2. * pi )
        self.far_coeff = CONSTS * PARS
        
        # Saves info for writing simulation description file
        self.fccalc = True     # Boolean saying that far_coeff was calculated by this function
        self.ne = ne
        self.P0 = P0
        self.freq0 = freq0
        self.Gamma = Gam
        self.B = B
        self.A0 = A0
    
    def _process_key_(self, keyname, keyvalue, confsection, allowed = {}, ignore_none = True, convert = False ):
        """
        Utility function to process key values by priority: specified on call --> conf file (if provided) --> 
        default.
        
        Can filter for allowed data types and values using allowed dictionary (optional keyword).
        
        Required parameters:
            
            keyname         String
            
                                The name of the parameter as it appears in the config file (if provided). Also 
                                used in error messages.
                                
            keyvalue        (Any)
            
                                Value of the parameter set on function call. Method will distinguish if it is 
                                the default parameter class or user-specified.
                                
            confsection     ConfigParser SectionProxy object, or None
            
                                To refer to a config file when the key is not specified directly on call, 
                                supply the section of the relevent config file, as read in by configparser, 
                                here, eg:
                                
                                    >>> conf = configparser.ConfigParser(inline_comment_prefixes=['#']) 
                                    >>> conf.read( 'my_config_file.par')
                                    >>> self._process_key_( keyname, keyvalue, conf['Relevent Section Name'] )
                                    
                                If no config file is provided or desired for cross-referencing, enter None.
        
        Optional parameters:
            
            allowed         Dict or OrderedDict
            
                                [ Default = dict() ]
                                
                                Used to check that values provided by user and read from the parameter file 
                                (if applicable) are the appropriate data types/values for the parameter. 
                                
                                Dictionary format should have the allowed data types as dictionary keys. If 
                                only specific values for a given data type are allowed, then the dictionary 
                                entry for that data type will point to a list or tuple of the allowed values 
                                associated with that data type. If any values of a given data type are 
                                allowed, the dictionary entry for that data type will be None.
                                
                                For example, to check a keyword that can be either a boolean or a string named 
                                'auto':
                                
                                    >>> allowed = { bool: None, str: [ 'auto' ] }
                                
                                If there is an order in which the data types should be checked, use an 
                                OrderedDict instead of a Dict object for allowed:
                                    
                                    >>> # If we need to check if a string is None, but it can also be a string
                                    >>> from collections import OrderedDict
                                    >>> allowed = OrderedDict()
                                    >>> allowed[None] = None
                                    >>> allowed[str]  = None
                                
                                If the data type is a list or NumPy array, instead of specific values allowed 
                                for the keyword, the allowed dictionary should point to a length-2 list with 
                                the allowed data types and length of the list. Either can be specified as None 
                                to remove constraints. Multiple data types can be specified with nesting, 
                                which will be taken in priority order:
                                    
                                    >>> # To require the value to be a length-4 list of floats
                                    >>> allowed = { list: [ float, 4 ] }
                                    >>>
                                    >>> # To require the value to be list of float or string of any length, 
                                    >>> #    with floats preferred over strings
                                    >>> allowed = { list: [ ( float, str ), None ] }
                                
                                Note: If data type is specified for a list/array, values will be converted, if 
                                possible.
                                
                                List and NumPy array are considered interchangable for checking data type, but 
                                whichever is specified in the allowed dictionary is what the item will be 
                                returned as.
                                    
                                    >>> # The above examples will return a list object
                                    >>> # To repeat the len-4 list of floats returned as a numpy array
                                    >>> allowed = { np.ndarray: [ float, 4 ] }
                                    
                                Note 1: If no data type is specified for a list/array type object and data is 
                                        read in from the config file, the data type of the values in the list 
                                        will be a string.
                                        
                                Note 2: The data type of a numpy array is numpy.ndarray, NOT numpy.array.
            
            ignore_none     Boolean 
            
                                [ Default = True ]
                                Whether to treat any parameters set as None in the config file as being unset 
                                (True) or treat None as a viable parameter for the key (False). Only used if a 
                                confsection is provided.
            
            convert         Boolean
            
                                [ Default = False ]
                                
                                Whether to try to convert any user-provided values into the data types, or 
                                simply check if they have the correct data type. This only applies to values 
                                provided on object call, not any read from the config file or built in 
                                defaults. Note: If convert is turned on, and multiple data types are 
                                acceptable, you MUST use an OrderedDict for your allowed values to ensure 
                                consistent type conversion.
            
        Returns:
            
           out_value        The value of the key, as determined by prioritizing on call -> conf file -> 
           					default.
           					
                            Will have one of the data types specified by the allowed dictionary. 
                                
        """
                
        # Sets output value as empty default; will override if correct data type is found
        out_value = _default_(None)
        
        # First, checks if the key value provided on object initialization was specified explicitly (i.e. is not default)
        #     If so, just uses that value
        if not isinstance( keyvalue, _default_ ):
            
            # Checks allowed data types and values, if provided
            if len(allowed.keys()) > 0:
                
                # Begins iterating through the data types; if allowed is an OrderedDict, these will be in the specified
                #   order
                for allowed_dtype in allowed.keys():
                    
                    # Checks if the provided value has that data type
                    # First, does the single values that are not list/tuple/array
                    if allowed_dtype not in [list, np.ndarray]:
                        if not convert or allowed_dtype is None:
                            if isinstance( keyvalue, allowed_dtype ):
                            
                                # Checks any allowed specific values and sets out_value
                                if allowed[allowed_dtype] is None or ( keyvalue in allowed[allowed_dtype] ):
                                    out_value = keyvalue
                                else:
                                    raise ValueError( "Accepted values for keyword {0} as data type {1} are {2}.\n    (Current value: '{3}')".format( \
                                        keyname, allowed_dtype.__name__, ', '.join(allowed[allowed_dtype]), keyvalue ) )
                        
                        # If conversion is turned on, tries converting to the desired data type
                        elif convert:
                            try:
                                if allowed_dtype is bool and isinstance(keyvalue, str):
                                    test_conversion = string_to_bool( keyvalue )
                                else:
                                    test_conversion = allowed_dtype( keyvalue )
                            except:
                                pass
                            else:
                                # Checks any allowed specific values and sets out_value
                                if allowed[allowed_dtype] is None or ( test_conversion in allowed[allowed_dtype] ):
                                    out_value = test_conversion
                                else:
                                    raise ValueError( "Accepted values for keyword {0} as data type {1} are {2}.\n    (Current value: '{3}')".format( \
                                        keyname, allowed_dtype.__name__, ', '.join(allowed[allowed_dtype]), test_conversion ) )
                        
                    # Processes any list type objects down here; checks immediately if the value is a list or array
                    elif isinstance( keyvalue, list ) or isinstance( keyvalue, np.ndarray ):
                        
                        # If no constraints on the list/array, just makes sure that output data type is as requested
                        if allowed[allowed_dtype] is None or ( len(allowed[allowed_dtype])==2 and list(allowed[allowed_dtype]) == [None,None] ):
                            if allowed_dtype is list:
                                out_value = list( keyvalue )
                            else:
                                out_value = np.array( keyvalue )
                        
                        # If there are constraints on the data type or length, processes
                        else:
                            if allowed[allowed_dtype][0] is not None:
                                
                                # If only one data type for a list/array is specified, makes that into len-1 list
                                if not isinstance( allowed[allowed_dtype][0], tuple ) and \
                                                                not isinstance( allowed[allowed_dtype][0], list ):
                                    
                                    allowed[allowed_dtype][0] = [ allowed[allowed_dtype][0] , ]
                                    
                                # Iterates through accepted data types, trying to convert
                                for allowed_list_dtype in allowed[allowed_dtype][0]:
                                    try:
                                        test_conversion = np.array( keyvalue ).astype( allowed_list_dtype )
                                    except:
                                        pass
                                    # If the conversion works, checks the length, if any, and saves that as the output 
                                    #   value and breaks the for loop
                                    else:
                                        if isinstance(allowed[allowed_dtype][1],int):
                                            allowed[allowed_dtype][1] = [ allowed[allowed_dtype][1], ]
                                        if allowed[allowed_dtype][1] is None or test_conversion.size in allowed[allowed_dtype][1]:
                                            if allowed_dtype is list:
                                                out_value = list( test_conversion )
                                            else:
                                                out_value = np.array( test_conversion )
                                        else:
                                            raise ValueError( 'Keyword {0} list/array must have length {1}. (Current length {2})'.format( \
                                                            keyname, ', '.join(allowed[allowed_dtype][1]),  test_conversion.size ) )
                                            
                                        break
                                
                                # If value is a list/array but not one of the allowed data types, raises error
                                if isinstance( out_value, _default_ ):
                                    dtnames = [ x.__name__ for x in allowed[allowed_dtype][0] ]
                                    raise ValueError( 'Values in {0} list/array must be data type {1}.'.format( \
                                                                        keyname, ', '.join(dtnames) ) )
                                
                            # If no constraints on data type for list and there are length constraints, checks
                            else:
                                if isinstance(allowed[allowed_dtype][1],int):
                                    allowed[allowed_dtype][1] = [ allowed[allowed_dtype][1], ]
                                if np.array(keyvalue).size in allowed[allowed_dtype][1]:
                                    out_value = allowed_dtype( keyvalue )
                                else:
                                    raise ValueError( 'Keyword {0} list/array must have length {1}. (Current length {2})'.format( \
                                                    keyname, ', '.join(allowed[allowed_dtype][1]),  np.array(keyvalue).size ) )
                        
                    # If the data type to be checked is a list or array and the value is not, skips
                
                    # At the end of checking a given allowed dictionary entry, breaks the loop if out_value has been
                    #   successfully set
                    if not isinstance( out_value, _default_ ):
                        break
                
                # If none of the allowed data types match the value provided, raises an exception
                if isinstance( out_value, _default_ ):
                    dtnames = [ x.__name__ for x in allowed.keys() ]
                    if 'list' in dtnames: 
                        dtnames.append('ndarray')
                    elif 'ndarray' in dtnames:
                        dtnames.append('list')
                    raise ValueError( 'Acceptable data types for keyword {0} are {1}. (Current type: {2})'.format(keyname, '/'.join(dtnames), keyvalue.dtype ) )
                
            # If no constraints on allowed values, just sets and returns
            else:
                out_value = keyvalue
            
        
        # If value isn't explicitly provided on class initialization, checks if (1) parameter file section is provided,
        #   (2) key name appears in the parameter file, and (3) keyword in parameter file isn't set to be ignored by
        #   being marked as none when none values are being ignored (ignore_none)
        elif confsection is not None and keyname in confsection and not (ignore_none and confsection[keyname].lower() == 'none') :
            
            
            # Checks allowed data types and values, in order, if provided
            # Processing here includes converting data type from string to desired type, so some data types need
            #   different conversion functions
            if len(allowed.keys()) > 0:
                val_from_conf = confsection[keyname].strip()
                for allowed_dtype in allowed.keys():
                    
                    # Does lists first; these must start with [ and end with ]
                    if allowed_dtype in [list, np.ndarray]:
                        if val_from_conf.startswith('[') and val_from_conf.endswith(']'):
                            
                            # If no constraints on the list/array, just makes sure that output data type is as requested
                            # Values within the list/array will still be strings
                            if allowed[allowed_dtype] is None or ( len(allowed[allowed_dtype])==2 and list(allowed[allowed_dtype]) == [None,None] ):
                                if allowed_dtype is list:
                                    out_value = string_to_list( val_from_conf )
                                else:
                                    out_value = np.array( string_to_list( val_from_conf ) )
                                                
                            # If there are constraints on the data type, processes
                            elif allowed[allowed_dtype][0] is not None:
                            
                                # If only one data type for a list/array is specified, makes that into len-1 list
                                if not isinstance( allowed[allowed_dtype][0], tuple ) and \
                                                                    not isinstance( allowed[allowed_dtype][0], list ):
                                    allowed[allowed_dtype][0] = [ allowed[allowed_dtype][0] , ]
                                
                                # Iterates through accepted data types, trying to convert
                                for allowed_list_dtype in allowed[allowed_dtype][0]:
                                    try:
                                        test_conversion = string_to_list( val_from_conf, astype = allowed_list_dtype )
                                    except:
                                        pass
                                        
                                    # If the conversion works, checks the length, if any, and saves that as the output 
                                    #   value and breaks the for loop
                                    else:
                                        if isinstance(allowed[allowed_dtype][1],int):
                                            allowed[allowed_dtype][1] = [ allowed[allowed_dtype][1], ]
                                        if allowed[allowed_dtype][1] is None or len(test_conversion) in allowed[allowed_dtype][1]:
                                            if allowed_dtype is list:
                                                out_value = list( test_conversion )
                                            else:
                                                out_value = np.array( test_conversion )
                                        else:
                                            raise ValueError( 'Keyword {0} list/array must have length {1} in parameter file. (Current length {2})'.format( \
                                                            keyname, ', '.join(allowed[allowed_dtype][1]),  test_conversion.size ) )
                                        break
                                        
                                
                                # If value is a list/array but not one of the allowed data types, raises error
                                if isinstance( out_value, _default_ ):
                                    dtnames = [ x.__name__ for x in allowed[allowed_dtype][0] ]
                                    raise ValueError( 'Values in {0} list/array must be data type {1} in parameter file.'.format( \
                                                                                        keyname, ', '.join(dtnames) ) )
                            
                            # If no constraints on data type for list and there are length constraints, checks
                            else:
                                if isinstance(allowed[allowed_dtype][1],int):
                                    allowed[allowed_dtype][1] = [ allowed[allowed_dtype][1], ]
                                test_conversion = string_to_list( val_from_conf )
                                if len(test_conversion) in allowed[allowed_dtype][1]:
                                    if allowed_dtype is list:
                                        out_value = list( test_conversion )
                                    else:
                                        out_value = np.array( test_conversion )
                                else:
                                    raise ValueError( 'Keyword {0} list/array must have length {1} in parameter file. (Current length {2})'.format( \
                                                    keyname, ', '.join(allowed[allowed_dtype][1]),  len(test_conversion) ) )
                    
                    
                    # Checks datatypes int, float, and numpy longdouble, which can be evaluated from a string
                    elif allowed_dtype in [int, float, np.longdouble]:
                        
                        # Tries to convert
                        try:
                            test_conversion = allowed_dtype( eval( val_from_conf ) )
                        except:
                            pass
                        else:
                            # Checks if there are specific allowable values for this data type
                            if allowed[allowed_dtype] is None or ( test_conversion in allowed[allowed_dtype] ):
                                out_value = test_conversion
                            else:
                                raise ValueError( "Accepted values for keyword {0} as data type {1} are {2} in parameter file.\n    (Current value: '{3}')".format( \
                                    keyname, allowed_dtype.__name__, ', '.join(allowed[allowed_dtype]), test_conversion ) )
                    
                    
                    # If data type is a bool, converts with specialized function from utils
                    elif allowed_dtype is bool:
                        
                        # Converts to bool if 'true'/'false' (not case sensitive); if neither, leaves as a string
                        test_conversion = string_to_bool( val_from_conf )
                        if isinstance( test_conversion, bool ):
                            
                            # Checks if there are specific allowable values for this data type
                            if allowed[allowed_dtype] is None or ( test_conversion in allowed[allowed_dtype] ):
                                out_value = test_conversion
                            else:
                                raise ValueError( "Accepted values for keyword {0} as data type {1} are {2} in parameter file.\n    (Current value: '{3}')".format( \
                                    keyname, allowed_dtype.__name__, ', '.join(allowed[allowed_dtype]), test_conversion ) )
                    
                    
                    # If data type is NoneType (ignore_none must be turned off to make it here)
                    elif allowed_dtype is None and val_from_conf.lower() == 'none':
                        out_value = None
                    
                    
                    # Any other data types
                    else:
                        
                        # Tries to convert
                        try:
                            test_conversion = allowed_dtype( val_from_conf )
                        except:
                            pass
                        else:
                            # Checks if there are specific allowable values for this data type
                            if allowed[allowed_dtype] is None or ( test_conversion in allowed[allowed_dtype] ):
                                out_value = test_conversion
                            else:
                                raise ValueError( "Accepted values for keyword {0} as data type {1} are {2} in parameter file.\n    (Current value: '{3}')".format( \
                                    keyname, allowed_dtype.__name__, ', '.join(allowed[allowed_dtype]), test_conversion ) )
                    
                    
                    # At the end of checking a given allowed dictionary entry, breaks the loop if out_value has been
                    #   successfully set
                    if not isinstance( out_value, _default_ ):
                        break
                
                # If none of the allowed data types match the value provided, raises an exception
                if isinstance( out_value, _default_ ):
                    dtnames = [ x.__name__ for x in allowed.keys() ]
                    if 'list' in dtnames: 
                        dtnames.append('ndarray')
                    elif 'ndarray' in dtnames:
                        dtnames.append('list')
                    raise ValueError( 'Acceptable data types for keyword {0} are {1} in parameter file. (Current value: {2})'.format(keyname, '/'.join(dtnames), val_from_conf ) )
                    
            # If no constraints on allowed values and not ignoring, just sets as the output value
            else:
                out_value = val_from_conf
            
        # If value hasn't been set yet, uses default, which should be value of the _default_ class object provided as
        #   keyvalue
        # Note: default class objects assumed to be pre-vetted for validity; not checked for data type/value
        else:
            out_value = keyvalue.value
                    
        
        # Returns the output value
        return out_value
    
    def _read_par_file_(self, parfile ):
        """
        Checks if a parameter file exists and, if so, reads with configparser.
        
        Allows comments prefixed by '#'.
        
        Required Parameters:
            
            parfile         String
            
                                Path/name of the parameter file.
        
        Returned:
            
            conf            configparser.ConfigParser object
            
                                The configuration file in the form of a configparser.ConfigParser object.
        
        """
        
        # First, checks that the config file exists
        if not os.path.exists( parfile ):
            raise FileNotFoundError( 'Parameter file {0} does not exist.'.format(parfile) )
        
        # Reads in the config file, allowing for in-line comments with '#'
        conf = configparser.ConfigParser(inline_comment_prefixes=['#'])
        successful_files = conf.read( parfile )

        # If no files were successfully read, raises an error
        if len( successful_files ) == 0:
            raise ValueError( 'Parsing of config file {0} failed.'.format( parfile ) )

        # Returns config file object
        return conf
        


############################### Primary object class for single parameter set ###############################

class maser(_maser_base_):
    def __init__(self, parfile = None, \
                    omegabar = _default_( None ), fracLOS = _default_( None ), theta = _default_( None ), \
                    iquv0 = _default_( None ), W = _default_( None ), k = _default_( None ), \
                    phi = _default_( 0.0 ), n = _default_( 50 ), outpath = _default_( '' ), \
                    far_coeff = _default_( 0.0 ), etapm = _default_( 1.0 ), alphapm = _default_( 1.0 ), \
                    cloud = _default_( 1 ), iquvF = _default_( None ), \
                    taufs = _default_( None ), resume = _default_( False ), lastdelta = None, \
                    verbose = _default_( True ), ftol = _default_( 6e-10 ), filename = _default_( 'FaradayOut' ), \
                    endfill = _default_( 'zero' ), trend = _default_( False ), lastdelta2 = None ):
        """
        Object for calculating the dimensionless population inversions for a given parameter set.
        
        Initializing the object establishes object attributes described below. It does not calculate the 
        Faraday coefficient from provided terms (see calc_far_coeff method), find the best fit population 
        inversions (see run method), or read in output files from previous runs (see readin method).
        
        
        Optional Parameters:
            
            parfile         String
            
                                If provided, gives the path and file name (from current directory) of a 
                                parameter file containing values for any of the keywords accepted by this 
                                object class initialization. Values in this parameter file will override any 
                                default values. 
                                
                                Parameter file ingestion also allows the specification of the omegabar array 
                                by min, max, and stepsize, as well as the specification of fracLOS by number 
                                of resolution elements (both of which are not currently supported when set 
                                explicitly on object initialization.)
            
            
                        --- Parameters that must be either provided or read from parameter file ---
                                            --- (see parfile parameter above) ---
            
            omegabar        NumPy Array
            
                                Array of angular frequencies [s^-1] relative to line center for each angular 
                                frequency bin. 
                                
                                Should be 1D with length NV + 4k, where NV is the number of angular frequency 
                                bins *not* removed by edge effects when calculating population inversions, 
                                delta. The 2k angular frequency bins on either end of the frequency range will 
                                be lost during calculation, and will be accounted for by the edge handling
                                method specified by optional keyword endfill. 
                                
                                Saved as object attribute, omegabar. 
                                
                                NOTE: NumPy longdouble precision recommended.
                                
            fracLOS         NumPy Array
            
                                Array establishing bin locations through the cloud (fracLOS = cloud fraction
                                along Line Of Sight). Values in array should indicate the fraction of the 
                                cloud through which the ray has passed, ranging from 0 to 1, inclusive. (If 
                                cloud=2, this indicates the fraction of the cloud through which Ray 1 has 
                                passed.) Will be multiplied by the total optical depth (tauf) for the cloud 
                                for calculation. 
                                
                                Saved as object attribute, fracLOS. 
                                
                                NOTE: NumPy longdouble precision recommended.
                                
            theta           Float
            
                                The angle between the magnetic field and the line of sight [radians].
                                
                                If cloud=2, this will be taken as the theta for Ray 1, where Ray 2 has 
                                theta_2 = - theta. 
                                
                                Saved as object attribute, theta.
                                
            iquv0           Length-4 NumPy Array
            
                                The initial values of (unitless) Stokes i, q, u, and v for the ray prior to 
                                passing through the cloud. If cloud=2, these values are used for Ray 1 only. 
                                Use optional parameter iquvF to set the corresponding values for Ray 2 (see 
                                below). 
                                
                                Saved as object attribute, iquv0.
                                
            W               Float
            
                                The Doppler Width in angular frequency [s^-1]. 
                                
                                Saved as object attribute, W. 
                                
            k               Integer
            
                                The number of angular frequency bins in omegabar spanned by the Zeeman shift, 
                                delta omega. 
                                
                                Saved as object attribute, k.
             
                               --- Important parameters with useful defaults for all applications ---
                                        
            phi             Float   
            
                                [ Default = 0.0 ]
                                
                                The sky-plane angle [radians]. If cloud=2, this will be taken as the phi for 
                                Ray 1, where Ray 2 has phi_2 = - phi. 
                                
                                Saved as object attribute, phi.
                            
            n               Integer
                                [ Default = 50 ]
                                The number of terms in the LDI expansion. Counting begins at 0. Saves as 
                                object attribute, n.
            outpath         String
                                [ Default = '' ]
                                The directory path (from current directory) to which the output 
                                dimensionless inversions will be saved for each tauf value. Saved as 
                                object attribute, outpath.
            far_coeff       Float
                                [ Default = 0.0 ]
                                Unitless value gives -gamma_QU / cos(theta). Can either be specified 
                                explicitly here or calculated from components later by calc_far_coeff 
                                method. Saved as object attribute, far_coeff.
            etapm           Float or Length-2 List or NumPy Array
                                [ Default = 1.0 ]
                                Contains values [ eta^+ , eta^- ], where each corresponds to the
                                relative line strengths between the +/- transition (respectively) and 
                                the 0 transition. I.e. eta^+ = | dhat^+ |^2 / | dhat^0 |^2 and 
                                eta^- = | dhat^- |^2 / | dhat^0 |^2, where the dhat terms are the 
                                dipole moments of each transition. Unitless. If specified as a float 
                                instead of a len-2 list/array, will use the value specified as both 
                                eta^+ and eta^-. Saves eta^+ as object attribute etap and eta^- as 
                                object attribute etam. 
            alphapm         Float or Length-2 List or NumPy Array
                                [ Default = 1.0 ]
                                Contains values [ alpha^+ , alpha^- ], where each corresponds to the
                                ratio between the pump rate of the +/- state, respectively, and that
                                of the 0 state. I.e. alpha^+ = P^+ / P^0 and alpha^- = P^- / P^0. 
                                Unitless. If specified as a float instead of a len-2 list/array, will 
                                use the value specified as both alpha^+ and alpha^-. Saves alpha^+ as 
                                object attribute alphap and alpha^- as object attribute alpha^-. 
            
                                      --- Parameters Needed for Bidirectional Solutions ---
                                    
            cloud           1 or 2  
                                [ Default = 1 ]
                                Switch indicating whether the solution should be calculated for a 
                                uni-directional cloud (cloud=1) or a bi-directional cloud (cloud=2). 
                                If cloud = 2, requires optional parameter iquvF to be set. Saved as 
                                object attribute, cloud.
            iquvF           Length-4 NumPy Array    
                                [ Default = None ]
                                The initial values of (unitless) Stokes i, q, u, and v for Ray 2 prior 
                                to passing through the cloud. Only used if cloud=2. If cloud=2 and iquvF
                                is not provided (i.e., is None), will use values provided for iquv0 as 
                                iquvF. Saved as object attribute, iquvF, if cloud=2. 
                            
                                 --- Parameters Needed when Starting or Resuming a Solution ---
                                      
            taufs           Float or NumPy Array
                                [ Default = None ]
                                Value or an array of total optical depths for the cloud. Unitless.
                                IF INITIALIZING A NEW CALCULATION, SET THIS TO NOT BE NONE. If
                                reading in prior calculation, setting this is not required. 
                                Length should be N, where N is the total number of solutions that will 
                                be calculated. Values should be specified in increasing order, as 
                                the solution for each optical depth will be used as the initial guess 
                                for the next, and computation time increases with increasing optical
                                depth. If no prior results are available, recommend starting with the
                                first value in this array < 1.0. Each value will be multiplied by
                                array fracLOS to set the optical depth transversed the cloud at each 
                                spatial resolution point. Saved as object attribute, taufs.
                                
                                      --- Parameters Needed when Resuming a Previous Solution ---
                                      
            resume          Boolean
                                [ Default = False ]
                                Whether the solving method, run, will begin with no prior information 
                                about the deltas array (resume=False) or if this is a continuation of a 
                                previous attempt at solving (resume=True). If the former, the initial 
                                guess for the first solution with taufs[0] will be an array of ones. If 
                                you wish to continue a previous solving run, set resume to be True and 
                                optional parameter lastdelta (see below) to be the last known array of 
                                deltas, which will be used as the inital guess for the new taufs[0] 
                                solution. May also use trend fitting to extrapolate an initial guess 
                                for deltas using lastdelta and lastdelta2. Saved as object attribute, 
                                resume.
            lastdelta       NumPy Array
                                [ Default = None ]
                                The array of deltas to be used as the initial guess (if there is no 
                                trend fitting) or the most recent delta solution (if there is trend 
                                fitting). Only used if resume = True. Array shape should be (T,NV+4k,3), 
                                where NV is the number of angular frequency bins *not* removed by edge 
                                effects and T is the number of tau bins (i.e. the length of fracLOS). 
                                The three rows along the 2nd axis divide the population inversions by 
                                transition, with lastdelta[:,:,0] = delta^-, lastdelta[:,:,1] = delta^0, 
                                and lastdelta[:,:,2] = delta^+. Saved as object attribute, lastdelta.
                                
                                
                                --- Parameters that Probably Don't Need Adjusting (in most cases) ---
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process. 
                                Saved as object attribute, verbose.
            ftol            Float
                                [ Default = 6e-10 ]
                                The tolerance in minimizing the delta residuals used to determine 
                                convergence on a new solution. Passed directly to 
                                scipy.optimize.newton_krylov as its parameter, f_tol. Saved as object 
                                attribute, ftol.
            filename        String
                                [ Default = 'FaradayOut' ]
                                Beginning of the file name used to save the output inversion solutions.
                                For each total optical depth, tauf, the inversion solutions will be
                                saved to a file named '[filename]_tauf[tauf]_dminus.[fits/txt]' in the 
                                directory specified by outpath.
                                Note: previous versions of this code called tauf 'beta'. To provide
                                backwards compatibility, when reading files, this code will look first
                                for files with the new 'tauf' naming convention, but will look for 
                                any with the old 'beta' naming convention if those are not found.
            endfill         'fit' or 'zero'
                                [ Default = 'zero' ]
                                Determines how 2k angular frequency bins on either end will be filled 
                                in the inversion calculation. If endfill='fit', a 4th order polynomial 
                                as a function of frequency will be fit to the output delta arrays at 
                                each optical depth, and the end values will be filled in according to 
                                their frequency. If endfill='zero', the end points will just be set to 
                                zero. Saved as object attribute, endfill.
            trend           Boolean or 'auto'
                                [ Default = False ]
                                Keyword indicating whether to use one previous fitted delta (lastdelta) 
                                as the initial guess for the next iteration (if trend = False), or use 
                                the two fitted previous deltas (lastdelta and lastdelta2) to generate an 
                                initial guess for the next iteration (if trend = True). For the latter, 
                                it uses lastdelta + ( lastdelta2 - lastdelta )/2 as the initial guess. 
                                Finally, if trend = 'auto', calculates the residual using both lastdelta 
                                and lastdelta + ( lastdelta2 - lastdelta )/2 before starting the zero-
                                finding using the one with the lower residual as the initial guess. Saved 
                                as object attribute, trend.
                                Note: Unclear that trend fitting actually saves computation time. Not 
                                recommended.
            lastdelta2      NumPy Array
                                [ Default = None ]
                                Only used if resume = True AND trend = 'auto' or True.
                                The second most recent delta solution. See optional parameter trend above 
                                for details on usage. If specified, array shape should be (T,NV+4k,3), 
                                where NV is the number of angular frequency bins *not* removed by edge 
                                effects and T is the number of tau bins (i.e. the length of fracLOS). The 
                                three rows along the 2nd axis divide the population inversions by 
                                transition, with lastdelta2[:,:,0] = delta^-,  lastdelta2[:,:,1] = delta^0, 
                                and lastdelta2[:,:,2] = delta^+. Saved as object attribute, lastdelta2.
                                
        Additional Object Attributes Set:
            
            fccalc          Boolean
                                Whether far_coeff attribute was set explicitly (False) or calculated by 
                                calc_far_coeff. As object initialization only saves provided far_coeff,
                                sets this to False. If calc_far_coeff is later run, that method will 
                                update fccalc attribute.
            sintheta        Float
                                sin(theta). Saved for easy use in calculations.
            costheta        Float
                                sin(theta). Saved for easy use in calculations.
            sintwophi       Float
                                sin(2*phi). Saved for easy use in calculations.
            costwophi       Float
                                cos(2*phi). Saved for easy use in calculations.
            tau             NumPy Array
                                Array of fracLOS * tauf for a given total optical depth tauf. During object
                                initialization, initializes this attribute by assuming tauf=1, so tau
                                = fracLOS.
            dtau            Float
                                Bin width for tau attribute.
        
            
        """
        
        
        #### Uses _maser_base_ initialization to load parfile (if any) and the following attributes:
        ####     phi, n, outpath, far_coeff, etapm, alphapm, cloud, taufs, resume, lastdelta, verbose, ftol, filename,
        ####     endfill, trend, lastdelta2
        ####     + fccalc, sintwophi, costwophi
        #### Saves config file as attribute conf, name of parfile as attribute parfile, and name of base section in
        ####     config file as attribute sect
        super().__init__( parfile = parfile, ignore = [], \
                    omegabar = omegabar, fracLOS = fracLOS, theta = theta, iquv0 = iquv0, W = W, k = k, \
                    phi = phi, n = n, outpath = outpath, \
                    far_coeff = far_coeff, etapm = etapm, alphapm = alphapm, \
                    cloud = cloud, iquvF = iquvF, \
                    taufs = taufs, resume = resume, lastdelta = lastdelta, \
                    verbose = verbose, ftol = ftol, filename = filename, \
                    endfill = endfill, trend = trend, lastdelta2 = lastdelta2)
        
        #### Some extra work on taufs attribute, setting tauf, tau, and dtau
        
        # If taufs is an array, sets tauf, tau, and dtau attributes based on first value in array
        if isinstance( self.taufs, np.ndarray ):
            self.update_tauf( self.taufs[0] )
        
        # If taufs is None, sets tauf, tau, and dtau attributes
        elif self.taufs is None:
            self.tauf = None
            self.tau  = None
            self.dtau = None
    
    
    ### Main functions for performing calculations ###
    
    def calc_far_coeff(self, ne, freq0, Gam, B, A0, P0, mode='cm' ):
        """
        Calculates the faraday coefficient, gamma_QU/cos(theta) given:
            ne      = electron density [ cm^-3 or m^-3 ]
            freq0   = rest frequency of the line [ Hz ]
            Gamma     = loss rate [ s^-1 ]
            B       = magnetic field strength [Gauss]
            A0      = Einstein A coefficient [ s^-1 ]
            P0      = pump rate into the 0 state [cm^-3 s^-1 or m^-3 s^-1 ]
        Also uses the Doppler width in Hz given in object initialization.
        
        Keyword mode can be set to 'cm' or 'm' to specify units of the given ne and P0 values.
        Default is 'cm'. If set to 'cm', these values will be converted to SI prior to calculation.
        
        Overwrites self.far_coeff.
        """
        # Uses _maser_base_ class method, but hands it the doppler width, W, from the attribute
        super().calc_far_coeff( ne, freq0, Gam, B, A0, P0, self.W, mode=mode )
        
    def run( self, maxiter=100, sim_desc = True ):
        """
        Program to run solver.
        
        Optional parameter maxiter is the maximum number of iterations that the newton_krylov
        solver will perform; if that's exceeded, it will stop the program. Default is 100.
        """
        
        # Determines output path
        cwd = os.getcwd()
        path = '{0}/{1}'.format( cwd, self.outpath )
        
        # If the output path doesn't exist already, it creates it
        if not os.path.exists(path):
            os.makedirs(path)
        print('PRISM.MASER:  Output Path', path )
        
        # Writes sim description file if requested
        if sim_desc:
            self.write_desc(path)
        
        # If checkfile should be written out, begins array of output gamma values to print
        check_out = np.array([])
        
        # Begins iteration across tauf values for solving
        for b in range( self.taufs.size ):
            
            # If this is the first iteration, creates an initial guess array
            if b == 0:
                
                # If this is not resuming a previous run, that guess is an array of ones
                if not self.resume:
                    lastdelta = np.ones(( self.fracLOS.size, self.omegabar.size, 3 )).astype(np.longdouble)
                
                # Otherwise, sets that initial guess as the input resume array
                else:
                    lastdelta = self.lastdelta.astype(np.longdouble)
                    if self.trend == 'auto' or self.trend == True:
                        lastdelta2 = self.lastdelta2.astype(np.longdouble)
            
            # Sets tauf, tau, and dtau attributes for current tauf value
            self.update_tauf(  self.taufs[ b ] )
            
            
            # Prints feedback if requested
            if self.verbose  == True:
                print('PRISM.MASER: Beginning iteration {0} with tauf={1}...'.format(b, self.tauf ))
            
            # If the trend fitting is 'auto', calculates residual for both offsets if possible
            if self.trend == 'auto':
                print('         Auto Trend Fitting: Calculating residuals...')
                
                if b > 1 or self.resume:
                    # First calculates the residual with no trend fitting:
                    resid0 = self.inversion_resid( lastdelta )
                
                    # Then calculates the residual with trendfitting
                    trenddelta1 = lastdelta + 0.5*(lastdelta - lastdelta2)
                    resid1 = self.inversion_resid( trenddelta1 )
                    trenddelta2 = lastdelta + (lastdelta - lastdelta2)
                    resid2 = self.inversion_resid( trenddelta2 )
                    trenddelta3 = lastdelta + 1.5*(lastdelta - lastdelta2)
                    resid3 = self.inversion_resid( trenddelta3 )
                    trenddelta4 = lastdelta + 2.*(lastdelta - lastdelta2)
                    resid4 = self.inversion_resid( trenddelta4 )
                    
                    # Makes lists for feedback
                    opt = ['NO', '0.50 x', '1 x', '1.5 x', '2 x']
                    maxes = np.array([ np.abs(resid0).max(), np.abs(resid1).max(), np.abs(resid2).max(), np.abs(resid3).max(), np.abs(resid4).max() ])
                    trendlist = [lastdelta, trenddelta1, trenddelta2, trenddelta3, trenddelta4]
                    
                    # Determines which has a smaller residual 
                    imin = maxes.argmin()
                    print('         Auto Trend Fitting: Smaller Initial Residual with {0} Trend Fitting.'.format(opt[imin]))
                    initdelta = trendlist[imin]
                    
                
                # If two taufs haven't been iterated through yet, no trend to fit
                else:
                    print('         Auto Trend Fitting: No trend to fit yet.')
                    initdelta = lastdelta
            
            # Otherwise, if trend fitting is on, just sets last delta to be the trend
            elif self.trend == True:
                print('         Trend Fitting ON.')
                initdelta = lastdelta + 0.5*(lastdelta - lastdelta2)
            
            # IF trend fitting is off, sets initial delta to just be last delta
            else:
                print('         Trend Fitting OFF.')
                initdelta = lastdelta
            
            # Root finding! Finds what values of self.detlas (the dimensionless inversions) 
            #     constitute a solution
            try:
                #print('PRISM.MASER: Input delta data type ',end=" ")
                #print(self.deltas.dtype)
                deltas_new = optimize.newton_krylov( self.inversion_resid, initdelta, maxiter=maxiter, \
                                                          f_tol = self.ftol, verbose=self.verbose )
            
            # If solver converged on a non-solution, prints an error and shunts last delta to a
            #     text file before breaking
            except Exception as e:
                print('ERROR: Solution not found in {0} iterations or otherwise broken.'.format( maxiter ))
                print(e)
                
                # Saves resulting deltas with write_deltas method and breaks
                # self.write_deltas(self.tauf, deltas_new, ext='fits', broken=True)
                break

            
            # Prints feedback if requested
            if self.verbose == True:
                print('PRISM.MASER: Iteration {0} with tauf={1} Complete.'.format(b, self.tauf))
                print('PRISM.MASER: Output data type', deltas_new.dtype)
                print('PRISM.MASER: Writing output to file...')
            
            # Saves resulting deltas with write_deltas method
            self.write_deltas(self.tauf, deltas_new, ext='fits')
            
            # New deltas become old for the next iteration
            if self.trend == True or self.trend == 'auto':
                lastdelta2 = lastdelta.copy()
    
            lastdelta = deltas_new.copy()
            self.deltas = lastdelta
            del deltas_new
            
    def update_tauf( self, tauf ):
        """
        Updates tauf value (i.e. the total optical depth of the cloud multiplied by fracLOS).
        
        Updates object attributes self.tauf, self.tau, and self.dtau.
        """
        
        # Saves new tauf value as object attribute tauf
        self.tauf = tauf
        
        # Scales tau array appropriately
        self.tau = self.fracLOS * self.tauf
    
        # Determines the spacing in tau
        self.dtau = self.tau[1] - self.tau[0]
    
    
    ### Functions for analysis ###
            
    def readin(self, tauf, ext='fits', updatepars = False ): 
        """
        Program to read in files generated by iterative root finding in __init__ function.
        
        Reads in the -, 0, and + delta arrays for the specified tauf from the output directory.
        
        Returns array with shape (T,NV+4k,3) array of delta values. Zeroth axis separates by optical
        depth, tau, first axis separates by frequency, and second axis separates by transition for
        -, 0, and +, resp.
        
        
        Note: previous versions of this code called tauf 'beta'. To provide backwards compatibility, 
        when reading files, this code will look first for files with the new 'tauf' naming convention, 
        but will look for any with the old 'beta' naming convention if those are not found.
        """
        
        # Makes sure . not provided in requested extension
        if ext.startswith('.'):
            ext = ext[1:]
        
        # Reading in if text file
        if ext == 'txt':
        
            # Determines path names for each delta using desired extension
            dminus_path = '{0}{1}_tauf{2}_dminus.{3}'.format(self.outpath, self.filename, tauf, ext )
            dzero_path  = '{0}{1}_tauf{2}_dzero.{3}'.format(self.outpath, self.filename, tauf, ext )
            dplus_path  = '{0}{1}_tauf{2}_dplus.{3}'.format(self.outpath, self.filename, tauf, ext )
        
            # Reads in minus file, if it exists with new tauf naming convention
            if os.path.exists( dminus_path ):
                dminus = np.genfromtxt( dminus_path )
            else:
                dminus_path = dminus_path.replace( 'tauf', 'beta' )
                if os.path.exists( dminus_path ):
                    dminus = np.genfromtxt( dminus_path )
                
            # Reads in zero file, if it exists with new tauf naming convention
            if os.path.exists( dzero_path ):
                dzero = np.genfromtxt( dzero_path )
            else:
                dzero_path = dzero_path.replace( 'tauf', 'beta' )
                if os.path.exists( dzero_path ):
                    dzero = np.genfromtxt( dzero_path )
                
            # Reads in plus file, if it exists with new tauf naming convention
            if os.path.exists( dplus_path ):
                dplus = np.genfromtxt( dplus_path )
            else:
                dplus_path = dplus_path.replace( 'tauf', 'beta' )
                if os.path.exists( dplus_path ):
                    dplus = np.genfromtxt( dplus_path )
        
        # Reading in if fits file
        elif ext == 'fits':
        
            # Determines path names for single fits file
            outpath = '{0}{1}_tauf{2}.{3}'.format(self.outpath, self.filename, tauf, ext )
            if not os.path.exists( outpath ):
                outpath = outpath.replace( 'tauf','beta' )
            
            # Opens fits file for reading
            hdu = fits.open( outpath )
            
            # Gets delta arrays from extensions
            dminus = hdu[1].data
            dzero  = hdu[2].data
            dplus  = hdu[3].data
            
            # Updates object attributes from header if updatepars requested
            # Does not overwrite outpath, verbose, resume, trend, lastdelta, or lastdelta2.
            if updatepars:
                
                # Sets aside 0-ext header for easy ref
                hdr = hdu[0].header
                
                # Reconstruct omegabar array from header keys. Assumes omegabar is centered on omega_0
                AFbins = hdr['AFbins']
                dAF = hdr['AFres']
                Nplus = ( AFbins - 1 ) / 2
                self.omegabar = np.linspace( -Nplus, Nplus, AFbins ).astype(np.longdouble) * dAF
                
                # Reconstructs fracLOS assuming fracLOS is fraction of cloud transversed from 0 to 1
                self.fracLOS = np.linspace( 0, 1, hdr['taubins'] ).astype(np.longdouble)
                
                # Retrieves theta
                self.theta = hdr['theta']
                
                # Reconstructs iquv0
                self.iquv0 = np.array([ hdr['i0'], hdr['q0'], hdr['u0'], hdr['v0'] ])
                
                # Sets W, k, phi, and farcoeff directly
                self.W = hdr['Doppler']
                self.k = hdr['k']
                self.phi = hdr['phi']
                self.far_coeff = hdr['farcoeff']
                
                # Sets eta p/m and alpha p/m
                self.etap = hdr['etap']
                self.etam = hdr['etam']
                self.alphap = hdr['alphap']
                self.alpham = hdr['alpham']
                
                # Saves n and endfill
                self.n = hdr['nexp']
                self.endfill = hdr['endfill']
                
                # Sets cloud and ray2 stuff
                self.cloud = hdr['cloud']
                if self.cloud == 2:
                    self.iquvF = np.array([ hdr['iF'], hdr['qF'], hdr['uF'], hdr['vF'] ])
                else:
                    self.iquvF = None
                
                # Saves tauf and ftol
                self.ftol = hdr['ftol']
                
                # Gets fcalc info if in the header
                if 'ne' in hdr.keys():
                    self.fccalc = True
                    self.ne = hdr['ne']
                    self.P0 = hdr['P0']
                    self.freq0 = hdr['AF0']
                    self.Gamma = hdr['Gamma']
                    self.B = hdr['B']
                    self.A0 = hdr['A0']
                    
                # Otherwise, assumes fccoeff set manually
                else:
                    self.fccalc = False
                
                # Updates sin and cos
                self.sintheta = sin(self.theta)
                self.costheta = cos(self.theta)
                self.sintwophi = sin(2.*self.phi)
                self.costwophi = cos(2.*self.phi)
        
                # Updates tauf
                self.update_tauf( float(tauf) )
            
            # Closes fits file
            hdu.close()
        
        return np.dstack(( dminus, dzero, dplus ))
        
    def stokes(self, verbose = False ):
        """
        Program that calculates the dimensionless stokes values, fractional polarizations, and EVPA
        for a given inversion solution.
        
        Prior to calling, the following attributes should be set/up to date:
            
            deltas          NumPy Array of shape (T, NV, 3)
                                The dimensionless inversion solution for the given parameter set.
                                Can be read in from file. T is the number of bins along the line of
                                sight, NV is the number of bins in omegabar. Final axis indicates
                                the transition, separated as delta^-, delta^0, delta^+.
            
            dtau            Float
                                The bin width along optical depth. Depends on both resolution along
                                line of sight and the total optical depth, tauf. Updated by method 
                                update_tauf. Is *not* modified by updatepars option in readin method.
            
            far_coeff       Float
                                Unitless value gives -gamma_QU / cos(theta). Can either be specified 
                                explicitly on object initialization, or calculated from components 
                                using the calc_far_coeff method. *IS* updated if the updatepars
                                option is used when readin in a deltas solution from a fits file.
                                
        
        Other object attributes used by this method that are set on object initialization:
        
            theta, costheta, sintheta, phi, costwophi, sintwophi, etap, etam, fracLOS, cloud, iquv0, 
            iquvF, k
            
        Optional Parameters:
            
            verbose         Boolean
                                [ Default = False ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Object Attributes Created/Updated by This Method:
            
            stoki           NumPy array of shape (T,NV-2k)
                                Unitless Stokes i corresponding to the deltas solution.
            
            stokq           NumPy array of shape (T,NV-2k)
                                Unitless Stokes q corresponding to the deltas solution.
            
            stoku           NumPy array of shape (T,NV-2k)
                                Unitless Stokes u corresponding to the deltas solution.
            
            stokv           NumPy array of shape (T,NV-2k)
                                Unitless Stokes v corresponding to the deltas solution.
            
            mc              NumPy array of shape (T,NV-2k)
                                Fractional circular polarization for the deltas solution. Calculated
                                as the ratio Stokes v / i. Does preserve sign of circular polarization.
            
            ml              NumPy array of shape (T,NV-2k)
                                Fractional linear polarization for the deltas solution. Does not 
                                preserve the direction of linear polarization.
            
            evpa            NumPy array of shape (T,NV-2k)
                                Electric vector position angle of the linear polarization at each
                                point in the solution grid. Calculated as 0.5 * arctan2( u, q ).
                                
            
        """
        
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        delta_m = self.deltas[:,:,0]
        delta_0 = self.deltas[:,:,1]
        delta_p = self.deltas[:,:,2]
        
        # Then separates out end stokes values
        i0, q0, u0, v0 = self.iquv0
        if self.cloud == 2:
            iF, qF, uF, vF = self.iquvF
        
        # simplify k name
        k = self.k
        
        if verbose:
            # Sets appropriate print options
            np.set_printoptions(precision=4, linewidth=180)
            print('STOKES TEST:')
            print('    tau min:', tau[0], '   tau max:', tau[-1] )
            print('    dtau: ', dtau )
        
        
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
        
        
            # Prints test output if verbose
            if verbose:
                print('    gamma_I: ', gamma_I[:,verbose+k])
                
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            GI = np.zeros( gamma_I[0].shape )
            GQ = np.zeros( gamma_I[0].shape )
            GU = np.zeros( gamma_I[0].shape )
            GV = np.zeros( gamma_I[0].shape )
            GQU= np.zeros( gamma_I[0].shape )
            
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i]
                GIi  = 0.5*gamma_I[0,:]  + gamma_I[1:i,:].sum(axis=0)  + 0.5*gamma_I[i,:]
                GQi  = 0.5*gamma_Q[0,:]  + gamma_Q[1:i,:].sum(axis=0)  + 0.5*gamma_Q[i,:]
                GUi  = 0.5*gamma_U[0,:]  + gamma_U[1:i,:].sum(axis=0)  + 0.5*gamma_U[i,:]
                GVi  = 0.5*gamma_V[0,:]  + gamma_V[1:i,:].sum(axis=0)  + 0.5*gamma_V[i,:]
                GQUi = 0.5*gamma_QU[0,:] + gamma_QU[1:i,:].sum(axis=0) + 0.5*gamma_QU[i,:]
                
                GI = np.vstack(( GI, GIi ))
                GQ = np.vstack(( GQ, GQi ))
                GU = np.vstack(( GU, GUi ))
                GV = np.vstack(( GV, GVi ))
                GQU = np.vstack(( GQU, GQUi ))
                
                # Prints test output if verbose
            if verbose: # i == tau.size-1 and 
                print()
                print('    gamma_I sum: ', GI[:,verbose+k])
             
                
            # These are only sums. Multiplies these GX arrays by self.dtau before continuing
            GI = GI*self.dtau
            GQ = GQ*self.dtau
            GU = GU*self.dtau
            GV = GV*self.dtau
            GQU = GQU*self.dtau
            
            # Deletes gamma arrays and GSi now that we're done with them
            del gamma_I, gamma_Q, gamma_U, gamma_V, gamma_QU, A
            del GIi, GQi, GUi, GVi, GQUi
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    GAMMA_I: ', GI[:,verbose+k])
            
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            scratch1 = np.ones( GI.shape )
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes = stokes0.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            scratch0 = np.zeros( GI.shape )
            Grow0 = np.stack(( GI,      -GQ,      -GU,      -GV ))
            Grow1 = np.stack((-GQ,       GI,      GQU, scratch0 ))
            Grow2 = np.stack((-GU,     -GQU,       GI, scratch0 ))
            Grow3 = np.stack((-GV, scratch0, scratch0,       GI ))
            Garray = np.stack(( Grow0, Grow1, Grow2, Grow3 ))
            
            # Deletes integrated GSTokes sub-arrays and Grows since we're done with them now
            del GI, GQ, GU, GV, GQU, scratch1, scratch0
            del Grow0, Grow1, Grow2, Grow3
            
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    Gproduct = Garray.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    Gproduct = np.matmul( Gproduct, Garray, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term_n = np.matmul( Gproduct, self.iquv0, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes += ( term_n / float(factorial(n)) )
                else:
                    tempterm = ( term_n / float(factorial(170)) )
                    stokes  += tempterm * float( factorial(170)/factorial(n) )
            
            # Deletes Gproduct and Garray, now that we're done with the stokes calculation
            del Garray, Gproduct, term_n
            if self.n > 170:
                del tempterm
            
            
            # After calculation, unpacks stokes
            stoki, stokq, stoku, stokv = stokes
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    StokesI: ', stoki[:,verbose+k])
            
             
            
            
            
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
            # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            #print('Preparing for gain matrix integration')
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            # 1 is the direction from tau[0] to tau[i] and 2 is the direction from tau[T-1] to tau[i]
            # Iterating through ray 2 backwards, so this is the last column
            GI1 = np.zeros( gamma_I[0].shape )
            GQ1 = np.zeros( gamma_I[0].shape )
            GU1 = np.zeros( gamma_I[0].shape )
            GV1 = np.zeros( gamma_I[0].shape )
            GQU1= np.zeros( gamma_I[0].shape )
            GI2 = np.zeros( gamma_I[0].shape )
            GQ2 = np.zeros( gamma_I[0].shape )
            GU2 = np.zeros( gamma_I[0].shape )
            GV2 = np.zeros( gamma_I[0].shape )
            GQU2= np.zeros( gamma_I[0].shape )
            #print('Integrating gain matrix')
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i] for ray 1
                GI1i  = 0.5*gamma_I[0,:]   + gamma_I[1:i,:].sum(axis=0)   + 0.5*gamma_I[i,:]
                GQ1i  = 0.5*gamma_Q[0,:]   + gamma_Q[1:i,:].sum(axis=0)   + 0.5*gamma_Q[i,:]
                GU1i  = 0.5*gamma_U1[0,:]  + gamma_U1[1:i,:].sum(axis=0)  + 0.5*gamma_U1[i,:]
                GV1i  = 0.5*gamma_V1[0,:]  + gamma_V1[1:i,:].sum(axis=0)  + 0.5*gamma_V1[i,:]
                GQU1i = 0.5*gamma_QU1[0,:] + gamma_QU1[1:i,:].sum(axis=0) + 0.5*gamma_QU1[i,:]
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[i] and tau[T-1] for ray 2
                # This is being calculated backwards, so this is column -(i+1) for ray 2 
                GI2i  = 0.5*gamma_I[-1,:]   + gamma_I[-i:-1,:].sum(axis=0)  + 0.5*gamma_I[-i-1,:]
                GQ2i  = 0.5*gamma_Q[-1,:]   + gamma_Q[-i:-1,:].sum(axis=0)  + 0.5*gamma_Q[-i-1,:]
                GU2i  = 0.5*gamma_U2[-1,:]  + gamma_U2[-i:-1,:].sum(axis=0) + 0.5*gamma_U2[-i-1,:]           
                GV2i  = 0.5*gamma_V2[-1,:]  + gamma_V2[-i:-1,:].sum(axis=0) + 0.5*gamma_V2[-i-1,:]         
                GQU2i = 0.5*gamma_QU2[-1,:] + gamma_QU2[-i:-1,:].sum(axis=0)+ 0.5*gamma_QU2[-i-1,:]   
                
                # Adds the ith row onto the Gamma arrays for ray 1
                GI1 = np.vstack(( GI1, GI1i ))
                GQ1 = np.vstack(( GQ1, GQ1i ))
                GU1 = np.vstack(( GU1, GU1i ))
                GV1 = np.vstack(( GV1, GV1i ))
                GQU1 = np.vstack(( GQU1, GQU1i ))
                
                # Adds the ith row onto the Gamma arrays for ray q; technically the -(i+1) index for ray 2
                #   so does a reverse stack
                GI2 = np.vstack(( GI2i, GI2 ))
                GQ2 = np.vstack(( GQ2i, GQ2 ))
                GU2 = np.vstack(( GU2i, GU2 ))
                GV2 = np.vstack(( GV2i, GV2 ))
                GQU2 = np.vstack(( GQU2i, GQU2 ))
            
            # Prints test output if verbose
            if verbose:  # i == tau.size-1 and 
                print()
                print('    gamma_I1 sum: ', GI1[:,verbose+k])
                print('    gamma_I2 sum: ', GI2[:,verbose+k])
            
            # These are only sums. Multiplies these GX1 & GX2 arrays by self.dtau 
            GI1 = GI1*self.dtau
            GQ1 = GQ1*self.dtau
            GU1 = GU1*self.dtau
            GV1 = GV1*self.dtau
            GQU1 = GQU1*self.dtau
            GI2 = GI2*self.dtau
            GQ2 = GQ2*self.dtau
            GU2 = GU2*self.dtau
            GV2 = GV2*self.dtau
            GQU2 = GQU2*self.dtau
            
            # Deletes gamma arrays and GSi now that we're done with them
            del gamma_I, gamma_Q, gamma_U1, gamma_V1, gamma_QU1, gamma_U2, gamma_V2, gamma_QU2, A
            del GI1i, GQ1i, GU1i, GV1i, GQU1i, GI2i, GQ2i, GU2i, GV2i, GQU2i
            
            
            # Prints test output if verbose
            if verbose:
                print()
                print('    GAMMA_I1: ', GI1[:,verbose+k])
                print('    GAMMA_I2: ', GI2[:,verbose+k])
                        
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            
            # Creates some scratch arrays that can be used for both directions
            scratch1 = np.ones( GI1.shape )
            scratch0 = np.zeros( GI1.shape )
            
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            # Starts with iquv0 for direction 1
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes1 = stokes0.copy()
            # Then does iquvF for direction 2
            stokesF = np.stack((iF*scratch1, qF*scratch1, uF*scratch1, vF*scratch1))
            stokes2 = stokesF.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            # Does this first for Direction 1:
            G1row0 = np.stack(( GI1,     -GQ1,     -GU1,     -GV1 ))
            G1row1 = np.stack((-GQ1,      GI1,     GQU1, scratch0 ))
            G1row2 = np.stack((-GU1,    -GQU1,      GI1, scratch0 ))
            G1row3 = np.stack((-GV1, scratch0, scratch0,      GI1 ))
            G1array = np.stack(( G1row0, G1row1, G1row2, G1row3 ))
            # Then does the same for direction 2:
            G2row0 = np.stack(( GI2,     -GQ2,     -GU2,     -GV2 ))
            G2row1 = np.stack((-GQ2,      GI2,     GQU2, scratch0 ))
            G2row2 = np.stack((-GU2,    -GQU2,      GI2, scratch0 ))
            G2row3 = np.stack((-GV2, scratch0, scratch0,      GI2 ))
            G2array = np.stack(( G2row0, G2row1, G2row2, G2row3 ))
            
            # Deletes integrated GSTokes sub-arrays and Grows since we're done with them now
            del GI1, GQ1, GU1, GV1, GQU1, GI2, GQ2, GU2, GV2, GQU2, scratch1, scratch0
            del G1row0, G1row1, G1row2, G1row3, G2row0, G2row1, G2row2, G2row3
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    G1product = G1array.copy()
                    G2product = G2array.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    G1product = np.matmul( G1product, G1array, axes=[(0,1),(0,1),(0,1)] )
                    G2product = np.matmul( G2product, G2array, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term1_n = np.matmul( G1product, self.iquv0, axes=[(0,1),(0,),(0,)])
                term2_n = np.matmul( G2product, self.iquvF, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes1 += ( term1_n / float(factorial(n)) )
                    stokes2 += ( term2_n / float(factorial(n)) )
                else:
                    tempterm1 = term1_n / float(factorial(170))
                    tempterm2 = term2_n / float(factorial(170))
                    stokes1 += tempterm1 * float( factorial(170)/factorial(n) )
                    stokes2 += tempterm2 * float( factorial(170)/factorial(n) )
            
            # Deletes Gproduct and Garray, now that we're done with the stokes calculation
            del G1array, G2array, G1product, G2product, term1_n, term2_n
            if self.n > 170:
                del tempterm1, tempterm2
            
            # After calculation, unpacks stokes
            stoki1, stokq1, stoku1, stokv1 = stokes1
            stoki2, stokq2, stoku2, stokv2 = stokes2
            
            # Finally, calculates the average of the stokes values from both arrays at each
            #     frequency and optical depth
            # UPDATED: Summing instead of averaging. As of bw10, Ray 2 no longer being flipped in velocity/frequency.
            #     Stokes u and v still being flipped in orientation.
            stoki = ( stoki1 + stoki2 )
            stokq = ( stokq1 + stokq2 )
            stoku = ( stoku1 - stoku2 )
            stokv = ( stokv1 - stokv2 )
            
        # Stacks output arrays and returns
        # stokes_out = np.dstack(( stoki, stokq, stoku, stokv ))
        # return stokes_out
        
        # In lieu of returning stokes arrays, sets arrays as object attributes
        self.stoki = stoki
        self.stokq = stokq
        self.stoku = stoku
        self.stokv = stokv
        
        # Calculates fractional circular and linear pol, and pol angle
        self.mc = self.stokv / self.stoki
        self.ml = np.sqrt( self.stokq**2 + self.stoku**2 ) / self.stoki
        self.evpa = 0.5 * np.arctan2( self.stoku, self.stokq )
        
    def cloud_end_stokes(self, taufs, ext='fits', tau_idx = -1, saveas = None, overwrite = False, verbose = True ):
        """
        Calculates the dimensionless stokes values, fractional polarizations, and EVPA at the end of
        the cloud for a variety of total optical depths, tauf.
        
        Prior to calling, the following attributes should be set/up to date:
            
            far_coeff       Float
                                Unitless value gives -gamma_QU / cos(theta). Can either be specified 
                                explicitly on object initialization, or calculated from components 
                                using the calc_far_coeff method. *IS* updated if the updatepars
                                option is used when readin in a deltas solution from a fits file.
                                
        
        Other object attributes used by this method that are set on object initialization:
        
            theta, costheta, sintheta, phi, costwophi, sintwophi, etap, etam, fracLOS, cloud, iquv0, 
            iquvF, k, filename, outpath
            
        Required Parameters:
            
            taufs           1D NumPy array
                                The values of total optical depth, tauf, for which the stokes
                                values are desired. Solution files should already exist for all
                                specified values in the outpath, of file extension indicated by
                                ext. (Of size B, for comparison with array attributes set below.)
        
        Optional Parameters:
            
            ext             String ('fits' or 'txt')
                                [ Default = 'fits' ]
                                The file extension of the output inversion solutions in the 
                                outpath. All tauf solutions should use the same file extension.
            
            tau_idx         Integer
                                [ Default = -1 ]
                                The index within the cloud at which the solution will be retained. 
                                To use the solution at the far end of the cloud (i.e. where the
                                ray exits the cloud in a single ray solution), use tau_idx = -1. 
                                To use the solution at the near end of the cloud (i.e. where the
                                ray enters the cloud in a single ray solution), use tau_idx = 0.
                                Latter should only matter if a 2-ray solution with different 
                                rays.
            
            saveas          String or None
                                [ Default = None ]
                                If provided as string, will save the produced stokes and polarization
                                arrays to files in the outpath, with the file name given in the string.
                                Results will be saved as a fits file. String name given by saveas does
                                not need to end in a '.fits' suffix, though it can.
                                The resulting fits file will have 8 extensions - the 0th extension
                                contains a header with basic information on the results, the tau bin,
                                and the values of tauf, while the remaining 7 extensions contain the 
                                data arrays for stokes i, stokes q, stokes u, stokes v, mc, ml, and
                                evpa, respectively.
            
            overwrite       Boolean
                                [ Default = False ]
                                Whether to overwrite any existing fits file with the same name when 
                                creating the output (True) or not (False). Used only if saveas is not
                                None.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Object Attributes Created/Updated by This Method:
            
            tau_idx         Integer
                                Saved directly from user input.
                                
            stacked_stoki   NumPy array of shape (B,NV-2k)
                                Unitless Stokes i at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stokq   NumPy array of shape (B,NV-2k)
                                Unitless Stokes q at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stoku   NumPy array of shape (B,NV-2k)
                                Unitless Stokes u at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stokv   NumPy array of shape (B,NV-2k)
                                Unitless Stokes v at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_mc      NumPy array of shape (B,NV-2k)
                                Fractional circular polarization at the end of a cloud at each 
                                optical depth as a function of frequency. Calculated as the ratio 
                                Stokes v / i. Does preserve sign of circular polarization.
            
            stacked_ml      NumPy array of shape (B,NV-2k)
                                Fractional linear polarization at the end of a cloud at each optical 
                                depth as a function of frequency. Does not preserve the direction of 
                                linear polarization.
            
            stacked_evpa    NumPy array of shape (B,NV-2k)
                                Electric vector position angle of the linear polarization at each
                                point in the solution grid. Calculated as 0.5 * arctan2( u, q ).
                                
        """
        
        # Overrites taufs object with provided attribute
        self.taufs = np.array( taufs ).astype(float)
        
        # Saves tau_idx as attribute
        self.tau_idx = tau_idx
        
        # Initializes output objects
        self.stacked_stoki = None
        self.stacked_stokq = None
        self.stacked_stoku = None
        self.stacked_stokv = None
        self.stacked_mc    = None
        self.stacked_ml    = None
        self.stacked_evpa  = None
        
        # Begins iterating through tauf values
        for i, tauf in enumerate(taufs):
            
            # Prints output if verbose
            if verbose:
                print('  -- Loading tauf = {0}...'.format(tauf))
            
            # Reads file to get deltas array
            self.deltas = self.readin( str( round( tauf, 1 )), ext=ext )
            
            # Updates dtau and other values associated with tauf
            self.update_tauf( float(tauf) )
            
            # Calculates stokes, mc, ml, and evpa for the deltas solution
            self.stokes( verbose = False )
            
            # Extracts desired row from stokes calculation for this tauf and adds to stacked array
            if i != 0:
                self.stacked_stoki = np.vstack(( self.stacked_stoki, self.stoki[tau_idx,:] ))
                self.stacked_stokq = np.vstack(( self.stacked_stokq, self.stokq[tau_idx,:] ))
                self.stacked_stoku = np.vstack(( self.stacked_stoku, self.stoku[tau_idx,:] ))
                self.stacked_stokv = np.vstack(( self.stacked_stokv, self.stokv[tau_idx,:] ))
                self.stacked_mc    = np.vstack(( self.stacked_mc   , self.mc[   tau_idx,:] ))
                self.stacked_ml    = np.vstack(( self.stacked_ml   , self.ml[   tau_idx,:] ))
                self.stacked_evpa  = np.vstack(( self.stacked_evpa , self.evpa[ tau_idx,:] ))
            else:
                self.stacked_stoki = np.array( self.stoki[tau_idx,:] )
                self.stacked_stokq = np.array( self.stokq[tau_idx,:] )
                self.stacked_stoku = np.array( self.stoku[tau_idx,:] )
                self.stacked_stokv = np.array( self.stokv[tau_idx,:] )
                self.stacked_mc    = np.array( self.mc[   tau_idx,:] )
                self.stacked_ml    = np.array( self.ml[   tau_idx,:] )
                self.stacked_evpa  = np.array( self.evpa[ tau_idx,:] )
            
            # Clears out the single-tau attributes
            del self.stoki, self.stokq, self.stoku, self.stokv, self.mc, self.ml, self.evpa
        
        # Saves, if requested
        if saveas is not None and isinstance( saveas, str ):
            
            # Makes path for file to save and makes sure the file name ends in .fits extension
            savepath = '{0}{1}'.format( self.outpath, saveas )
            if not savepath.lower().endswith('.fits'):
                savepath = '{0}.fits'.format(savepath)
            
            # Makes primary HDU with no data
            prime_hdu = fits.PrimaryHDU()
            
            # Populates primary header with info these stokes arrays
            prime_hdu.header['AFmin'] = ( self.omegabar[0+self.k], 'Angular Freq Min for Stokes Arrays [s^-1]' )
            prime_hdu.header['AFmax'] = ( self.omegabar[-1-self.k], 'Angular Freq Max for Stokes Arrays [s^-1]' )
            prime_hdu.header['AFres'] = ( self.omegabar[1]-self.omegabar[0], 'Angular Freq Resolution [s^-1]' )
            prime_hdu.header['AFbins'] = ( self.omegabar.size, 'Total Angular Freq Bins' )
            prime_hdu.header['AFdata'] = ( self.omegabar.size - 2*self.k, 'Angular Freq Bins for Stokes Data' )
            prime_hdu.header['k'] = ( self.k, 'Zeeman splitting [bins]' )
            prime_hdu.header['taures'] = ( self.fracLOS.size, 'Number of Tau Resolution Bins along LoS' )
            prime_hdu.header['tauidx'] = ( tau_idx, 'Index along tau of Enclosed Stokes' )
            
            # These header keys used to be 'betaN', 'betamin', and 'betamax', and may still be in old files
            prime_hdu.header['taufN'] = ( self.taufs.size, 'Number of Optical Depths' )
            prime_hdu.header['taufmin'] = ( self.taufs[0], 'Min of Optical Depths' )
            prime_hdu.header['taufmax'] = ( self.taufs[-1], 'Max of Optical Depths' )
            
            
            # Populates primary header with other info about calculation
            prime_hdu.header['cloud'] = ( self.cloud, 'number of rays' )
            prime_hdu.header['Doppler'] = ( self.W, 'Doppler width [s^-1]' )
            prime_hdu.header['Zeeman'] = ( (self.omegabar[1]-self.omegabar[0])*float(self.k), 'Zeeman splitting [s^-1]' )
            prime_hdu.header['theta'] = ( self.theta, 'B field angle [rad]' )
            prime_hdu.header['phi'] = ( self.phi, 'Sky angle [rad]' )
            prime_hdu.header['etap'] = ( self.etap, '|d^+|^2 / |d^0|^2' )
            prime_hdu.header['etam'] = ( self.etam, '|d^-|^2 / |d^0|^2' )
            prime_hdu.header['alphap'] = ( self.alphap, 'P^+ / P^0' )
            prime_hdu.header['alpham'] = ( self.alpham, 'P^- / P^0' )
            prime_hdu.header['i0'] = ( self.iquv0[0], 'Ray 1 initial Stokes i' )
            prime_hdu.header['q0'] = ( self.iquv0[1], 'Ray 1 initial Stokes q' )
            prime_hdu.header['u0'] = ( self.iquv0[2], 'Ray 1 initial Stokes u' )
            prime_hdu.header['v0'] = ( self.iquv0[3], 'Ray 1 initial Stokes v' )
            if self.cloud == 2:
                prime_hdu.header['iF'] = ( self.iquvF[0], 'Ray 2 initial Stokes i' )
                prime_hdu.header['qF'] = ( self.iquvF[1], 'Ray 2 initial Stokes q' )
                prime_hdu.header['uF'] = ( self.iquvF[2], 'Ray 2 initial Stokes u' )
                prime_hdu.header['vF'] = ( self.iquvF[3], 'Ray 2 initial Stokes v' )
            prime_hdu.header['endfill'] = ( self.endfill, 'Mode for handling freq edges' )
            prime_hdu.header['farcoeff'] = ( self.far_coeff, '-gamma_qu/cos(theta)' )
            prime_hdu.header['nexp'] = ( self.n, 'Number of Expansion Terms' )
            prime_hdu.header['ftol'] = ( self.ftol, 'Tolerance for convergence' )
            if self.fccalc:
                prime_hdu.header['ne'] = ( self.ne, 'Electron number density [m^-3]' )
                prime_hdu.header['AF0'] = ( self.freq0, 'Angular frequency at line center [s^-1]' )
                prime_hdu.header['Gamma'] = ( self.Gamma, 'Loss rate [s^-1]' )
                prime_hdu.header['B'] = ( self.B, 'Magnetic field strength [G]' )
                prime_hdu.header['A0'] = ( self.A0, 'Einstein A coeff [s^-1]' )
                prime_hdu.header['P0'] = ( self.P0, 'Pi Pump rate [m^-3 s^-1]' )
            
            # Saves array of taufs values as data of primary hdu
            prime_hdu.data = taufs
               
            # Makes HDU for each stokes array
            ext1 = fits.ImageHDU( self.stacked_stoki.astype( np.double ) )
            ext2 = fits.ImageHDU( self.stacked_stokq.astype( np.double ) )
            ext3 = fits.ImageHDU( self.stacked_stoku.astype( np.double ) )
            ext4 = fits.ImageHDU( self.stacked_stokv.astype( np.double ) )
            ext5 = fits.ImageHDU( self.stacked_mc.astype(    np.double ) )
            ext6 = fits.ImageHDU( self.stacked_ml.astype(    np.double ) )
            ext7 = fits.ImageHDU( self.stacked_evpa.astype(  np.double ) )
            
            # Makes HDU list with each hdu as an extension
            hdu = fits.HDUList([ prime_hdu, ext1, ext2, ext3, ext4, ext5, ext6, ext7 ])
            
            # Labeling each
            extnames = ['stokesi', 'stokesq', 'stokesu', 'stokesv', 'mc', 'ml', 'evpa' ]
            for i in range(1,8):
                hdu[0].header['EXT{0}'.format(i)] = (extnames[i-1], 'Data stored in ext {0}'.format(i))
                hdu[i].name = extnames[i-1]
            
            # Writes hdulist to file
            hdu.writeto( savepath, overwrite = overwrite )
            
            # Prints feedback if requested
            if self.verbose == True:
                print('PRISM.MASER: Stokes file {0} written.'.format( savepath ) )
            
    def read_cloud_end_stokes(self, filename, verbose = True ):
        """
        Reads in fits file created by method cloud_end_stokes using saveas option and updates object attributes
        accordingly. 
        
        Required Parameters:
            
            filename        String
                                Name of the fits file to which the calculated stokes values were saved by
                                cloud_end_stokes method. Assumed to be in path specified by object attribute
                                outpath.
        
        Optional Parameters:
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Updates Attributes:
            
            tau_idx
            taufs
            stacked_stokesi
            stacked_stokesq
            stacked_stokesu
            stacked_stokesv
            stacked_mc
            stacked_ml
            stacked_evpa
        
        
        Other Functionality:
            
            Checks other attributes stored in fits file header against object attributes, and raises a 
            warning if there is disagreement (but will still run).
        
        """
        
        # File assumed to be within the outpath
        filepath = '{0}{1}'.format( self.outpath, filename )
        if not os.path.exists(filepath):
            
            # If file is not in the outpath, checks from current directory
            if os.path.exists(filename):
                print('    Warning: File {0} not found in outpath {1}, but found in current directory.'.format(filename, self.outpath))
                print('             Using file in current directory.')
                filepath = filename
            
            # if file is not in outpath or current directory
            else:
                raise FileNotFoundError('Files {0} or {1} not found.'.format(filepath, filename))
        
        # Opens file in such a way that it will close automatically when over
        with fits.open(filepath) as hdu:
            
            
            
            #### Checks parameters from file against object attributes ####
            if verbose:
            
                # Makes warning string templates
                warning_temp_line = '    MASER.READ_CLOUD_END_STOKES WARNING:  Parameter {0} in file does not match object attribute. (Diff : {1:.2e})'
                
                # Starts with k, since that'll affect the omegabar arrays
                if hdu[0].header['k'] != self.k:
                    print( warning_temp_line1.format( 'k' ) )
                    print( warning_temp_line2.format( hdu[0].header['k'], self.k, hdu[0].header['k'] - self.k ) )
            
                # Then checks omegabar values
                check_dict = OrderedDict([ ( 'AFmin' , self.omegabar[0+self.k] ), \
                                           ( 'AFmax' , self.omegabar[-1-self.k] ), \
                                           ( 'AFres' , self.omegabar[1]-self.omegabar[0] ), \
                                           ( 'AFbins', self.omegabar.size ), \
                                           ( 'AFdata', self.omegabar.size - 2*self.k ) ])
                for key in check_dict.keys():
                    if hdu[0].header[key] != check_dict[key]:
                        print( warning_temp_line.format( key , hdu[0].header[key] - check_dict[key]) )
            
                # Checks tau and tauf values
                #    tauf used to be called beta, so if tauf key is not in the hdu header, looks for the old
                #    beta key
                check_dict = OrderedDict([ ( 'taures' , self.fracLOS.size ), \
                                           ( 'taufN'  , self.taufs.size ), \
                                           ( 'taufmin', self.taufs[0] ), \
                                           ( 'taufmax', self.taufs[-1] ) ])
                for key in check_dict.keys():
                    hdu_key = str(key)
                    if hdu_key not in hdu[0].header.keys():
                        hdu_key = hdu_key.replace( 'tauf', 'beta' )
                    if hdu[0].header[hdu_key] != check_dict[key]:
                        print( warning_temp_line.format( key , hdu[0].header[hdu_key] - check_dict[key]) )
            
                # Checks other values present for all sims
                check_dict = OrderedDict([ ( 'cloud'   , self.cloud ), \
                                           ( 'Doppler' , self.W ), \
                                           ( 'Zeeman'  , (self.omegabar[1]-self.omegabar[0])*float(self.k) ), \
                                           ( 'theta'   , self.theta ), \
                                           ( 'phi'     , self.phi ), \
                                           ( 'etap'    , self.etap ), \
                                           ( 'etam'    , self.etam ), \
                                           ( 'alphap'  , self.alphap ), \
                                           ( 'alpham'  , self.alpham ), \
                                           ( 'i0'      , self.iquv0[0] ), \
                                           ( 'q0'      , self.iquv0[1] ), \
                                           ( 'u0'      , self.iquv0[2] ), \
                                           ( 'v0'      , self.iquv0[3] ), \
                                           ( 'endfill' , self.endfill ), \
                                           ( 'farcoeff', self.far_coeff ), \
                                           ( 'nexp'    , self.n ), \
                                           ( 'ftol'    , self.ftol ) ])
                for key in check_dict.keys():
                    if hdu[0].header[key] != check_dict[key]:
                        print( warning_temp_line.format( key , hdu[0].header[key] - check_dict[key]) )
            
                # Checking fcalc values if set for both
                if 'ne' in hdu[0].header.keys() and self.fcalc:
                    check_dict = OrderedDict([ ( 'ne'   , self.ne ), \
                                               ( 'AF0'  , self.freq0 ), \
                                               ( 'Gamma', self.Gamma ), \
                                               ( 'B'    , self.B ), \
                                               ( 'A0'   , self.A0 ), \
                                               ( 'P0'   , self.P0 ) ])
                    for key in check_dict.keys():
                        if hdu[0].header[key] != check_dict[key]:
                            print( warning_temp_line.format( key , hdu[0].header[key] - check_dict[key]) )
            
                # Checking values for 2nd ray if both are bi-directional
                if hdu[0].header['cloud'] == 2 and self.cloud == 2:
                    check_dict = OrderedDict([ ( 'iF'      , self.iquvF[0] ), \
                                               ( 'qF'      , self.iquvF[1] ), \
                                               ( 'uF'      , self.iquvF[2] ), \
                                               ( 'vF'      , self.iquvF[3] ) ])
                    for key in check_dict.keys():
                        if hdu[0].header[key] != check_dict[key]:
                            print( warning_temp_line.format( key , hdu[0].header[key] - check_dict[key]) )
            
            
            
            
            #### Retrieves other values associated with stokes calculation ####
            
            # Retrieving index of optical depth at which the stokes values are calculated
            tau_idx = hdu[0].header['tauidx']
            
            # Saves array of taufs values
            self.taufs = hdu[0].data
            
            # Retrieves each stokes array and sets as corresponding attribute
            self.stacked_stoki = hdu[1].data
            self.stacked_stokq = hdu[2].data
            self.stacked_stoku = hdu[3].data
            self.stacked_stokv = hdu[4].data
            self.stacked_mc    = hdu[5].data
            self.stacked_ml    = hdu[6].data
            self.stacked_evpa  = hdu[7].data
            
        # Saves tau_idx as object attribute
        self.tau_idx = tau_idx
            
    def calc_R(self, summed = True, Gamma = None, verbose = False, sep = False ):
        """
        Program that calculates the stimulated emission rate, R, at the end of the cloud from the 
        input dimensionless inversion equations, delta.
        
        If the loss rate (Gamma, in inverse seconds) is not provided, only calculates the ratio of the
        stimulated emission rate to the loss rate. If the loss rate, Gamma, is provided, will return
        the stimulated emission rate, R, in inverse seconds.
        
        Optional Parameters:
            
            Gamma           Float or None 
            					
            					[ default = None ]
            
                                The loss rate in inverse seconds. If provided, will calculate and 
                                return the stimulated emission rate, R. If set to None, will 
                                calculate and return R/Gamma. 
            
            summed			Boolean True/False
            					
            					[ default = True ]
            					
            					Whether to calculate and return the sum of the stimulated emission 
            					rate, R, over all n angular frequency bins (if True), or only 
            					return/save the values of R calculated in the line center angular 
            					frequency bin (if False).
            					
            					Note: the line center bin is determined as the bin at which the
            					value of omegabar is closest to zero. If the omegabar array was
            					selected in such a way that there is not a bin at omegabar = 0, this
            					will not truly be the line center stimulated emission rate.
                                
            verbose         Boolean True/False [ default = False ]
            
                                Whether to print out progress during calculation.
                                
            sep             Boolean True/False [ default = False ]
            
                                If set to True, will return stimulated emission rate values for each
                                transition separately, in the order minus, zero, plus. If set to
                                False, will return only one value with the three stimulated emission
                                rate values summed.
        
        Other Object Attributes Used:
            
            self.theta      The angle between the magnetic field and line of sight in radians.
            self.costheta   cos( self.theta )
            self.sintheta   sin( self.theta )
            self.phi        The sky-plane angle in radians.
            self.costwophi  cos( 2 * self.phi )
            self.sintwophi  sin( 2 * self.phi )
            self.etap       The squared ratio of the + dipole moment to the 0th dipole moment.
            self.etam       The squared ratio of the - dipole moment to the 0th dipole moment.
            self.omegabar   The array of angular frequencies.
            self.k          The number of frequency bins spanned by the Zeeman shift, delta omega
            
        Returns:
            
            outval          Float
            
                                Either the stimulated emission rate, in inverse seconds, at the cloud
                                end if Gamma was provided, or the ratio of R/Gamma, if Gamma was not
                                provided.
                                
                                If summed was set to True, this will be the value summed over all
                                n angular frequency bins. If summed was set to False, the value 
                                returned only reflects the value in the central angular frequency
                                bin.
        """
        
        # Figures out tauf value from tau and fracLOS attributes
        tauf = float(self.tau[-1]) / float(self.fracLOS[-1])
        
        # Simplifies name of attribute k
        k = self.k
        
        # Calculates 1 + cos(theta)^2 for quick reference
        cos2thetap1 = 1.0 + self.costheta**2
        
        # If Gamma wasn't provided, sets to 1.0 
        if Gamma is None: 
            Gamma = 1.0
        
        # Calculates the Stokes values for the currently loaded array
        #    Stokes array shapes are (T,NV+2k)
        self.stokes( verbose = False )
        
        # Calculates the stimulated emission rates for each 0th, -, and + transition, as arrays
        #    varying with angular frequency bins at line end. Shape should be (NV,).
        Rzero_n  = 2.0 * Gamma * self.sintheta * self.sintheta * ( self.stoki[ -1, k : -1*k ] \
                   - self.stokq[ -1, k : -1*k ] * self.costwophi - self.stoku[ -1, k : -1*k ] * self.sintwophi )
                   
        Rplus_n  = Gamma * self.etap * ( cos2thetap1 * self.stoki[ -1, : -2*k ] - 2. * self.stokv[ -1, : -2*k ] * self.costheta \
                   + ( self.costwophi * self.stokq[ -1, : -2*k ] + self.sintwophi * self.stoku[ -1, : -2*k ] ) * self.sintheta * self.sintheta )
        
        Rminus_n = Gamma * self.etam * ( cos2thetap1 * self.stoki[ -1, 2*k : ]  + 2. * self.stokv[ -1, 2*k : ] * self.costheta \
                   + ( self.costwophi * self.stokq[ -1, 2*k :  ] + self.sintwophi * self.stoku[ -1, 2*k :  ] ) * self.sintheta * self.sintheta )
        
        if verbose:
            print(' Rzero_n shape : {0},   Rplus_n shape : {1},   R_minus_n shape : {2} '.format( Rzero_n.shape, \
                                                                                        Rplus_n.shape, Rminus_n.shape ))
        # Integrates each term along angular frequency bin using trap rule
        # domegabar = self.omegabar[1] - self.omegabar[0]
        # Rzero  = ( 0.5*Rzero_n[0]  + Rzero_n[1:-1].sum()  + 0.5*Rzero_n[-1]  ) * domegabar
        # Rplus  = ( 0.5*Rplus_n[0]  + Rplus_n[1:-1].sum()  + 0.5*Rplus_n[-1]  ) * domegabar
        # Rminus = ( 0.5*Rminus_n[0] + Rminus_n[1:-1].sum() + 0.5*Rminus_n[-1] ) * domegabar
        
        # Sums each term over all n wavelength bins; does assume that the values -> 0 in the bins at the end, if requested
        if summed:
	        Rzero  = Rzero_n.sum()
	        Rplus  = Rplus_n.sum()
        	Rminus = Rminus_n.sum()
        
        # If line center R is requested, just extracts values from line center bins
        else:
        	
        	# omegabar array is 1D with shape ( NV+4k, ), while R####_n arrays calculated 
        	#	above have shape ( NV, )
        	# Finds the index where omegabar is closest to zero
        	ncent = np.where( np.abs(self.omegabar) == np.nanmin(np.abs(self.omegabar)) )[0][0]
        	
        	# Pulls out values at line center, which are shifted in index by 2k bins
        	Rzero  = Rzero_n[  ncent + 2*k ]
        	Rplus  = Rplus_n[  ncent + 2*k ]
        	Rminus = Rminus_n[ ncent + 2*k ]
        
        # If we're summing the three rates, sums
        if not sep:
            outval = Rminus + Rzero + Rplus
        
        # If returning separate values, sets outval to be tuple
        else:
            outval = ( Rminus, Rzero, Rplus )
        
        # Returns
        return outval
    
    
    ### Functions for plotting figures ###
            
    def plot_cloud( self, plotvalue, convert_freq = True, subtitle = None, figname = None, show = True ):
        """
        Plots desired value vs. frequency (x-axis) and fractional optical depth of the cloud (y-axis).
    
        Can be used to plot mc, ml, evpa, stoki, stokq, stoku, stokv, fracq (stokq/stoki), or fracu (stoku/stoki).
        
        Note: Unlike plot_freq_tau method, which plots the values for multiple total optical depth solutions, this
        plots a single solution / total optical depth tauf along the line of sight.
        
        Intended to be run *after* the deltas solution for a given tauf value has been read in (and set as the 
        object's deltas attribute), the tauf value and the associated attributes have been updated accordingly, and
        the stokes parameters have been calculated. Eg:
        
            >>> my_maser = maser( parfile = myparfile, theta = 10.*pi/180., outpath = 'my_path', k = 1)
            >>>
            >>> # Read in tauf = 1.0 solution
            ... my_maser.deltas = my_maser.readin( 1.0, ext='fits', updatepars = False ) 
            >>>
            >>> # Update tauf values
            ... my_maser.update_tauf( 1.0 )
            >>>
            >>> # Calculate Stokes attributes
            ... my_maser.stokes()
            >>>
            >>> # Finally, plot cloud
            ... my_maser.plot_cloud( 'stoki' )
        
        Required Parameters:
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), or 'fracu' for (stoku/stoki).
            
        Optional Parameters:
            
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False).
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
        """
        
        
        #### Determines frequency extent for plot and converts, if requested ####
        
        # Gets ANGULAR frequency range for x-axis of plot; used for imshow extent, not plot limits
        dfreq = self.omegabar[1] - self.omegabar[0]
        freqmin = self.omegabar[self.k] - dfreq/2.
        freqmax = self.omegabar[-self.k-1] + dfreq/2.
    
        # Converts these to frequency, if requested by freq_conv
        if convert_freq:
            dfreq = dfreq / ( 2.*pi )
            freqmin = freqmin / ( 2.*pi )
            freqmax = freqmax / ( 2.*pi )
        
            # Converts to MHz
            dfreq *= 1e-6
            freqmin *= 1e-6
            freqmax *= 1e-6
        
            # Generates axis label for frequency, while we're here
            xlabel = r'$\nu$ [MHz]'
    
        # If no conversion requested, just generates axis label
        else:
            xlabel = r'$\varpi$ [s$^{-1}$]'
        
        
        
        
        
        #### Gets tau limits for plot ####
        # Figure will go from minimum to maximum tau
        tau_min = self.tau[0]
        tau_max = self.tau[-1]
        
        # Plot extent (bins) from half a bin width below min to half a bin above max
        tau_ext_min = tau_min - 0.5 * self.dtau
        tau_ext_max = tau_max + 0.5 * self.dtau
        
        
        
        
        
        #### Converts theta to degrees for labels ####
        theta_degrees = self.theta * 180. / pi
        
        
        
        
        
        
        #### Preliminary checks of provided values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "Setting show = False without specifying a file name for the plot will result in no figure produced.\n"+\
                      "        Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
            
        
        # Makes sure that specified plotvalue is allowed:
        if plotvalue not in ['mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', 'fracu']:
            err_msg = "plotvalue '{0}' not recognized.\n".format(plotvalue) + \
                      "        Allowed values are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', or 'fracu'."
            raise ValueError(err_msg)
        
        
        
        
        #### Determining which array to plot ####
        
        # Start with Stokes i
        elif plotvalue == 'stoki':
            
            # Set aside array with plotable range
            temparray = self.stoki
            
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes i with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'viridis'
        
        # Next, Stokes q
        elif plotvalue == 'stokq':
        
            # Array to plot is just stokq
            temparray = self.stokq
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Next, Stokes u
        elif plotvalue == 'stoku':
        
            # Array to plot is just stoku
            temparray = self.stoku
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Next, Stokes v
        elif plotvalue == 'stokv':
        
            # Array to plot is just stokv
            temparray = self.stokv
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes v with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Next, fractional stokes q
        elif plotvalue == 'fracq':
        
            # Array to plot is just stokq/stoki
            temparray = self.stokq / self.stoki
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q/i with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Next, fractional stokes u
        elif plotvalue == 'fracu':
        
            # Array to plot is just stokq/stoki
            temparray = self.stoku / self.stoki
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u/i with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Next, does ml
        elif plotvalue == 'ml':
            
            # Array to plot is just ml
            temparray = self.ml
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l$ with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'viridis'
        
        # Next, does mc
        elif plotvalue == 'mc':
            
            # Array to plot is just mc
            temparray = self.mc
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_c$ with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
        
        # Finally, does evpa
        elif plotvalue == 'evpa':
        
            # Array to plot is just evpa; makes sure phase wrapped between 0 and pi
            temparray = ( self.evpa + pi ) % pi
        
            # Sets plot title & colormap, while we're here
            fig_title = r'EVPA with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$ and $\tau_f=$'+str(round(self.tauf,1))
            cmap = 'RdBu'
         
            
            
        
        
        
        #### Figures out min and max for color ####
            
        if temparray.size != 0:
            
            vmax = np.nanmax( np.abs( temparray[ : , 2*self.k : -2*self.k ] ) )
            if plotvalue == 'evpa':
                vmin = 0.0
                vmax = pi
            elif cmap == 'RdBu':
                vmin = -1.0 * vmax
            else:
                vmin = 0.0
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            P.imshow( temparray, aspect='auto', extent=[freqmin, freqmax, tau_ext_min, tau_ext_max ], \
                origin='lower', vmin = vmin, vmax = vmax, cmap=cmap )
            
            # Axis limits & log scale on y-axis
            P.xlim( freqmin, freqmax )
            P.ylim( tau_min, tau_max )
            
            # If frequency conversion not requested, makes x-axis frequency ticks in scientific notation
            if not convert_freq:
                P.ticklabel_format( axis='x', style = 'sci', scilimits = (0,0) )
            
            # Axis labels and title
            P.xlabel(xlabel)
            P.ylabel(r'$\tau$ along L.o.S.')
            if subtitle is None:
                P.title( fig_title )
            else:
                P.title( fig_title + '\n' + subtitle )
            
            # Colorbar
            cbar = P.colorbar()
            if vmax <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
        
            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
            
    def plot_freq_tau(self, plotvalue, taufmax = None, plottaufmax = 100.0, convert_freq = True, tau_scale = 'linear', \
                             interp = 'cubic', subtitle = None, figname = None, show = True, verbose = True ):
        """
        Plots desired value vs. frequency (x-axis) and total optical depth of the cloud (y-axis).
    
        Can be used to plot mc, ml, evpa, stoki, stokq, stoku, stokv, fracq (stokq/stoki), or fracu (stoku/stoki).
        
        Intended to be run *after* stokes at a given point in cloud have been read in for a range of tauf values
        with cloud_end_stokes method.
        
        Required Parameters:
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), or 'fracu' for (stoku/stoki).
            
        Optional Parameters:
            
            taufmax         None or Float 
                                [ Default = None ]
                                Maximum value of tauf to show data for in the plot. If None, plots all available 
                                data.
                                
            plottaufmax     Float or None
                                [ Default = None ]
                                Y-limit (optical depth) shown on the plot axes. If None, will be the same as
                                taufmax. (Only useful if you want to set the y-limit used by the figure to be
                                the same as other figures despite not having tauf up to that value for this
                                parameter set.)
                        
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False).
            
            tau_scale       String ('log' or 'linear')
                                [ Default = 'linear' ]
                                Scale for the y-axis (total optical depth) on the plot.
                                
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'.
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        
        
        #### Determines frequency extent for plot and converts, if requested ####
        
        # Gets ANGULAR frequency range for x-axis of plot; used for imshow extent, not plot limits
        dfreq = self.omegabar[1] - self.omegabar[0]
        freqmin = self.omegabar[self.k] - dfreq/2.
        freqmax = self.omegabar[-self.k-1] + dfreq/2.
    
        # Converts these to frequency, if requested
        if convert_freq:
            dfreq = dfreq / ( 2.*pi )
            freqmin = freqmin / ( 2.*pi )
            freqmax = freqmax / ( 2.*pi )
        
            # Converts to MHz
            dfreq *= 1e-6
            freqmin *= 1e-6
            freqmax *= 1e-6
        
            # Generates axis label for frequency, while we're here
            xlabel = r'$\nu$ [MHz]'
    
        # If no conversion requested, just generates axis label
        else:
            xlabel = r'$\varpi$ [s$^{-1}$]'
        
        
        
        
        #### Processing defaults for taufmax and plottaufmax ####
        
        # Finds index of maximum tauf present for all ml arrays
        if taufmax is not None:
            ibmax = np.where( self.taufs == taufmax )[0]
            if ibmax.size > 0:
                ibmax = ibmax[0]
            else:
                raise ValueError("taufmax value must be in taufs array attribute.")
        
        # If using default taufmax, just uses last one with stacked_stoki calculated
        elif taufmax is None:
            if 'stacked_stoki' in self.__dict__.keys():
                ibmax = self.stacked_stoki.shape[0] - 1
                taufmax = self.taufs[ibmax]
            else:
                raise AttributeError('Maser object has no attribute stacked_stoki. Please run cloud_end_stokes before\n'+\
                                     '    calling this method.')
        
        # Sets plottaufmax, if not set
        if plottaufmax is None:
            plottaufmax = taufmax
        
        
            
            
            
        
        #### Converts theta to degrees for labels ####
        theta_degrees = self.theta * 180. / pi
        
        
        
        
        
        
        #### Preliminary checks of provided values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "Setting show = False without specifying a file name for the plot will result in no figure produced.\n"+\
                      "        Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Checks that tau_scale is an acceptable value
        tau_scale = tau_scale.lower()
        if tau_scale not in ['linear','log']:
            err_msg = "tau_scale '{0}' not recognized.\n".format(tau_scale) + \
                      "        Allowed values are 'linear' or 'log'."
            raise ValueError(err_msg)
            
        
        # Makes sure that specified plotvalue is allowed:
        if plotvalue not in ['mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', 'fracu']:
            err_msg = "plotvalue '{0}' not recognized.\n".format(plotvalue) + \
                      "        Allowed values are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', or 'fracu'."
            raise ValueError(err_msg)
        
        
        
        
        #### Determining which array to plot ####
        
        # Start with Stokes i
        elif plotvalue == 'stoki':
            
            # Set aside array with plotable range
            temparray = self.stacked_stoki[ :ibmax+1, : ]
            
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes i at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'viridis'
        
        # Next, Stokes q
        elif plotvalue == 'stokq':
        
            # Array to plot is just stokq_all
            temparray = self.stacked_stokq[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Next, Stokes u
        elif plotvalue == 'stoku':
        
            # Array to plot is just stoku_all
            temparray = self.stacked_stoku[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Next, Stokes v
        elif plotvalue == 'stokv':
        
            # Array to plot is just stokv_all
            temparray = self.stacked_stokv[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes v at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Next, fractional stokes q
        elif plotvalue == 'fracq':
        
            # Array to plot is just stokq_all/stoki_all
            temparray = self.stacked_stokq[ :ibmax+1, : ] / self.stacked_stoki[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q/i at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Next, fractional stokes u
        elif plotvalue == 'fracu':
        
            # Array to plot is just stokq_all/stoki_all
            temparray = self.stacked_stoku[ :ibmax+1, : ] / self.stacked_stoki[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u/i at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Next, does ml
        elif plotvalue == 'ml':
            
            # Array to plot is just stacked_ml
            temparray = self.stacked_ml[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l$ at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'viridis'
        
        # Next, does mc
        elif plotvalue == 'mc':
            
            # Array to plot is just stacked_mc
            temparray = self.stacked_mc[ :ibmax+1, : ]
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_c$ at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
        
        # Finally, does evpa
        elif plotvalue == 'evpa':
        
            # Array to plot is just evpa_all; makes sure phase wrapped between 0 and pi
            temparray = ( self.stacked_evpa[ :ibmax+1, : ] + pi ) % pi
        
            # Sets plot title & colormap, while we're here
            fig_title = r'EVPA at Cloud End with $\theta=$'+str(round(theta_degrees,1))+r'$^{\circ}$'
            cmap = 'RdBu'
            
        
        
        
        
        #### Regridding array for smooth distribution of tauf solutions ####
        
        if temparray.size != 0:
            
            if verbose:
                print('Regridding data...')
            
            # Ravels temparray to prepare for regridding
            temparray = np.ravel( temparray )
        
            # Creates grid of existing frequency/tauf points
            freqpts, taufpts = np.meshgrid( self.omegabar[self.k : -self.k], self.taufs[ :ibmax+1] )
            freqpts = np.ravel( freqpts )
            taufpts = np.ravel( taufpts )
            points  = np.vstack(( freqpts, taufpts )).T
            
            # Creates grid of desired total optical depths and regrids. 
            #   Assumed frequency is already equi-spaced so doesn't change.
            if tau_scale == 'log':
                taufgoal = np.logspace( log10(self.taufs[0]), log10(taufmax), 1001)
                freqgrid, taufgrid = np.meshgrid( self.omegabar[self.k : -self.k], taufgoal )
            
                # Converts taufgoal to log space to figure out extent limits for imshow
                dtaufgoal = taufgoal[1] - taufgoal[0]
                tau_ext_min = taufgoal[0]  - dtaufgoal/2.0
                tau_ext_max = taufgoal[-1] + dtaufgoal/2.0
            
            # Does the same if linear tauf is desired
            else:
                taufgoal = np.linspace( self.taufs[0], taufmax, 1001)
                freqgrid, taufgrid = np.meshgrid( self.omegabar[self.k : -self.k], taufgoal )
            
                dtaufgoal = taufgoal[1] - taufgoal[0]
                tau_ext_min = taufgoal[0]  - dtaufgoal/2.0
                tau_ext_max = taufgoal[-1] + dtaufgoal/2.0
                
            
            # Re-grids data array with desired interpolation
            zs = griddata( points, temparray, (freqgrid, taufgrid), method=interp)
            
            
            
        
        
        
            #### Figures out min and max for color ####
            
            if verbose:
                print('Plotting figure...')
            
            vmax = np.nanmax( np.abs( zs[ : , 2*self.k : -2*self.k ] ) )
            if plotvalue == 'evpa':
                vmin = 0.0
                vmax = pi
            elif cmap == 'RdBu':
                vmin = -1.0 * vmax
            else:
                vmin = 0.0
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            P.close()
            fig, ax = P.subplots(nrows=1, ncols=1, figsize = (5.5,4.5))
            fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.88 )
            
            # Plots image
            if tau_scale == 'log':
                # Get correct bin ends on y-scale
                dy = log10(taufgoal[1]/taufgoal[0])
                #print('dy = {0}'.format(dy))
                freq_plot = np.linspace( freqmin, freqmax, (self.omegabar.size - 2*self.k ) + 1 )
                tauf_plot  = np.logspace( log10(self.taufs[0]) - dy/2. , log10(taufmax) + dy/2., taufgoal.size+1 )
                #print('theta_plot from {0} to {1}'.format(theta_plot[0], theta_plot[-1]))
                #print('tauf_plot from {0} to {1}'.format(tauf_plot[0], tauf_plot[-1]))
                im = ax.pcolormesh( freq_plot, tauf_plot, zs, vmin = vmin, vmax = vmax, cmap = cmap )
                ax.set_yscale( 'log' )
            else:
                dtauf  = taufgoal[1]  - taufgoal[0]
                im = ax.imshow( zs, aspect='auto', origin='lower', vmin = vmin, vmax = vmax, cmap = cmap, \
                                extent = [ freqmin, freqmax, taufgoal[0] - dtauf/2., taufgoal[-1] + dtauf/2.] )
            
            # Axis limits & log scale on y-axis
            ax.set_xlim( freqmin, freqmax )
            ax.set_ylim( self.taufs[0], plottaufmax )

            if tau_scale == 'log':
                ax.set_yscale( 'log' )
            
            # If frequency conversion not requested, makes x-axis frequency ticks in scientific notation
            if not convert_freq:
                ax.set_ticklabel_format( axis='x', style = 'sci', scilimits = (0,0) )
            
            # Axis labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Total $\tau_f$')

            if subtitle is None:
                ax.set_title( fig_title )
            else:
                ax.set_title( fig_title + '\n' + subtitle )
            
            # Colorbar
            cbar = fig.colorbar( im, ax=ax )
            if vmax <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
        
            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                    if verbose:
                        print('Saved figure to {0}.'.format(figname))
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
    
        
    
    ### Functions for writing output ###

    def write_deltas(self, deltas, ext='fits', broken=False):
        """
        Writes delta{-,0,+} to output fits or txt file, depending on requested extension. Used by run.
        """
        
        # Makes sure extension doesn't start with .
        if ext.startswith('.'):
            ext = ext[1:]
        
        
        # Writing if txt file requested
        if ext == 'txt':
        
            # Creates path for output for delta minus, 0, and plus
            if not broken:
                outpath_minus = '{0}{1}_tauf{2}_dminus.{3}'.format(self.outpath, self.filename, self.tauf, ext )
                outpath_zero  = '{0}{1}_tauf{2}_dzero.{3}'.format(self.outpath, self.filename, self.tauf, ext )
                outpath_plus  = '{0}{1}_tauf{2}_dplus.{3}'.format(self.outpath, self.filename, self.tauf, ext )
            else:
                outpath_minus = '{0}{1}_tauf{2}_dminus_BROKEN.{3}'.format(self.outpath, self.filename, self.tauf, ext )
                outpath_zero  = '{0}{1}_tauf{2}_dzero_BROKEN.{3}'.format(self.outpath, self.filename, self.tauf, ext )
                outpath_plus  = '{0}{1}_tauf{2}_dplus_BROKEN.{3}'.format(self.outpath, self.filename, self.tauf, ext )
            
            # Writes output to text file with numpy savetxt
            np.savetxt(outpath_minus,deltas[:,:,0],fmt='%.18f')
            np.savetxt(outpath_zero, deltas[:,:,1],fmt='%.18f')
            np.savetxt(outpath_plus, deltas[:,:,2],fmt='%.18f')
            
            # Prints feedback if requested
            if self.verbose == True:
                print('Done.')
                print('PRISM.MASER: Output files written:')
                print('PRISM.MASER:     {0}'.format(outpath_minus))
                print('PRISM.MASER:     {0}'.format(outpath_zero))
                print('PRISM.MASER:     {0}'.format(outpath_plus))
        
        # Writing fits file if requested
        elif ext == 'fits':
            
            # Creates single path for output for delta minus, 0, and plus
            if not broken:
                outpath = '{0}{1}_tauf{2}.{3}'.format(self.outpath, self.filename, self.tauf, ext )
            else:
                outpath = '{0}{1}_tauf{2}_BROKEN.{3}'.format(self.outpath, self.filename, self.tauf, ext )
            
            # Makes primary HDU with no data
            prime_hdu = fits.PrimaryHDU()
            
            # Populates primary header with sim info
            prime_hdu.header['cloud'] = ( self.cloud, 'number of rays' )
            prime_hdu.header['Doppler'] = ( self.W, 'Doppler width [s^-1]' )
            prime_hdu.header['Zeeman'] = ( (self.omegabar[1]-self.omegabar[0])*float(self.k), 'Zeeman splitting [s^-1]' )
            prime_hdu.header['k'] = ( self.k, 'Zeeman splitting [bins]' )
            prime_hdu.header['AFres'] = ( self.omegabar[1]-self.omegabar[0], 'Angular Freq Resolution [s^-1]' )
            prime_hdu.header['AFbins'] = ( self.omegabar.size, 'Angular Freq Resolution Bins' )
            prime_hdu.header['taubins'] = ( self.fracLOS.size, 'Number of Tau Resolution Bins' )
            prime_hdu.header['tau'] = ( self.tau[-1] / self.fracLOS[-1], 'Total Optical Depth' )
            prime_hdu.header['theta'] = ( self.theta, 'B field angle [rad]' )
            prime_hdu.header['phi'] = ( self.phi, 'Sky angle [rad]' )
            prime_hdu.header['etap'] = ( self.etap, '|d^+|^2 / |d^0|^2' )
            prime_hdu.header['etam'] = ( self.etap, '|d^-|^2 / |d^0|^2' )
            prime_hdu.header['alphap'] = ( self.alphap, 'P^+ / P^0' )
            prime_hdu.header['alpham'] = ( self.alpham, 'P^- / P^0' )
            prime_hdu.header['i0'] = ( self.iquv0[0], 'Ray 1 initial Stokes i' )
            prime_hdu.header['q0'] = ( self.iquv0[1], 'Ray 1 initial Stokes q' )
            prime_hdu.header['u0'] = ( self.iquv0[2], 'Ray 1 initial Stokes u' )
            prime_hdu.header['v0'] = ( self.iquv0[3], 'Ray 1 initial Stokes v' )
            if self.cloud == 2:
                prime_hdu.header['iF'] = ( self.iquvF[0], 'Ray 2 initial Stokes i' )
                prime_hdu.header['qF'] = ( self.iquvF[1], 'Ray 2 initial Stokes q' )
                prime_hdu.header['uF'] = ( self.iquvF[2], 'Ray 2 initial Stokes u' )
                prime_hdu.header['vF'] = ( self.iquvF[3], 'Ray 2 initial Stokes v' )
            prime_hdu.header['endfill'] = ( self.endfill, 'Mode for handling freq edges' )
            prime_hdu.header['farcoeff'] = ( self.far_coeff, '-gamma_qu/cos(theta)' )
            prime_hdu.header['nexp'] = ( self.n, 'Number of Expansion Terms' )
            prime_hdu.header['ftol'] = ( self.ftol, 'Tolerance for convergence' )
            if self.fccalc:
                prime_hdu.header['ne'] = ( self.ne, 'Electron number density [m^-3]' )
                prime_hdu.header['AF0'] = ( self.freq0, 'Angular frequency at line center [s^-1]' )
                prime_hdu.header['Gamma'] = ( self.Gamma, 'Loss rate [s^-1]' )
                prime_hdu.header['B'] = ( self.B, 'Magnetic field strength [G]' )
                prime_hdu.header['A0'] = ( self.A0, 'Einstein A coeff [s^-1]' )
                prime_hdu.header['P0'] = ( self.P0, 'Pi Pump rate [m^-3 s^-1]' )
                
            # Makes HDU for each -/0/+
            hdu_m = fits.ImageHDU( deltas[:,:,0].astype( np.double ) )
            hdu_z = fits.ImageHDU( deltas[:,:,1].astype( np.double ) )
            hdu_p = fits.ImageHDU( deltas[:,:,2].astype( np.double ) )
            
            # Makes HDU list with each hdu as an extension
            hdu = fits.HDUList([ prime_hdu, hdu_m, hdu_z, hdu_p ])
            
            # Writes hdulist to file
            hdu.writeto( outpath )
            
            # Prints feedback if requested
            if self.verbose == True:
                print('PRISM.MASER: Output file {0} written.'.format( outpath ) )
    
    def write_desc(self, path):
        """
        Program to create text file summarizing the simulation.
        """
        
        # Makes path for output sim description
        descpath = '{0}sim_description.txt'.format(path)
        
        # writes output file describing this simulation
        desc = open(descpath, 'w+')
        desc.write('This simulation is for a:\n\n')
        
        # Line for if cloud is 1- or 2-ended
        lin = '  {0}-ended cloud\n'.format(self.cloud)
        desc.write(lin)
        
        # line for number of expansion terms
        lin = '  n = {0}\n'.format( self.n )
        desc.write(lin) 
        
        # Line for Zeeman splitting
        lin = '  Zeeman splitting {0:e} s^-1\n'.format( (self.omegabar[1]-self.omegabar[0])*float(self.k) )
        desc.write(lin)
        
        # Line for Doppler Width
        lin = '  Doppler Width {:.3e} s^-1\n'.format( self.W )
        desc.write(lin)
        
        # Line for frequency resolution
        lin = '  Omegabar resolution {0} s^-1 (k={1})\n'.format( int(self.omegabar[1]-self.omegabar[0]), self.k )
        desc.write(lin)
        
        # Line for number of tau resolution elements
        lin = '  Tau resolution elements = {0}\n'.format( self.fracLOS.size )
        desc.write(lin)
        
        # Line for maximum tauf
        lin = '  tauf = {0}\n'.format( self.taufs )
        desc.write(lin)
        
        # Line for angles
        lin = '  theta = {0}; phi = {1}\n'.format( self.theta, self.phi )
        desc.write(lin)
        
        # Line for etas
        lin = '  etap = {0}; etam = {1}\n'.format( self.etap, self.etam )
        desc.write(lin)
        
        # initial iquv line(s)
        lin = '  iquv0 = {0}\n'.format( self.iquv0 )
        desc.write(lin)
        if self.cloud == 2:
            lin = '  iquvF = {0}\n'.format( self.iquvF )
            desc.write(lin)
        
        # line for endfill argument
        lin = '  endfill = {0}\n\n'.format( self.endfill )
        desc.write(lin) 
        
        # If Faraday Coeff was calculated by the relevant subroutine, writes that section
        if self.fccalc:
            
            # Header line for this section
            lin = 'Faraday Coeff = {0}:\n'.format( self.far_coeff )
            desc.write(lin)
            
            # Line for electron density
            lin = '  ne = {0:.2e} m^-3 = {1:.2e} cm^-3\n'.format( self.ne, self.ne*1e-6 )
            desc.write(lin)
            
            # Line for central frequency
            lin = '  freq0 = {:.6e} s^-1\n'.format( self.freq0 )
            desc.write(lin)
            
            # Line for Gamma
            lin = '  Gamma = {:.0} s^-1\n'.format( self.Gamma )
            desc.write(lin)
            
            # Line for Mag Field, B
            lin = '  B = {:.0} G\n'.format( self.B )
            desc.write(lin)
            
            # Line for Einstein A coeff
            lin = '  A0 = {:.0e} s^-1\n'.format( self.A0 )
            desc.write(lin)
            
            # Line for P0
            lin = '  P0 = {0:.1e} m^-3 s^-1 = {1:.1e} cm^-3 s^-1\n\n'.format( self.P0, self.P0*1e-6 )
            desc.write(lin)
        
        else:
            
            # Otherwise, just prints the faraday coefficient
            lin = 'Faraday Coeff = {0}\n\n'.format( self.far_coeff )
            desc.write(lin)
        
        # Finally, prints full fracLOS and omegabar arrays
        lin = 'Full fracLOS Array (size={0}):\n  {1}\n\n'.format(self.fracLOS.size, self.fracLOS )
        desc.write(lin)
        lin = 'Full Omegabar array (size={0}):\n  {1}'.format( self.omegabar.size, self.omegabar )
        desc.write(lin)
        
        # Closes file
        desc.close()
        
    
    ### Lower-level calculation functions ###
    
    def inversion(self, delta, verbose=False):
        """
        Program that calculates the dimensionless inversion equations.
       
        The main input for this function, delta, should be a numpy array with dimensions 
        (T,NV+4k,3), where NV is the number of velocity bins and T is the number of tau bins. The 
        0th axis specifies values across different tau at constant frequency for a single 
        transition. The 1st axis specifies values across frequency at constant tau for a transition.
        The three rows along the 0th axis should be for delta^- (delta[0]), delta^0 (delta[1]), and 
        delta^+ (delta[2]).
        
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.tau is the array of 
        tau values that are being integrated over. self.endfill determines the method with which to 
        fill in the 2k end values of delta before object return
       
        self.cloud option can be 1 or 2 depending on whether the seed radiation is entering one (1) 
        or both (2) ends of the cloud. 
        
        self.iquv0 is an array of Stokes I, Q, U, and V values at tau[0]. This should be an array of 
        length 4 of the input stokes values for the light ray. If cloud=2, this will be the initial
        of ray 1.
        
        self.iquvF is the same as iquv0 but for the far end of the cloud corresponding to tau[T-1]. 
        This is only used for cloud=2.
        
        self.k is the number of frequency bins spanned by the Zeeman shift, delta omega.
        
        self.endfill (either 'fit' or 'zero'; default='fit') keyword determines how 2k frequency
        bins on either end will be filled in inversion function. If endfill='fit', a 4th order 
        polynomial as a function of frequency will be fit to the output arrays at each optical 
        depth, and the end values will be filled in according to their frequency. If endfill='zero', 
        the end points will just be set to zero.
        
        Always returns a shape (T,NV+4k,3) array of delta values in same format as input delta 
        array.
        
        Includes option to also return stokes i, q, u, and v values if stokes=True. Default is 
        False. If True, the second of 2 object returned will be a shape (4,NV+2k) array with
        unitless stokes i (iquv[0]), q (iquv[1]), u (iquv[2]), and v (iquv[3]) with the other axis
        giving the value at a certain frequency centered on the [k:-k] values of self.omegabar.
        
        """
    
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        #print('Inversion initialized. Splitting delta -/0/+')
        delta_m = delta[:,:,0]
        delta_0 = delta[:,:,1]
        delta_p = delta[:,:,2]
        
        # Then separates out end stokes values
        #print('Setting initial Stokes iquv')
        i0 = self.iquv0[0]
        q0 = self.iquv0[1]
        u0 = self.iquv0[2]
        v0 = self.iquv0[3]
        if self.cloud == 2:
            iF = self.iquvF[0]
            qF = self.iquvF[1]
            uF = self.iquvF[2]
            vF = self.iquvF[3]
        
        # simplify k name
        k = self.k
        
        
        if verbose:
            # Sets appropriate print options
            np.set_printoptions(precision=4, linewidth=180)
            print('STOKES TEST:')
            print('    tau:  ', self.tau)
            print('    dtau: ', self.dtau)
            print()
            print('    Delta in shape:', delta_m.shape )
            print('    Delta- in:  ', delta_m[:,verbose+2*k])
            print('    Delta0 in:  ', delta_0[:,verbose+2*k])
            print('    Delta+ in:  ', delta_p[:,verbose+2*k])
        
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
        
        
            # Prints test output if verbose
            if verbose:
                print('    gamma_I: ', gamma_I[:,verbose+k])
            
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            GI = np.zeros( gamma_I[0].shape )
            GQ = np.zeros( gamma_I[0].shape )
            GU = np.zeros( gamma_I[0].shape )
            GV = np.zeros( gamma_I[0].shape )
            GQU= np.zeros( gamma_I[0].shape )
            
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i]
                GIi  = 0.5*gamma_I[0,:]  + gamma_I[1:i,:].sum(axis=0)  + 0.5*gamma_I[i,:]
                GQi  = 0.5*gamma_Q[0,:]  + gamma_Q[1:i,:].sum(axis=0)  + 0.5*gamma_Q[i,:]
                GUi  = 0.5*gamma_U[0,:]  + gamma_U[1:i,:].sum(axis=0)  + 0.5*gamma_U[i,:]
                GVi  = 0.5*gamma_V[0,:]  + gamma_V[1:i,:].sum(axis=0)  + 0.5*gamma_V[i,:]
                GQUi = 0.5*gamma_QU[0,:] + gamma_QU[1:i,:].sum(axis=0) + 0.5*gamma_QU[i,:]
                
                GI = np.vstack(( GI, GIi ))
                GQ = np.vstack(( GQ, GQi ))
                GU = np.vstack(( GU, GUi ))
                GV = np.vstack(( GV, GVi ))
                GQU = np.vstack(( GQU, GQUi ))
                
                # Prints test output if verbose
            if verbose: # i == tau.size-1 and 
                print()
                print('    gamma_I sum: ', GI[:,verbose+k])
             
                
            # These are only sums. Multiplies these GX arrays by self.dtau before continuing
            GI = GI*self.dtau
            GQ = GQ*self.dtau
            GU = GU*self.dtau
            GV = GV*self.dtau
            GQU = GQU*self.dtau
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    GAMMA_I: ', GI[:,verbose+k])
            
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            scratch1 = np.ones( GI.shape )
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes = stokes0.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            scratch0 = np.zeros( GI.shape )
            Grow0 = np.stack(( GI,      -GQ,      -GU,      -GV ))
            Grow1 = np.stack((-GQ,       GI,      GQU, scratch0 ))
            Grow2 = np.stack((-GU,     -GQU,       GI, scratch0 ))
            Grow3 = np.stack((-GV, scratch0, scratch0,       GI ))
            Garray = np.stack(( Grow0, Grow1, Grow2, Grow3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    Gproduct = Garray.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    Gproduct = np.matmul( Gproduct, Garray, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term_n = np.matmul( Gproduct, self.iquv0, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes += ( term_n / float(factorial(n)) )
                else:
                    tempterm = ( term_n / float(factorial(170)) )
                    stokes  += tempterm * float( factorial(170)/factorial(n) )
            
            # After calculation, unpacks stokes
            stoki, stokq, stoku, stokv = stokes
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    StokesI: ', stoki[:,verbose+k])
            
                        
            # Calculates the dimensionless inversion equations, resulting in 1D arrays with length NV
            d_expterm = np.exp( - self.omegabar[2*k:-2*k]**2 /  self.W**2 ) 
            
            dplus_A = 2.*delta_0[:,2*k:-2*k]*( stoki[:,k:-k] - stokq[:,k:-k]*self.costwophi - \
                                                stoku[:,k:-k]*self.sintwophi ) * self.sintheta**2
            dplus_B = self.etam*delta_m[:,2*k:-2*k] * ( (1.+self.costheta**2)*stoki[:,2*k:] \
                            + 2.*stokv[:,2*k:]*self.costheta + ( stokq[:,2*k:]*self.costwophi \
                            + stoku[:,2*k:]*self.sintwophi )*self.sintheta**2 )
            dplus_C = 2.*self.etap * delta_p[:,2*k:-2*k] * ( (1.+self.costheta**2)*stoki[:,:-2*k] \
                            - 2.*stokv[:,:-2*k]*self.costheta + ( stokq[:,:-2*k]*self.costwophi \
                            + stoku[:,:-2*k]*self.sintwophi )*self.sintheta**2 )
            dplus_out = self.alphap * d_expterm  - ( dplus_A + dplus_B + dplus_C )
                    
            
            dzero_A = 4.*delta_0[:,2*k:-2*k]*( stoki[:,k:-k] - stokq[:,k:-k]*self.costwophi \
                                             - stoku[:,k:-k]*self.sintwophi ) * self.sintheta**2
            dzero_B = self.etap*delta_p[:,2*k:-2*k]*( (1+self.costheta**2)*stoki[:,:-2*k] \
                        + ( stokq[:,:-2*k]*self.costwophi + stoku[:,:-2*k]*self.sintwophi ) \
                        * self.sintheta**2 \
                        - 2.*stokv[:,:-2*k]*self.costheta )
            dzero_C = self.etam*delta_m[:,2*k:-2*k]*( (1+self.costheta**2)*stoki[:,2*k:] \
                        + ( stokq[:,2*k:]*self.costwophi + stoku[:,2*k:]*self.sintwophi ) \
                        * self.sintheta**2 \
                        + 2.*stokv[:,2*k:]*self.costheta )
            dzero_out = d_expterm - ( dzero_A + dzero_B + dzero_C )
            
            
            dminus_A = 2.*delta_0[:,2*k:-2*k]*( stoki[:,k:-k] - stokq[:,k:-k]*self.costwophi \
                                                - stoku[:,k:-k]*self.sintwophi ) * self.sintheta**2
            dminus_B = self.etap*delta_p[:,2*k:-2*k]*( (1+self.costheta**2)*stoki[:,:-2*k] \
                        - 2.*stokv[:,:-2*k]*self.costheta + ( stokq[:,:-2*k]*self.costwophi \
                        + stoku[:,:-2*k]*self.sintwophi )*self.sintheta**2 )
            dminus_C = 2.*self.etam*delta_m[:,2*k:-2*k]*( (1+self.costheta**2)*stoki[:,2*k:] \
                        + 2.*stokv[:,2*k:]*self.costheta + ( stokq[:,2*k:]*self.costwophi \
                        + stoku[:,2*k:]*self.sintwophi ) *self.sintheta**2 )
            dminus_out = self.alpham * d_expterm - ( dminus_A + dminus_B + dminus_C )
            
            
            
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
            # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            #print('Preparing for gain matrix integration')
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            # 1 is the direction from tau[0] to tau[i] and 2 is the direction from tau[T-1] to tau[i]
            # Iterating through ray 2 backwards, so this is the last column
            GI1 = np.zeros( gamma_I[0].shape )
            GQ1 = np.zeros( gamma_I[0].shape )
            GU1 = np.zeros( gamma_I[0].shape )
            GV1 = np.zeros( gamma_I[0].shape )
            GQU1= np.zeros( gamma_I[0].shape )
            GI2 = np.zeros( gamma_I[0].shape )
            GQ2 = np.zeros( gamma_I[0].shape )
            GU2 = np.zeros( gamma_I[0].shape )
            GV2 = np.zeros( gamma_I[0].shape )
            GQU2= np.zeros( gamma_I[0].shape )
            #print('Integrating gain matrix')
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i] for ray 1
                GI1i  = 0.5*gamma_I[0,:]   + gamma_I[1:i,:].sum(axis=0)   + 0.5*gamma_I[i,:]
                GQ1i  = 0.5*gamma_Q[0,:]   + gamma_Q[1:i,:].sum(axis=0)   + 0.5*gamma_Q[i,:]
                GU1i  = 0.5*gamma_U1[0,:]  + gamma_U1[1:i,:].sum(axis=0)  + 0.5*gamma_U1[i,:]
                GV1i  = 0.5*gamma_V1[0,:]  + gamma_V1[1:i,:].sum(axis=0)  + 0.5*gamma_V1[i,:]
                GQU1i = 0.5*gamma_QU1[0,:] + gamma_QU1[1:i,:].sum(axis=0) + 0.5*gamma_QU1[i,:]
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[i] and tau[T-1] for ray 2
                # This is being calculated backwards, so this is column -(i+1) for ray 2 
                GI2i  = 0.5*gamma_I[-1,:]   + gamma_I[-i:-1,:].sum(axis=0)  + 0.5*gamma_I[-i-1,:]
                GQ2i  = 0.5*gamma_Q[-1,:]   + gamma_Q[-i:-1,:].sum(axis=0)  + 0.5*gamma_Q[-i-1,:]
                GU2i  = 0.5*gamma_U2[-1,:]  + gamma_U2[-i:-1,:].sum(axis=0) + 0.5*gamma_U2[-i-1,:]           
                GV2i  = 0.5*gamma_V2[-1,:]  + gamma_V2[-i:-1,:].sum(axis=0) + 0.5*gamma_V2[-i-1,:]         
                GQU2i = 0.5*gamma_QU2[-1,:] + gamma_QU2[-i:-1,:].sum(axis=0)+ 0.5*gamma_QU2[-i-1,:]  
                
                # Adds the ith row onto the Gamma arrays for ray 1
                GI1 = np.vstack(( GI1, GI1i ))
                GQ1 = np.vstack(( GQ1, GQ1i ))
                GU1 = np.vstack(( GU1, GU1i ))
                GV1 = np.vstack(( GV1, GV1i ))
                GQU1 = np.vstack(( GQU1, GQU1i ))
                
                # Adds the ith row onto the Gamma arrays for ray q; technically the -(i+1) index for ray 2
                #   so does a reverse stack
                GI2 = np.vstack(( GI2i, GI2 ))
                GQ2 = np.vstack(( GQ2i, GQ2 ))
                GU2 = np.vstack(( GU2i, GU2 ))
                GV2 = np.vstack(( GV2i, GV2 ))
                GQU2 = np.vstack(( GQU2i, GQU2 ))
            
            # These are only sums. Multiplies these GX1 & GX2 arrays by self.dtau 
            GI1 = GI1*self.dtau
            GQ1 = GQ1*self.dtau
            GU1 = GU1*self.dtau
            GV1 = GV1*self.dtau
            GQU1 = GQU1*self.dtau
            GI2 = GI2*self.dtau
            GQ2 = GQ2*self.dtau
            GU2 = GU2*self.dtau
            GV2 = GV2*self.dtau
            GQU2 = GQU2*self.dtau
                        
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            #print('Preparing for Stokes calculation')
            # Creates some scratch arrays that can be used for both directions
            scratch1 = np.ones( GI1.shape )
            scratch0 = np.zeros( GI1.shape )
            
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            # Starts with iquv0 for direction 1
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes1 = stokes0.copy()
            
            # Then does iquvF for direction 2
            stokesF = np.stack((iF*scratch1, qF*scratch1, uF*scratch1, vF*scratch1))
            stokes2 = stokesF.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            # Does this first for Direction 1:
            G1row0 = np.stack(( GI1,     -GQ1,     -GU1,     -GV1 ))
            G1row1 = np.stack((-GQ1,      GI1,     GQU1, scratch0 ))
            G1row2 = np.stack((-GU1,    -GQU1,      GI1, scratch0 ))
            G1row3 = np.stack((-GV1, scratch0, scratch0,      GI1 ))
            G1array = np.stack(( G1row0, G1row1, G1row2, G1row3 ))
            
            # Then does the same for direction 2:
            G2row0 = np.stack(( GI2,     -GQ2,     -GU2,     -GV2 ))
            G2row1 = np.stack((-GQ2,      GI2,     GQU2, scratch0 ))
            G2row2 = np.stack((-GU2,    -GQU2,      GI2, scratch0 ))
            G2row3 = np.stack((-GV2, scratch0, scratch0,      GI2 ))
            G2array = np.stack(( G2row0, G2row1, G2row2, G2row3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    G1product = G1array.copy()
                    G2product = G2array.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    G1product = np.matmul( G1product, G1array, axes=[(0,1),(0,1),(0,1)] )
                    G2product = np.matmul( G2product, G2array, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term1_n = np.matmul( G1product, self.iquv0, axes=[(0,1),(0,),(0,)])
                term2_n = np.matmul( G2product, self.iquvF, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes1 += ( term1_n / float(factorial(n)) )
                    stokes2 += ( term2_n / float(factorial(n)) )
                else:
                    tempterm1 = term1_n / float(factorial(170))
                    tempterm2 = term2_n / float(factorial(170))
                    stokes1 += tempterm1 * float( factorial(170)/factorial(n) )
                    stokes2 += tempterm2 * float( factorial(170)/factorial(n) )
                    
            # After calculation, unpacks stokes
            stoki1, stokq1, stoku1, stokv1 = stokes1
            stoki2, stokq2, stoku2, stokv2 = stokes2
            
            # Finally, calculates the average of the stokes values from both arrays at each
            #     frequency and optical depth
            # UPDATED: Summing instead of averaging. As of bw10, Ray 2 no longer being flipped in velocity/frequency.
            #     Stokes u and v still being flipped in orientation.
            stoki = ( stoki1 + stoki2 )
            stokq = ( stokq1 + stokq2 )
            stoku = ( stoku1 - stoku2 )
            stokv = ( stokv1 - stokv2 )
            
            # Sets aside central part of input delta arrays for easy reference
            delta_0c = delta_0[ : , 2*k:-2*k ]
            delta_mc = delta_m[ : , 2*k:-2*k ]
            delta_pc = delta_p[ : , 2*k:-2*k ]
            
            # Calculates the dimensionless inversion equations, resulting in 1D arrays with length NV
            d_expterm = np.exp( - self.omegabar[2*k:-2*k]**2 /  self.W**2 ) 
            
            # UPDATED: As of bw10, Ray 2 lo longer flipped in velocity/frequency. Sign changes from theta and phi cancel
            #     with those of ray 2 stokes u and v, so can calculate straight in ref frame 1.
            # This term is the same for all d terms, includes term of *2 and sin^2 theta
            dzero_term = 2.0 *(  delta_0c * stoki[:,k:-k] \
                               - delta_0c * stokq[:,k:-k] * self.costwophi  \
                               - delta_0c * stoku[:,k:-k] * self.sintwophi  )*self.sintheta**2
                      
            # delta_minus term
            dminus_term = self.etam * delta_mc * \
                      (  ( stoki[:,2*k:] ) * (1.+self.costheta**2) \
                       + ( stokq[:,2*k:] ) * self.costwophi*self.sintheta**2 \
                       + ( stoku[:,2*k:] ) * self.sintwophi*self.sintheta**2 \
                       + ( stokv[:,2*k:] ) * 2.*self.costheta   )
            
            # delta_plus term
            dplus_term = self.etap * delta_pc * \
                      (  ( stoki[:,:-2*k] ) * (1.+self.costheta**2) \
                       + ( stokq[:,:-2*k] ) * self.costwophi*self.sintheta**2 \
                       + ( stoku[:,:-2*k] ) * self.sintwophi*self.sintheta**2 \
                       - ( stokv[:,:-2*k] ) * 2.*self.costheta  )
            
            # Combine all terms for final output deltas
            dplus_out  = self.alphap * d_expterm - (    dzero_term + dminus_term + 2.*dplus_term )
            dminus_out = self.alpham * d_expterm - (    dzero_term + dplus_term  + 2.*dminus_term)
            dzero_out  =               d_expterm - ( 2.*dzero_term + dminus_term +    dplus_term )
            
        
        # Next the end points that were chopped off previously need to be filled. 
        
        # If self.endfill = 'fit', this is done by 
        #     fitting a cubic spline to the dplus_out, dzero_out, and dminus_out and using that fit
        #     to predict the 2*k bins on each end that were removed during calculation. 
        if self.endfill == 'fit':
            # First creates arrays for left and right side of each d<transition>_out:
            lhs_plus = np.array([])
            rhs_plus = np.array([])
            lhs_zero = np.array([])
            rhs_zero = np.array([])
            lhs_minus = np.array([])
            rhs_minus = np.array([])
        
        
            # This can't be done simultaneously across 2 dimensions (that I'm aware of) so this has to
            #     iterate over optical depth. 
            for t in range( self.fracLOS.shape[0] ):
        
                # First generates the x-values for the polynomial fit
                x = self.omegabar[2*k:-2*k]
        
                # Fits for the coefficients in the delta^+ case. Should be a len-4 array with 
                #     coefficients from highest to lowest order
                coeff_plus = np.polyfit( x, dplus_out[t], 3 )
            
                # Creates the polynomial function with the correct coefficients
                fplus = np.poly1d( coeff_plus )
        
                # Similarly, fits for the coefficients in the delta^0 case.
                coeff_zero = np.polyfit( x, dzero_out[t], 3 )
                
                # Creates the polynomial function with the correct coefficients
                fzero = np.poly1d( coeff_zero )
        
                # Finally, fits for the coefficients in the delta^- case.
                coeff_minus = np.polyfit( x, dminus_out[t], 3 )
                
                # Creates the polynomial function with the correct coefficients
                fminus = np.poly1d( coeff_minus )
        
                # Generates endpoints on left- and right-hand sides
                if t == 0:
                
                    # First for + transition
                    lhs_plus = fplus( self.omegabar[:2*k] )
                    rhs_plus = fplus( self.omegabar[-2*k:] )
                
                    # Then for zero transition
                    lhs_zero = fzero( self.omegabar[:2*k] )
                    rhs_zero = fzero( self.omegabar[-2*k:] )
                
                    # Then for minus transition
                    lhs_minus = fminus( self.omegabar[:2*k] )
                    rhs_minus = fminus( self.omegabar[-2*k:] )
                
                else:
            
                    # First for + transition
                    lhs_plus = np.vstack(( lhs_plus, fplus( self.omegabar[:2*k] ) ))
                    rhs_plus = np.vstack(( rhs_plus, fplus( self.omegabar[-2*k:] ) ))
                
                    # Then for zero transition
                    lhs_zero = np.vstack(( lhs_zero, fzero( self.omegabar[:2*k] ) ))
                    rhs_zero = np.vstack(( rhs_zero, fzero( self.omegabar[-2*k:] ) ))
                
                    # Finally for minus transition
                    lhs_minus = np.vstack(( lhs_minus, fminus( self.omegabar[:2*k] ) ))
                    rhs_minus = np.vstack(( rhs_minus, fminus( self.omegabar[-2*k:] ) ))
                    
            # Combines into single output array for +
            dplus_out = np.hstack(( lhs_plus, dplus_out, rhs_plus ))
        
            # Combines into single output array for 0
            dzero_out = np.hstack(( lhs_zero, dzero_out, rhs_zero ))

            # Combines into single output array for -
            dminus_out = np.hstack(( lhs_minus, dminus_out, rhs_minus ))
                
                
        # If instead self.endfill = 'zero' or 'zeros', these values are just filled in as zero
        elif self.endfill == 'zero' or self.endfill == 'zeros':
            #print('Filling ends of array')
            # Creates zero buffer array to be used for all edges
            zbuffer = np.zeros(( self.tau.size, 2*k ))
            
            # Combines into single output array for +
            dplus_out = np.hstack(( zbuffer, dplus_out, zbuffer ))
        
            # Combines into single output array for 0
            dzero_out = np.hstack(( zbuffer, dzero_out, zbuffer ))

            # Combines into single output array for -
            dminus_out = np.hstack(( zbuffer, dminus_out, zbuffer ))
            
        #print('Finishing up.')
        # Stacks output arrays and returns
        delta_out = np.dstack(( dminus_out, dzero_out, dplus_out ))
        return delta_out
    
    def inversion_resid(self, delta ):
        """
        Uses inversion function to calculate the residual for each of the dimensionless inversions.
        
        Returns array of shape (3,NV+4k) with the len-3 axis corresponding to the minus, zero, and 
        plus transition, respectively, and the other axis corresponding to frequency values of 
        self.omegabar. 
        
        This residual is just the difference between the calculated and input values of the 
        dimensionless inversions, delta. To be used for zero-finding.
        """
        
        # First calculates the RHS side of the inversion equation
        delta_out = self.inversion( delta)
        
        # Calcualtes the difference between the output delta and the input delta
        resid = delta_out - delta
        
        # Prints residual for each run
        if self.verbose:
            print('    delta_resid:   min abs = {0}    max abs = {1}    sum abs = {2}'.format( \
                                   np.abs(resid).min(), np.abs(resid).max(), (np.abs(resid).mean(axis=1)).sum() ) )
        
        
        # Returns the residual
        return resid
    
    def gain_matrix(self, delta, tauf = None ):
        """
        Program that calculates and returns the dimensionless gain matrix components.
       
        The main input for this function, delta, should be a numpy array with dimensions (3,NV),
        where NV is the number of velocity bins. The three rows along the 0th axis should be for 
        delta^- (delta[0]), delta^0 (delta[1]), and delta^+ (delta[2]).
       
        
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. self.etap and self.etam are eta^+ and eta^-, 
        respectively, or the squared ratio of the + and - dipole matrix components to that of the 
        0th. self.far_coeff is -gamma_QU / cos(theta). self.k is the number of frequency bins 
        spanned by the Zeeman shift, delta omega.
       

        """
        
        # If tauf is provided along with delta array, sets
        if tauf is not None:
            self.update_tauf( tauf )
        
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        #print('Inversion initialized. Splitting delta -/0/+')
        delta_m = delta[:,:,0]
        delta_0 = delta[:,:,1]
        delta_p = delta[:,:,2]
        
        # Then separates out end stokes values
        #print('Setting initial Stokes iquv')
        i0 = self.iquv0[0]
        q0 = self.iquv0[1]
        u0 = self.iquv0[2]
        v0 = self.iquv0[3]
        if self.cloud == 2:
            iF = self.iquvF[0]
            qF = self.iquvF[1]
            uF = self.iquvF[2]
            vF = self.iquvF[3]
        
        # simplify k name
        k = self.k
        
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )

        
            # Combine and output
            outstack = np.dstack(( gamma_I, gamma_Q, gamma_U, gamma_V, gamma_QU*np.ones(gamma_I.shape) ))
            
            # Returns array of gamma values
            return outstack
        
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
             # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            # Combine output stacks
            outstack1 = np.dstack(( gamma_I, gamma_Q, gamma_U1, gamma_V1, gamma_QU1*np.ones(gamma_I.shape) ))
            outstack2 = np.dstack(( gamma_I, gamma_Q, gamma_U2, gamma_V2, gamma_QU2*np.ones(gamma_I.shape) ))
            
        
            # Returns array of gamma values
            return outstack1, outstack2
    
    def integ_gain(self, delta, tauf = None):
        """
        Program that calculates and returns the integral of the dimensionless gain matrix 
        components, Gamma_Stokes.
       
        The main input for this function, delta, should be a numpy array with dimensions (3,NV),
        where NV is the number of velocity bins. The three rows along the 0th axis should be for 
        delta^- (delta[0]), delta^0 (delta[1]), and delta^+ (delta[2]).
       
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. self.etap and self.etam are eta^+ and eta^-, 
        respectively, or the squared ratio of the + and - dipole matrix components to that of the 
        0th. self.far_coeff is -gamma_QU / cos(theta). self.k is the number of frequency bins 
        spanned by the Zeeman shift, delta omega.
       

        """
        # If tauf is provided along with delta array, sets
        if tauf is not None:
            self.update_tauf( tauf )
            
            
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        delta_m = delta[:,:,0]
        delta_0 = delta[:,:,1]
        delta_p = delta[:,:,2]
        
        # Then separates out end stokes values
        i0 = self.iquv0[0]
        q0 = self.iquv0[1]
        u0 = self.iquv0[2]
        v0 = self.iquv0[3]
        if self.cloud == 2:
            iF = self.iquvF[0]
            qF = self.iquvF[1]
            uF = self.iquvF[2]
            vF = self.iquvF[3]
        
        # simplify k name
        k = self.k
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
        
            
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            GI = np.zeros( gamma_I[0].shape )
            GQ = np.zeros( gamma_I[0].shape )
            GU = np.zeros( gamma_I[0].shape )
            GV = np.zeros( gamma_I[0].shape )
            GQU= np.zeros( gamma_I[0].shape )
            
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i]
                GIi  = 0.5*gamma_I[0,:]  + gamma_I[1:i,:].sum(axis=0)  + 0.5*gamma_I[i,:]
                GQi  = 0.5*gamma_Q[0,:]  + gamma_Q[1:i,:].sum(axis=0)  + 0.5*gamma_Q[i,:]
                GUi  = 0.5*gamma_U[0,:]  + gamma_U[1:i,:].sum(axis=0)  + 0.5*gamma_U[i,:]
                GVi  = 0.5*gamma_V[0,:]  + gamma_V[1:i,:].sum(axis=0)  + 0.5*gamma_V[i,:]
                GQUi = 0.5*gamma_QU[0,:] + gamma_QU[1:i,:].sum(axis=0) + 0.5*gamma_QU[i,:]
                
                GI = np.vstack(( GI, GIi ))
                GQ = np.vstack(( GQ, GQi ))
                GU = np.vstack(( GU, GUi ))
                GV = np.vstack(( GV, GVi ))
                GQU = np.vstack(( GQU, GQUi ))
             
                
            # These are only sums. Multiplies these GX arrays by self.dtau before continuing
            GI = GI*self.dtau
            GQ = GQ*self.dtau
            GU = GU*self.dtau
            GV = GV*self.dtau
            GQU = GQU*self.dtau
            
            # Determines what the output array is going to be
            outstack = np.dstack(( GI, GQ, GU, GV, GQU ))
        
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
            # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            #print('Preparing for gain matrix integration')
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            # 1 is the direction from tau[0] to tau[i] and 2 is the direction from tau[T-1] to tau[i]
            # Iterating through ray 2 backwards, so this is the last column
            GI1 = np.zeros( gamma_I[0].shape )
            GQ1 = np.zeros( gamma_I[0].shape )
            GU1 = np.zeros( gamma_I[0].shape )
            GV1 = np.zeros( gamma_I[0].shape )
            GQU1= np.zeros( gamma_I[0].shape )
            GI2 = np.zeros( gamma_I[0].shape )
            GQ2 = np.zeros( gamma_I[0].shape )
            GU2 = np.zeros( gamma_I[0].shape )
            GV2 = np.zeros( gamma_I[0].shape )
            GQU2= np.zeros( gamma_I[0].shape )
            #print('Integrating gain matrix')
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i] for ray 1
                GI1i  = 0.5*gamma_I[0,:]   + gamma_I[1:i,:].sum(axis=0)   + 0.5*gamma_I[i,:]
                GQ1i  = 0.5*gamma_Q[0,:]   + gamma_Q[1:i,:].sum(axis=0)   + 0.5*gamma_Q[i,:]
                GU1i  = 0.5*gamma_U1[0,:]  + gamma_U1[1:i,:].sum(axis=0)  + 0.5*gamma_U1[i,:]
                GV1i  = 0.5*gamma_V1[0,:]  + gamma_V1[1:i,:].sum(axis=0)  + 0.5*gamma_V1[i,:]
                GQU1i = 0.5*gamma_QU1[0,:] + gamma_QU1[1:i,:].sum(axis=0) + 0.5*gamma_QU1[i,:]
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[i] and tau[T-1] for ray 2
                # This is being calculated backwards, so this is column -(i+1) for ray 2 
                GI2i  = 0.5*gamma_I[-1,:]   + gamma_I[-i:-1,:].sum(axis=0)  + 0.5*gamma_I[-i-1,:]
                GQ2i  = 0.5*gamma_Q[-1,:]   + gamma_Q[-i:-1,:].sum(axis=0)  + 0.5*gamma_Q[-i-1,:]
                GU2i  = 0.5*gamma_U2[-1,:]  + gamma_U2[-i:-1,:].sum(axis=0) + 0.5*gamma_U2[-i-1,:]           
                GV2i  = 0.5*gamma_V2[-1,:]  + gamma_V2[-i:-1,:].sum(axis=0) + 0.5*gamma_V2[-i-1,:]         
                GQU2i = 0.5*gamma_QU2[-1,:] + gamma_QU2[-i:-1,:].sum(axis=0)+ 0.5*gamma_QU2[-i-1,:]  
                
                # Adds the ith row onto the Gamma arrays for ray 1
                GI1 = np.vstack(( GI1, GI1i ))
                GQ1 = np.vstack(( GQ1, GQ1i ))
                GU1 = np.vstack(( GU1, GU1i ))
                GV1 = np.vstack(( GV1, GV1i ))
                GQU1 = np.vstack(( GQU1, GQU1i ))
                
                # Adds the ith row onto the Gamma arrays for ray q; technically the -(i+1) index for ray 2
                #   so does a reverse stack
                GI2 = np.vstack(( GI2i, GI2 ))
                GQ2 = np.vstack(( GQ2i, GQ2 ))
                GU2 = np.vstack(( GU2i, GU2 ))
                GV2 = np.vstack(( GV2i, GV2 ))
                GQU2 = np.vstack(( GQU2i, GQU2 ))
            
            # These are only sums. Multiplies these GX1 & GX2 arrays by self.dtau 
            GI1 = GI1*self.dtau
            GQ1 = GQ1*self.dtau
            GU1 = GU1*self.dtau
            GV1 = GV1*self.dtau
            GQU1 = GQU1*self.dtau
            GI2 = GI2*self.dtau
            GQ2 = GQ2*self.dtau
            GU2 = GU2*self.dtau
            GV2 = GV2*self.dtau
            GQU2 = GQU2*self.dtau
            
            # Determines what the output array is going to be
            outstack1 = np.dstack(( GI1, GQ1, GU1, GV1, GQU1*np.ones( GI1.shape) ))
            outstack2 = np.dstack(( GI2, GQ2, GU2, GV2, GQU2*np.ones( GI2.shape) ))
            
        # Returns array of Gamma values
        if self.cloud == 1:
            return outstack
        else:
            return outstack1, outstack2
    
    def stokes_per_ray(self, delta, tauf = None, verbose=False ):
        """
        Program that calculates the dimensionless stokes values from the input dimensionless 
        inversion equations, delta.
       
        The main input for this function, delta, should be a numpy array with dimensions 
        (T,NV+4k,3), where NV is the number of velocity bins and T is the number of tau bins. The 
        0th axis specifies values across different tau at constant frequency for a single 
        transition. The 1st axis specifies values across frequency at constant tau for a transition.
        The three rows along the 0th axis should be for delta^- (delta[0]), delta^0 (delta[1]), and 
        delta^+ (delta[2]).
        
        tauf is a float that is multiplied by self.fracLOS to determine the tau values that are
        integrated over
        
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.fracLOS is the array of 
        fracLOS values that are the multiplied by tauf and integrated over.
       
        self.cloud option can be 1 or 2 depending on whether the seed radiation is entering one (1) 
        or both (2) ends of the cloud. 
        
        self.iquv0 is an array of Stokes I, Q, U, and V values at tau[0]. This should be an array of 
        length 4 of the input stokes values for the light ray. If cloud=2, this will be the initial
        of ray 1.
        
        self.iquvF is the same as iquv0 but for the far end of the cloud corresponding to tau[T-1]. 
        This is only used for cloud=2.
        
        self.k is the number of frequency bins spanned by the Zeeman shift, delta omega.
        
        
        
        Returns an array with shape (T,NV+4k,4) of unitless Stokes values. Zeroth axis separates by
        optical depth, first axis separates by frequency, and second axis separates i, q, u, v.
        
        """
        
        # If tauf is provided along with delta array, sets
        if tauf is not None:
            self.update_tauf( tauf )
    
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        delta_m = delta[:,:,0]
        delta_0 = delta[:,:,1]
        delta_p = delta[:,:,2]
        
        # Then separates out end stokes values
        i0 = self.iquv0[0]
        q0 = self.iquv0[1]
        u0 = self.iquv0[2]
        v0 = self.iquv0[3]
        if self.cloud == 2:
            iF = self.iquvF[0]
            qF = self.iquvF[1]
            uF = self.iquvF[2]
            vF = self.iquvF[3]
        
        # simplify k name
        k = self.k
        
        if verbose:
            # Sets appropriate print options
            np.set_printoptions(precision=4, linewidth=180)
            print('STOKES TEST:')
            print('    tau min:', self.tau[0], '   tau max:', self.tau[-1] )
            print('    dtau: ', self.dtau )
        
        
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
        
        
            # Prints test output if verbose
            if verbose:
                print('    gamma_I: ', gamma_I[:,verbose+k])
                
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            GI = np.zeros( gamma_I[0].shape )
            GQ = np.zeros( gamma_I[0].shape )
            GU = np.zeros( gamma_I[0].shape )
            GV = np.zeros( gamma_I[0].shape )
            GQU= np.zeros( gamma_I[0].shape )
            
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i]
                GIi  = 0.5*gamma_I[0,:]  + gamma_I[1:i,:].sum(axis=0)  + 0.5*gamma_I[i,:]
                GQi  = 0.5*gamma_Q[0,:]  + gamma_Q[1:i,:].sum(axis=0)  + 0.5*gamma_Q[i,:]
                GUi  = 0.5*gamma_U[0,:]  + gamma_U[1:i,:].sum(axis=0)  + 0.5*gamma_U[i,:]
                GVi  = 0.5*gamma_V[0,:]  + gamma_V[1:i,:].sum(axis=0)  + 0.5*gamma_V[i,:]
                GQUi = 0.5*gamma_QU[0,:] + gamma_QU[1:i,:].sum(axis=0) + 0.5*gamma_QU[i,:]
                
                GI = np.vstack(( GI, GIi ))
                GQ = np.vstack(( GQ, GQi ))
                GU = np.vstack(( GU, GUi ))
                GV = np.vstack(( GV, GVi ))
                GQU = np.vstack(( GQU, GQUi ))
                
                # Prints test output if verbose
            if verbose: # i == tau.size-1 and 
                print()
                print('    gamma_I sum: ', GI[:,verbose+k])
             
                
            # These are only sums. Multiplies these GX arrays by self.dtau before continuing
            GI = GI*self.dtau
            GQ = GQ*self.dtau
            GU = GU*self.dtau
            GV = GV*self.dtau
            GQU = GQU*self.dtau
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    GAMMA_I: ', GI[:,verbose+k])
            
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            scratch1 = np.ones( GI.shape )
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes = stokes0.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            scratch0 = np.zeros( GI.shape )
            Grow0 = np.stack(( GI,      -GQ,      -GU,      -GV ))
            Grow1 = np.stack((-GQ,       GI,      GQU, scratch0 ))
            Grow2 = np.stack((-GU,     -GQU,       GI, scratch0 ))
            Grow3 = np.stack((-GV, scratch0, scratch0,       GI ))
            Garray = np.stack(( Grow0, Grow1, Grow2, Grow3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    Gproduct = Garray.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    Gproduct = np.matmul( Gproduct, Garray, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term_n = np.matmul( Gproduct, self.iquv0, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes += ( term_n / float(factorial(n)) )
                else:
                    tempterm = ( term_n / float(factorial(170)) )
                    stokes  += tempterm * float( factorial(170)/factorial(n) )
            
            # After calculation, return stokes
            return stokes
            
             
            
            
            
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
            # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            #print('Preparing for gain matrix integration')
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            # 1 is the direction from tau[0] to tau[i] and 2 is the direction from tau[T-1] to tau[i]
            # Iterating through ray 2 backwards, so this is the last column
            GI1 = np.zeros( gamma_I[0].shape )
            GQ1 = np.zeros( gamma_I[0].shape )
            GU1 = np.zeros( gamma_I[0].shape )
            GV1 = np.zeros( gamma_I[0].shape )
            GQU1= np.zeros( gamma_I[0].shape )
            GI2 = np.zeros( gamma_I[0].shape )
            GQ2 = np.zeros( gamma_I[0].shape )
            GU2 = np.zeros( gamma_I[0].shape )
            GV2 = np.zeros( gamma_I[0].shape )
            GQU2= np.zeros( gamma_I[0].shape )
            #print('Integrating gain matrix')
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i] for ray 1
                GI1i  = 0.5*gamma_I[0,:]   + gamma_I[1:i,:].sum(axis=0)   + 0.5*gamma_I[i,:]
                GQ1i  = 0.5*gamma_Q[0,:]   + gamma_Q[1:i,:].sum(axis=0)   + 0.5*gamma_Q[i,:]
                GU1i  = 0.5*gamma_U1[0,:]  + gamma_U1[1:i,:].sum(axis=0)  + 0.5*gamma_U1[i,:]
                GV1i  = 0.5*gamma_V1[0,:]  + gamma_V1[1:i,:].sum(axis=0)  + 0.5*gamma_V1[i,:]
                GQU1i = 0.5*gamma_QU1[0,:] + gamma_QU1[1:i,:].sum(axis=0) + 0.5*gamma_QU1[i,:]
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[i] and tau[T-1] for ray 2
                # This is being calculated backwards, so this is column -(i+1) for ray 2 
                GI2i  = 0.5*gamma_I[-1,:]   + gamma_I[-i:-1,:].sum(axis=0)  + 0.5*gamma_I[-i-1,:]
                GQ2i  = 0.5*gamma_Q[-1,:]   + gamma_Q[-i:-1,:].sum(axis=0)  + 0.5*gamma_Q[-i-1,:]
                GU2i  = 0.5*gamma_U2[-1,:]  + gamma_U2[-i:-1,:].sum(axis=0) + 0.5*gamma_U2[-i-1,:]           
                GV2i  = 0.5*gamma_V2[-1,:]  + gamma_V2[-i:-1,:].sum(axis=0) + 0.5*gamma_V2[-i-1,:]         
                GQU2i = 0.5*gamma_QU2[-1,:] + gamma_QU2[-i:-1,:].sum(axis=0)+ 0.5*gamma_QU2[-i-1,:]  
                
                # Adds the ith row onto the Gamma arrays for ray 1
                GI1 = np.vstack(( GI1, GI1i ))
                GQ1 = np.vstack(( GQ1, GQ1i ))
                GU1 = np.vstack(( GU1, GU1i ))
                GV1 = np.vstack(( GV1, GV1i ))
                GQU1 = np.vstack(( GQU1, GQU1i ))
                
                # Adds the ith row onto the Gamma arrays for ray q; technically the -(i+1) index for ray 2
                #   so does a reverse stack
                GI2 = np.vstack(( GI2i, GI2 ))
                GQ2 = np.vstack(( GQ2i, GQ2 ))
                GU2 = np.vstack(( GU2i, GU2 ))
                GV2 = np.vstack(( GV2i, GV2 ))
                GQU2 = np.vstack(( GQU2i, GQU2 ))
            
            # Prints test output if verbose
            if verbose:  # i == tau.size-1 and 
                print()
                print('    gamma_I1 sum: ', GI1[:,verbose+k])
                print('    gamma_I2 sum: ', GI2[:,verbose+k])
            
            # These are only sums. Multiplies these GX1 & GX2 arrays by self.dtau 
            GI1 = GI1*self.dtau
            GQ1 = GQ1*self.dtau
            GU1 = GU1*self.dtau
            GV1 = GV1*self.dtau
            GQU1 = GQU1*self.dtau
            GI2 = GI2*self.dtau
            GQ2 = GQ2*self.dtau
            GU2 = GU2*self.dtau
            GV2 = GV2*self.dtau
            GQU2 = GQU2*self.dtau
            
            
            # Prints test output if verbose
            if verbose:
                print()
                print('    GAMMA_I1: ', GI1[:,verbose+k])
                print('    GAMMA_I2: ', GI2[:,verbose+k])
                        
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            
            # Creates some scratch arrays that can be used for both directions
            scratch1 = np.ones( GI1.shape )
            scratch0 = np.zeros( GI1.shape )
            
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            # Starts with iquv0 for direction 1
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes1 = stokes0.copy()
            # Then does iquvF for direction 2
            stokesF = np.stack((iF*scratch1, qF*scratch1, uF*scratch1, vF*scratch1))
            stokes2 = stokesF.copy()
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            # Does this first for Direction 1:
            G1row0 = np.stack(( GI1,     -GQ1,     -GU1,     -GV1 ))
            G1row1 = np.stack((-GQ1,      GI1,     GQU1, scratch0 ))
            G1row2 = np.stack((-GU1,    -GQU1,      GI1, scratch0 ))
            G1row3 = np.stack((-GV1, scratch0, scratch0,      GI1 ))
            G1array = np.stack(( G1row0, G1row1, G1row2, G1row3 ))
            # Then does the same for direction 2:
            G2row0 = np.stack(( GI2,     -GQ2,     -GU2,     -GV2 ))
            G2row1 = np.stack((-GQ2,      GI2,     GQU2, scratch0 ))
            G2row2 = np.stack((-GU2,    -GQU2,      GI2, scratch0 ))
            G2row3 = np.stack((-GV2, scratch0, scratch0,      GI2 ))
            G2array = np.stack(( G2row0, G2row1, G2row2, G2row3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    G1product = G1array.copy()
                    G2product = G2array.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    G1product = np.matmul( G1product, G1array, axes=[(0,1),(0,1),(0,1)] )
                    G2product = np.matmul( G2product, G2array, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term1_n = np.matmul( G1product, self.iquv0, axes=[(0,1),(0,),(0,)])
                term2_n = np.matmul( G2product, self.iquvF, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    stokes1 += ( term1_n / float(factorial(n)) )
                    stokes2 += ( term2_n / float(factorial(n)) )
                else:
                    tempterm1 = term1_n / float(factorial(170))
                    tempterm2 = term2_n / float(factorial(170))
                    stokes1 += tempterm1 * float( factorial(170)/factorial(n) )
                    stokes2 += tempterm2 * float( factorial(170)/factorial(n) )
            
            # After calculation, unpacks stokes
            return stokes1, stokes2
    
    def LDI_terms(self, delta, tauf = None, verbose=False ):
        """
        Program that calculates the dimensionless stokes values from the input dimensionless 
        inversion equations, delta.
       
        The main input for this function, delta, should be a numpy array with dimensions 
        (T,NV+4k,3), where NV is the number of velocity bins and T is the number of tau bins. The 
        0th axis specifies values across different tau at constant frequency for a single 
        transition. The 1st axis specifies values across frequency at constant tau for a transition.
        The three rows along the 0th axis should be for delta^- (delta[0]), delta^0 (delta[1]), and 
        delta^+ (delta[2]).
        
        tauf is a float that is multiplied by self.fracLOS to determine the tau values that are
        integrated over
        
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.fracLOS is the array of 
        fracLOS values that are the multiplied by tauf and integrated over.
       
        self.cloud option can be 1 or 2 depending on whether the seed radiation is entering one (1) 
        or both (2) ends of the cloud. 
        
        self.iquv0 is an array of Stokes I, Q, U, and V values at tau[0]. This should be an array of 
        length 4 of the input stokes values for the light ray. If cloud=2, this will be the initial
        of ray 1.
        
        self.iquvF is the same as iquv0 but for the far end of the cloud corresponding to tau[T-1]. 
        This is only used for cloud=2.
        
        self.k is the number of frequency bins spanned by the Zeeman shift, delta omega.
        
        
        
        Returns an array with shape (T,NV+4k,4) of unitless Stokes values. Zeroth axis separates by
        optical depth, first axis separates by frequency, and second axis separates i, q, u, v.
        
        """
        
        # If tauf is provided along with delta array, sets
        if tauf is not None:
            self.update_tauf( tauf )
    
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        delta_m = delta[:,:,0]
        delta_0 = delta[:,:,1]
        delta_p = delta[:,:,2]
        
        # Then separates out end stokes values
        i0 = self.iquv0[0]
        q0 = self.iquv0[1]
        u0 = self.iquv0[2]
        v0 = self.iquv0[3]
        if self.cloud == 2:
            iF = self.iquvF[0]
            qF = self.iquvF[1]
            uF = self.iquvF[2]
            vF = self.iquvF[3]
        
        # simplify k name
        k = self.k
        
        if verbose:
            # Sets appropriate print options
            np.set_printoptions(precision=4, linewidth=180)
            print('STOKES TEST:')
            print('    tau min:', self.tau[0], '   tau max:', self.tau[-1] )
            print('    dtau: ', self.dtau )
        
        
        
        # Splits here for 1 directional integration versus 2-directional ray
        # First for a 1-directional ray
        if self.cloud == 1:
            
            # First calculates unitless gain matrix components. These arrays have shape (T,NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            gamma_U = A * self.sintwophi
            gamma_V = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
        
        
            # Prints test output if verbose
            if verbose:
                print('    gamma_I: ', gamma_I[:,verbose+k])
                
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            GI = np.zeros( gamma_I[0].shape )
            GQ = np.zeros( gamma_I[0].shape )
            GU = np.zeros( gamma_I[0].shape )
            GV = np.zeros( gamma_I[0].shape )
            GQU= np.zeros( gamma_I[0].shape )
            
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i]
                GIi  = 0.5*gamma_I[0,:]  + gamma_I[1:i,:].sum(axis=0)  + 0.5*gamma_I[i,:]
                GQi  = 0.5*gamma_Q[0,:]  + gamma_Q[1:i,:].sum(axis=0)  + 0.5*gamma_Q[i,:]
                GUi  = 0.5*gamma_U[0,:]  + gamma_U[1:i,:].sum(axis=0)  + 0.5*gamma_U[i,:]
                GVi  = 0.5*gamma_V[0,:]  + gamma_V[1:i,:].sum(axis=0)  + 0.5*gamma_V[i,:]
                GQUi = 0.5*gamma_QU[0,:] + gamma_QU[1:i,:].sum(axis=0) + 0.5*gamma_QU[i,:]
                
                GI = np.vstack(( GI, GIi ))
                GQ = np.vstack(( GQ, GQi ))
                GU = np.vstack(( GU, GUi ))
                GV = np.vstack(( GV, GVi ))
                GQU = np.vstack(( GQU, GQUi ))
                
                # Prints test output if verbose
            if verbose: # i == tau.size-1 and 
                print()
                print('    gamma_I sum: ', GI[:,verbose+k])
             
                
            # These are only sums. Multiplies these GX arrays by self.dtau before continuing
            GI = GI*self.dtau
            GQ = GQ*self.dtau
            GU = GU*self.dtau
            GV = GV*self.dtau
            GQU = GQU*self.dtau
            
            # Prints test output if verbose
            if verbose:
                # Sets appropriate print options
                print()
                print('    GAMMA_I: ', GI[:,verbose+k])
            
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            scratch1 = np.ones( GI.shape )
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes = stokes0.copy().reshape( 1, *stokes0.shape  )
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            scratch0 = np.zeros( GI.shape )
            Grow0 = np.stack(( GI,      -GQ,      -GU,      -GV ))
            Grow1 = np.stack((-GQ,       GI,      GQU, scratch0 ))
            Grow2 = np.stack((-GU,     -GQU,       GI, scratch0 ))
            Grow3 = np.stack((-GV, scratch0, scratch0,       GI ))
            Garray = np.stack(( Grow0, Grow1, Grow2, Grow3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    Gproduct = Garray.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    Gproduct = np.matmul( Gproduct, Garray, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term_n = np.matmul( Gproduct, self.iquv0, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    term = ( term_n / float(factorial(n)) )
                else:
                    tempterm = ( term_n / float(factorial(170)) )
                    term  = tempterm * float( factorial(170)/factorial(n) )
                term = term.reshape( 1, *term_n.shape )
                stokes = np.vstack(( stokes, term ))
            
            # After calculation, return stokes
            return stokes
            
             
            
            
            
        # Otherwise, if this is a 2-dimensional integration
        elif self.cloud == 2:
            
            # First calculates unitless gain matrix components
            # gamma_I and gamma_Q terms same in both directions; calculates those first. Array shape (T, NV+2k)
            gamma_I = 2.*delta_0[:,k:-k] * self.sintheta**2 + (1.+self.costheta**2) * \
                                         ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] )
            A = - ( self.etap * delta_p[:,2*k:] + self.etam * delta_m[:,:-2*k] - 2. * delta_0[:,k:-k] ) \
                                                                                    * self.sintheta**2
            gamma_Q = A * self.costwophi
            
            
            # Calculates gamma_u, gamma_v, and gamma_qu for Ray 1. These arrays have shape (T,NV+2k)
            gamma_U1 = A * self.sintwophi
            gamma_V1 = 2. * (self.etap * delta_p[:,2*k:] - self.etam * delta_m[:,:-2*k] )*self.costheta
            gamma_QU1 = - self.far_coeff * self.costheta * np.ones( gamma_I.shape )
            
            # Then calculates gamma_u, gamma_v, and gamma_qu for Ray 2. These arrays have shape (T,NV+2k)
            gamma_U2 = - 1.0 * gamma_U1
            gamma_V2 = - 1.0 * gamma_V1
            gamma_QU2 = -1.0 * gamma_QU1
            
            #print('Preparing for gain matrix integration')
            # Begins iterating through tau for the sums
            # Starts with tau=0, which should be 0 because there's no width to be integrated yet
            # 1 is the direction from tau[0] to tau[i] and 2 is the direction from tau[T-1] to tau[i]
            # Iterating through ray 2 backwards, so this is the last column
            GI1 = np.zeros( gamma_I[0].shape )
            GQ1 = np.zeros( gamma_I[0].shape )
            GU1 = np.zeros( gamma_I[0].shape )
            GV1 = np.zeros( gamma_I[0].shape )
            GQU1= np.zeros( gamma_I[0].shape )
            GI2 = np.zeros( gamma_I[0].shape )
            GQ2 = np.zeros( gamma_I[0].shape )
            GU2 = np.zeros( gamma_I[0].shape )
            GV2 = np.zeros( gamma_I[0].shape )
            GQU2= np.zeros( gamma_I[0].shape )
            #print('Integrating gain matrix')
            # Then begins iterating through remaining tau bins i using trapezoidal rule for numerical integ
            for i in range( 1, self.tau.size ):
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[0] and tau[i] for ray 1
                GI1i  = 0.5*gamma_I[0,:]   + gamma_I[1:i,:].sum(axis=0)   + 0.5*gamma_I[i,:]
                GQ1i  = 0.5*gamma_Q[0,:]   + gamma_Q[1:i,:].sum(axis=0)   + 0.5*gamma_Q[i,:]
                GU1i  = 0.5*gamma_U1[0,:]  + gamma_U1[1:i,:].sum(axis=0)  + 0.5*gamma_U1[i,:]
                GV1i  = 0.5*gamma_V1[0,:]  + gamma_V1[1:i,:].sum(axis=0)  + 0.5*gamma_V1[i,:]
                GQU1i = 0.5*gamma_QU1[0,:] + gamma_QU1[1:i,:].sum(axis=0) + 0.5*gamma_QU1[i,:]
                
                # Calculates sums for that particular value of tau of the values that come between
                #     tau[i] and tau[T-1] for ray 2
                # This is being calculated backwards, so this is column -(i+1) for ray 2 
                GI2i  = 0.5*gamma_I[-1,:]   + gamma_I[-i:-1,:].sum(axis=0)  + 0.5*gamma_I[-i-1,:]
                GQ2i  = 0.5*gamma_Q[-1,:]   + gamma_Q[-i:-1,:].sum(axis=0)  + 0.5*gamma_Q[-i-1,:]
                GU2i  = 0.5*gamma_U2[-1,:]  + gamma_U2[-i:-1,:].sum(axis=0) + 0.5*gamma_U2[-i-1,:]           
                GV2i  = 0.5*gamma_V2[-1,:]  + gamma_V2[-i:-1,:].sum(axis=0) + 0.5*gamma_V2[-i-1,:]         
                GQU2i = 0.5*gamma_QU2[-1,:] + gamma_QU2[-i:-1,:].sum(axis=0)+ 0.5*gamma_QU2[-i-1,:]  
                
                # Adds the ith row onto the Gamma arrays for ray 1
                GI1 = np.vstack(( GI1, GI1i ))
                GQ1 = np.vstack(( GQ1, GQ1i ))
                GU1 = np.vstack(( GU1, GU1i ))
                GV1 = np.vstack(( GV1, GV1i ))
                GQU1 = np.vstack(( GQU1, GQU1i ))
                
                # Adds the ith row onto the Gamma arrays for ray q; technically the -(i+1) index for ray 2
                #   so does a reverse stack
                GI2 = np.vstack(( GI2i, GI2 ))
                GQ2 = np.vstack(( GQ2i, GQ2 ))
                GU2 = np.vstack(( GU2i, GU2 ))
                GV2 = np.vstack(( GV2i, GV2 ))
                GQU2 = np.vstack(( GQU2i, GQU2 ))
            
            # Prints test output if verbose
            if verbose:  # i == tau.size-1 and 
                print()
                print('    gamma_I1 sum: ', GI1[:,verbose+k])
                print('    gamma_I2 sum: ', GI2[:,verbose+k])
            
            # These are only sums. Multiplies these GX1 & GX2 arrays by self.dtau 
            GI1 = GI1*self.dtau
            GQ1 = GQ1*self.dtau
            GU1 = GU1*self.dtau
            GV1 = GV1*self.dtau
            GQU1 = GQU1*self.dtau
            GI2 = GI2*self.dtau
            GQ2 = GQ2*self.dtau
            GU2 = GU2*self.dtau
            GV2 = GV2*self.dtau
            GQU2 = GQU2*self.dtau
            
            
            # Prints test output if verbose
            if verbose:
                print()
                print('    GAMMA_I1: ', GI1[:,verbose+k])
                print('    GAMMA_I2: ', GI2[:,verbose+k])
                        
            # Now that integrated Gammas are calculated, calculates the resulting unitless stokes 
            #     values at tau.
            
            # Creates some scratch arrays that can be used for both directions
            scratch1 = np.ones( GI1.shape )
            scratch0 = np.zeros( GI1.shape )
            
            # Begins by setting stokes i as initial values stokes0, i.e. the n=0 terms constant over 
            #    optical depth and frequency. Array shape is (4,T,NV+2k)
            # Starts with iquv0 for direction 1
            stokes0 = np.stack((i0*scratch1, q0*scratch1, u0*scratch1, v0*scratch1))
            stokes1 = stokes0.copy().reshape( 1, *stokes0.shape  )
            # Then does iquvF for direction 2
            stokesF = np.stack((iF*scratch1, qF*scratch1, uF*scratch1, vF*scratch1))
            stokes2 = stokesF.copy().reshape( 1, *stokes0.shape  )
            
            # Creates integrated Gamma array. Has shape (4,4,T,NV+2k)
            # Does this first for Direction 1:
            G1row0 = np.stack(( GI1,     -GQ1,     -GU1,     -GV1 ))
            G1row1 = np.stack((-GQ1,      GI1,     GQU1, scratch0 ))
            G1row2 = np.stack((-GU1,    -GQU1,      GI1, scratch0 ))
            G1row3 = np.stack((-GV1, scratch0, scratch0,      GI1 ))
            G1array = np.stack(( G1row0, G1row1, G1row2, G1row3 ))
            # Then does the same for direction 2:
            G2row0 = np.stack(( GI2,     -GQ2,     -GU2,     -GV2 ))
            G2row1 = np.stack((-GQ2,      GI2,     GQU2, scratch0 ))
            G2row2 = np.stack((-GU2,    -GQU2,      GI2, scratch0 ))
            G2row3 = np.stack((-GV2, scratch0, scratch0,      GI2 ))
            G2array = np.stack(( G2row0, G2row1, G2row2, G2row3 ))
            
            # Begins looping through n = 1-10 to calculate and add terms onto stokes array
            for n in range(1,self.n + 1):
                
                # If this is the first loop, just sets gamma product to be a copy of Garray
                if n == 1:
                    G1product = G1array.copy()
                    G2product = G2array.copy()
                
                # Otherwise, multiplies Garray by Gproduct again
                else:
                    G1product = np.matmul( G1product, G1array, axes=[(0,1),(0,1),(0,1)] )
                    G2product = np.matmul( G2product, G2array, axes=[(0,1),(0,1),(0,1)] )
                
                # Multiplies the Gproduct by the incoming radiation seed, stokes0/stokesF
                term1_n = np.matmul( G1product, self.iquv0, axes=[(0,1),(0,),(0,)])
                term2_n = np.matmul( G2product, self.iquvF, axes=[(0,1),(0,),(0,)])
                
                # Divides that term by n! and adds result onto running stokes calculation, stokes
                if n <= 170:
                    term1 = ( term1_n / float(factorial(n)) ).reshape( 1, *term1_n.shape )
                    term2 = ( term2_n / float(factorial(n)) ).reshape( 1, *term2_n.shape )
                else:
                    tempterm1 = ( term1_n / float(factorial(170)) ).reshape( 1, *term1_n.shape )
                    tempterm2 = ( term2_n / float(factorial(170)) ).reshape( 1, *term2_n.shape )
                    term1 = tempterm1 * float( factorial(170)/factorial(n) )
                    term2 = tempterm2 * float( factorial(170)/factorial(n) )
                
                stokes1 = np.vstack(( stokes1, term1 ))
                stokes2 = np.vstack(( stokes2, term2 ))
                
            # After calculation, unpacks stokes
            return stokes1, stokes2



############################# Object class for parameter set with varying theta #############################


class maser_v_theta(_maser_base_):
    def __init__(self, parfile = None, thetas = _default_( None ), theta_precision = _default_( 1 ), \
                       units = _default_('degrees'), outpaths = _default_(None), **kwargs):
        """
        Object for calculating the dimensionless population inversions for a given parameter set
        *as a function of theta*. 
        
        Generates one maser class object for each value of theta provided, and stores them in the dictionary
        attribute self.masers, with the value of theta as the dictionary key (rounded to the specified 
        precision).
        
        Optional Parameters:
            
            parfile         String or None
                                
                                [ Default = None ]
                                
                                If provided, gives the path and file name (from current directory) of
                                a parameter file containing values for any of the keywords accepted 
                                by this object class initialization. Values in this parameter file 
                                will override any default values. 
                                
                                Parameter file ingestion also allows the specification of the
                                omegabar array by min, max, and stepsize, as well as the
                                specification of fracLOS by number of resolution elements (both of which
                                are not currently supported when set explicitly on object 
                                initialization.)
            
            thetas          List or NumPy Array of floats (or None)
                                
                                [ Default = None ]
            
                                List of values for theta (in the units specified by the units keyword) 
                                for which to generate a maser object with the specified other parameters.
                                
                                Can also be read in from the parameter file, but must be under the section
                                header '[MASER_V_THETA CLASS]'. To read in from the parameter file, enter
                                None on object initialization and specify a parfile with the desired
                                value. 
                                
                                Note: This parameter MUST be specified either directly on object 
                                initialization or from a parameter file.
                                
            theta_precision Integer
                                
                                [ Default = 1 ]
                                
                                The precision to which thetas values will be rounded when used as keys 
                                for the masers dictionary attribute. (Keep in mind, these values will be
                                in the same units used on object initialization.)
            
            units           String: 'degrees' (or 'deg' or 'd') OR 'radians' (or 'rad' or 'r')
                                
                                [ Default = 'degrees' ]
                                
                                The units in which the thetas values are provided. The masers dictionary
                                attribute will use the values corresponding to these units, but they
                                they will be converted to radians for initializing the individual maser
                                objects.
            
            outpaths        List of Strings
                                
                                [ Default = None ]
                                
                                The output paths to which each of the thetas solutions will be mapped.
                                If provided, must be the same length as thetas. If not provided, will
                                be generated based on the thetas values provided.
            
            **kwargs        Any additional keyword arguments will be passed directly on to the 
                            initialization of every maser object.
        
        
        Object attributes:
            
            All attributes set by _maser_base_ class, plus:
            
            self.thetas             NumPy Array
                                
                                        Otherwise, as entered and rounded to the precision in 
                                        thetas_precision.
            
            self.theta_precision    Integer
                                        
                                        As entered
            
            self.units              String
                                        
                                        As entered
            
            self.masers             Dictionary
                                
                                        Keys are the values listed in thetas, and the values are the
                                        corresponding maser objects.
            
        """
        
        
        
        
        #### Uses _maser_base_ initialization to load parfile (if any) and the following attributes:
        ####     phi, n, outpath, far_coeff, etapm, alphapm, cloud, taufs, resume, lastdelta, verbose, ftol, filename,
        ####     endfill, trend, lastdelta2
        ####     + fccalc, sintwophi, costwophi
        #### Saves config file as attribute conf, name of parfile as attribute parfile, and name of base section in
        ####     config file as attribute sect
        super().__init__( parfile = parfile, ignore = ['theta'], **kwargs)
        
            
        
        #### Processing keys for this class specifically
        sect = 'MASER_V_THETA CLASS'
        
        
        
        
        #### Sets up thetas attribute with rounding
        
        # Sets the precision for theta attribute
        self.theta_precision = self._process_key_( 'theta_precision', theta_precision, self.conf[sect], \
                                                        allowed = { int: None }, convert = True, ignore_none = True  )
                                                        
        
        # Checks data type of thetas - list or numpy array; if it's a single value, sets but prints warning
        if isinstance( thetas, float ) or isinstance( thetas, int):
            print('Warning: The maser_v_theta class is intended for comparison of maser class objects with different values\n' + \
                  '         of theta and otherwise identical parameters. For a single parameter set, recommend using the\n' + \
                  '         maser class, instead.')
            thetas = np.array([ round(float(thetas),theta_precision), ])
        self.thetas = self._process_key_( 'thetas', thetas, self.conf[sect], \
                                        allowed = { np.ndarray: [float, None] }, convert = True, ignore_none = True  )
        if self.thetas is None:
            raise ValueError( 'Keyword thetas must be provided either on call or in parameter file.' )
        
        # Rounds all values in array
        self.thetas = np.array([ round(x, self.theta_precision ) for x in self.thetas ])
            
        
        #### Checking and saving units
        self.units = self._process_key_( 'units', units, self.conf[sect], \
                    allowed = { str: ['radians','rad','r','degrees','deg','d'] }, convert = False, ignore_none = True  )
        
        
        #### Outpaths array
        self.outpaths = self._process_key_( 'outpaths', outpaths, self.conf[sect], \
                        allowed = OrderedDict([ (list, [str, self.thetas.size]), (None,None) ]), \
                        convert = False, ignore_none = True  )
        if self.outpaths is None:
            if self.units in ['degrees','deg','d']:
                ndigits = self.theta_precision + 2
            else:
                ndigits = self.theta_precision + 1
            path_template = '{{0}}theta{{1:0>{0}}}'.format(ndigits)
            self.outpaths = [ path_template.format( self.outpath, round(th * 10**self.theta_precision) ) for th in self.thetas ]
        
        
        #### Makes masers dictionary and populates
        self.masers = {}
        
        # Makes sure there is no 'theta' key in the **kwargs
        for key in ['theta', 'outpath']:
            if key in kwargs:
                del kwargs[key]
        
        # Makes array of theta values in radians corresponding to each value in thetas; if units are radians, this 
        #   is identical to the self.thetas array
        if self.units in ['degrees','deg','d']:
            thetas_radians = self.thetas * pi / 180.
        else:
            thetas_radians = self.thetas.copy()
        
        # theta value in self.thetas already rounded
        for i, theta in enumerate(self.thetas):
            
            # Makes maser object
            self.masers[ theta ] = maser( parfile = parfile, theta = thetas_radians[i], outpath = self.outpaths[i], **kwargs )
        
        
        #### Some extra work on taufs attribute, setting tauf, tau, and dtau
        
        # If taufs is an array, sets tauf, tau, and dtau attributes based on first value in array
        if isinstance( self.taufs, np.ndarray ):
            self.update_tauf( self.taufs[0] )
        
        # If taufs is None, sets tauf, tau, and dtau attributes
        elif self.taufs is None:
            self.update_tauf( None )
        
    def calc_far_coeff(self, ne, freq0, Gam, B, A0, P0, mode='cm' ):
        """
        Calculates the faraday coefficient, gamma_QU/cos(theta) given:
            ne      = electron density [ cm^-3 or m^-3 ]
            freq0   = rest frequency of the line [ Hz ]
            Gamma     = loss rate [ s^-1 ]
            B       = magnetic field strength [Gauss]
            A0      = Einstein A coefficient [ s^-1 ]
            P0      = pump rate into the 0 state [cm^-3 s^-1 or m^-3 s^-1 ]
        Also uses the Doppler width in Hz given in object initialization.
        
        Keyword mode can be set to 'cm' or 'm' to specify units of the given ne and P0 values.
        Default is 'cm'. If set to 'cm', these values will be converted to SI prior to calculation.
        
        Overwrites self.far_coeff.
        """
        # Uses _maser_base_ class method, but hands it the doppler width, W, from the attribute
        super().calc_far_coeff( ne, freq0, Gam, B, A0, P0, self.W, mode=mode )
        
        # Iterates the new faraday coeff values down to the individual maser objects
        for theta in self.thetas:
            for key in ['far_coeff','fccalc','ne','P0','freq0','Gamma','B','A0']:
                self.masers[ theta ].__dict__[key] = self.__dict__[key]
    
    def update_tauf( self, tauf ):
        """
        Updates tauf value (i.e. the total optical depth of the cloud multiplied by fracLOS).
        
        Updates object attributes self.tauf, self.tau, and self.dtau.
        """
        
        if tauf is not None:
        
            # Saves new tauf value as object attribute tauf
            self.tauf = tauf
        
            # Scales tau array appropriately
            self.tau = self.fracLOS * self.tauf
    
            # Determines the spacing in tau
            self.dtau = self.tau[1] - self.tau[0]
        
        else:
            self.tauf = None
            self.tau  = None
            self.dtau = None
        
        # Sets for all lower level maser objects
        for theta in self.thetas:
            for key in ['tauf','tau','dtau']:
                self.masers[ theta ].__dict__[key] = self.__dict__[key]
    
    
    ### Functions for analysis ###
            
    def readin(self, tauf, as_attr, ext='fits', updatepars = False ): 
        """
        Program to read in files generated by iterative root finding for each maser object in the masers 
        dictionary attribute and saves them as an object attribute of the individual maser objects of 
        name as_attr. 
        
        For example, to load the tauf = 2.0 delta arrays (saved as fits files) as the lastdelta
        attributes for each object, run
            
            self.readin( 2.0, 'lastdelta', ext='fits', updatepars = False )
        
        Note: previous versions of this code called tauf 'beta'. To provide backwards compatibility, 
        when reading files, this code will look first for files with the new 'tauf' naming convention, 
        but will look for any with the old 'beta' naming convention if those are not found.
        
        Required Parameters:
            
            tauf            Float
                                
                                Value or an array of total optical depths for the cloud. Unitless.
                                Indicates which solution should be read in from the output path.
            
            as_attr         String
                                
                                The name of the attribute of each maser object in the masers dictionary
                                to which the corresponding deltas array will be saved.
            
        Optional Parameters:
            
            ext             String ('txt' or 'fits')
                                
                                [ Default = 'fits' ]
                                
                                The extension of the file to be read in.
                                
                                Recommend: 'fits'
            
            updatepars      Boolean
                                
                                [ Default = False ]
                                
                                Only used if ext = 'fits'. If set to True, will read the additional
                                parameters saved to the fits file and saves them to all of the 
                                corresponding object attributes for each maser object.
                                
        """
        
        # Makes sure . not provided in requested extension
        if ext.startswith('.'):
            ext = ext[1:]
        
        
        # Starts by iterating through all theta values
        for theta in self.thetas:
            
            # Reading in if text file
            if ext == 'txt':
        
                # Determines path names for each delta using desired extension
                dminus_path = '{0}{1}_tauf{2}_dminus.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, tauf, ext )
                dzero_path  = '{0}{1}_tauf{2}_dzero.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, tauf, ext )
                dplus_path  = '{0}{1}_tauf{2}_dplus.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, tauf, ext )
                
                # If files don't exist with new tauf naming convention, uses old beta naming convention
                if not os.path.exists( dminus_path ):
                    dminus_path = dminus_path.replace( 'tauf', 'beta' )
                if not os.path.exists( dzero_path ):
                    dzero_path = dzero_path.replace( 'tauf', 'beta' )
                if not os.path.exists( dplus_path ):
                    dplus_path = dplus_path.replace( 'tauf', 'beta' )
                    
                
                # Reads in files
                dminus = np.genfromtxt( dminus_path )
                dzero  = np.genfromtxt( dzero_path )
                dplus  = np.genfromtxt( dplus_path )
            
        
            # Reading in if fits file
            elif ext == 'fits':
        
                # Determines path names for single fits file
                outpath = '{0}{1}_tauf{2}.{3}'.format(self.masers[theta].outpath, self.filename, tauf, ext )
                if not os.path.exists( outpath ):
                    outpath = outpath.replace( 'tauf', 'beta' )
            
                # Opens fits file for reading
                hdu = fits.open( outpath )
            
                # Gets delta arrays from extensions
                dminus = hdu[1].data
                dzero  = hdu[2].data
                dplus  = hdu[3].data
            
                # Updates object attributes from header if updatepars requested
                # Does not overwrite outpath, verbose, resume, trend, lastdelta, or lastdelta2.
                if updatepars:
                
                    # Sets aside 0-ext header for easy ref
                    hdr = hdu[0].header
                
                    # Reconstruct omegabar array from header keys. Assumes omegabar is centered on omega_0
                    AFbins = hdr['AFbins']
                    dAF = hdr['AFres']
                    Nplus = ( AFbins - 1 ) / 2
                    self.masers[theta].omegabar = np.linspace( -Nplus, Nplus, AFbins ).astype(np.longdouble) * dAF
                
                    # Reconstructs fracLOS assuming fracLOS is fraction of cloud transversed from 0 to 1
                    self.masers[theta].fracLOS = np.linspace( 0, 1, hdr['taubins'] ).astype(np.longdouble)
                
                    # Retrieves theta
                    self.masers[theta].theta = hdr['theta']
                
                    # Reconstructs iquv0
                    self.masers[theta].iquv0 = np.array([ hdr['i0'], hdr['q0'], hdr['u0'], hdr['v0'] ])
                
                    # Sets W, k, phi, and farcoeff directly
                    self.masers[theta].W = hdr['Doppler']
                    self.masers[theta].k = hdr['k']
                    self.masers[theta].phi = hdr['phi']
                    self.masers[theta].far_coeff = hdr['farcoeff']
                
                    # Sets eta p/m and alpha p/m
                    self.masers[theta].etap = hdr['etap']
                    self.masers[theta].etam = hdr['etam']
                    self.masers[theta].alphap = hdr['alphap']
                    self.masers[theta].alpham = hdr['alpham']
                
                    # Saves n and endfill
                    self.masers[theta].n = hdr['nexp']
                    self.masers[theta].endfill = hdr['endfill']
                
                    # Sets cloud and ray2 stuff
                    self.masers[theta].cloud = hdr['cloud']
                    if self.masers[theta].cloud == 2:
                        self.masers[theta].iquvF = np.array([ hdr['iF'], hdr['qF'], hdr['uF'], hdr['vF'] ])
                    else:
                        self.masers[theta].iquvF = None
                
                    # Saves tauf and ftol
                    self.masers[theta].ftol = hdr['ftol']
                
                    # Gets fcalc info if in the header
                    if 'ne' in hdr.keys():
                        self.masers[theta].fccalc = True
                        self.masers[theta].ne = hdr['ne']
                        self.masers[theta].P0 = hdr['P0']
                        self.masers[theta].freq0 = hdr['AF0']
                        self.masers[theta].Gamma = hdr['Gamma']
                        self.masers[theta].B = hdr['B']
                        self.masers[theta].A0 = hdr['A0']
                    
                    # Otherwise, assumes fccoeff set manually
                    else:
                        self.masers[theta].fccalc = False
                
                    # Updates sin and cos
                    self.masers[theta].sintheta = sin(self.masers[theta].theta)
                    self.masers[theta].costheta = cos(self.masers[theta].theta)
                    self.masers[theta].sintwophi = sin(2.*self.masers[theta].phi)
                    self.masers[theta].costwophi = cos(2.*self.masers[theta].phi)
            
                # Closes fits file
                hdu.close()
            
            
            # Updates tauf
            self.update_tauf( float(tauf) )
        
            # Sets as requested attribute
            self.masers[theta].__dict__[as_attr] = np.dstack(( dminus, dzero, dplus ))
        
    def cloud_end_stokes( self, taufs, ext='fits', tau_idx = -1, saveas = None, overwrite = False, verbose = True ):
        """
        Calculates the dimensionless stokes values, fractional polarizations, and EVPA at the end of
        the cloud for a variety of total optical depths, tauf, for every maser object in the masers
        attribute dictionary.
        
        Prior to calling, the following attributes of the maser objects in maser should be set/up to 
        date:
            
            far_coeff       Float
                                Unitless value gives -gamma_QU / cos(theta). Can either be specified 
                                explicitly on object initialization, or calculated from components 
                                using the calc_far_coeff method. *IS* updated if the updatepars
                                option is used when readin in a deltas solution from a fits file.
                                
        
        Other maser attributes used by this method that are set on object initialization:
        
            theta, costheta, sintheta, phi, costwophi, sintwophi, etap, etam, fracLOS, cloud, iquv0, 
            iquvF, k, filename, outpath
            
        Required Parameters:
            
            taufs           1D NumPy array
                                The values of total optical depth, tauf, for which the stokes
                                values are desired. Solution files should already exist for all
                                specified values in the outpath, of file extension indicated by
                                ext. (Of size B, for comparison with array attributes set below.)
        
        Optional Parameters:
            
            ext             String ('fits' or 'txt')
                                [ Default = 'fits' ]
                                The file extension of the output inversion solutions in the 
                                outpath. All tauf solutions should use the same file extension.
            
            tau_idx         Integer
                                [ Default = -1 ]
                                The index within the cloud at which the solution will be retained. 
                                To use the solution at the far end of the cloud (i.e. where the
                                ray exits the cloud in a single ray solution), use tau_idx = -1. 
                                To use the solution at the near end of the cloud (i.e. where the
                                ray enters the cloud in a single ray solution), use tau_idx = 0.
                                Latter should only matter if a 2-ray solution with different 
                                rays.
            
            saveas          String or None
                                [ Default = None ]
                                If provided as string, will save the produced stokes and polarization
                                arrays to files in the outpath, with the file name given in the string.
                                Results will be saved as a fits file. String name given by saveas does
                                not need to end in a '.fits' suffix.
                                The resulting fits file will have 8 extensions - the 0th extension
                                contains a header with basic information on the results, the tau bin,
                                and the values of tauf, while the remaining 7 extensions contain the 
                                data arrays for stokes i, stokes q, stokes u, stokes v, mc, ml, and
                                evpa, respectively.  
            
            overwrite       Boolean
                                [ Default = False ]
                                Whether to overwrite any existing fits file with the same name when 
                                creating the output (True) or not (False). Used only if saveas is not
                                None.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Object Attributes Created/Updated by This Method:
            
            tau_idx         Integer
                                Saved directly from user input.
                                
            stacked_stoki   NumPy array of shape (B,NV-2k)
                                Unitless Stokes i at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stokq   NumPy array of shape (B,NV-2k)
                                Unitless Stokes q at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stoku   NumPy array of shape (B,NV-2k)
                                Unitless Stokes u at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_stokv   NumPy array of shape (B,NV-2k)
                                Unitless Stokes v at the end of a cloud at each optical depth as a 
                                function of frequency.
            
            stacked_mc      NumPy array of shape (B,NV-2k)
                                Fractional circular polarization at the end of a cloud at each 
                                optical depth as a function of frequency. Calculated as the ratio 
                                Stokes v / i. Does preserve sign of circular polarization.
            
            stacked_ml      NumPy array of shape (B,NV-2k)
                                Fractional linear polarization at the end of a cloud at each optical 
                                depth as a function of frequency. Does not preserve the direction of 
                                linear polarization.
            
            stacked_evpa    NumPy array of shape (B,NV-2k)
                                Electric vector position angle of the linear polarization at each
                                point in the solution grid. Calculated as 0.5 * arctan2( u, q ).
                                
        """
        
        # Overrites taufs object with provided attribute for top-level object and maser objects
        self.taufs = np.array( taufs ).astype(float)
        for theta in self.thetas:
            self.masers[ theta ].taufs = self.taufs.copy()
        
        # Saves high level tau_idx (individual maser objects will be updated by their cloud_end_stokes method)
        self.tau_idx = tau_idx
        
        # Begins iterating through theta objects to read in and calculate all stokes values
        for theta in self.thetas:
            if verbose:
                print('Loading results for theta = {0} {1}...'.format(theta, self.units))
        
            # Just hands the info to the prism maser object's cloud_end_stokes method
            self.masers[theta].cloud_end_stokes( taufs, ext = ext, tau_idx = tau_idx, saveas = saveas, \
                                                 overwrite = overwrite, verbose = verbose )
    
    def read_cloud_end_stokes( self, filename, verbose = True ):
        """
        Reads in fits files created by method cloud_end_stokes using saveas option and updates attributes of
        the maser objects in the masers attribute dictionary accordingly. For each maser object, saves the 
        optical depth index along tau at which stokes values in file were calculated to the individual maser
        objects in the masers dictionary as attribute tau_idx.
        
        Required Parameters:
            
            filename        String
                                Name of the fits file to which the calculated stokes values were saved by
                                cloud_end_stokes method. Assumed to be in path specified by object attribute
                                outpath for each maser object.
        
        Optional Parameters:
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Updates Attributes:
            
            taufs
            stacked_stokesi
            stacked_stokesq
            stacked_stokesu
            stacked_stokesv
            stacked_mc
            stacked_ml
            stacked_evpa
            tau_idx
        
        
        Other Functionality:
            
            Checks other attributes stored in fits file header against object attributes, and raises a 
            warning if there is disagreement (but will still run).
        
        """
        
        # Begins iterating through theta objects to read in all stokes values
        for i, theta in enumerate(self.thetas):
            if verbose:
                print('Loading results for theta = {0} {1}...'.format(theta, self.units))
        
            # Just hands the info to the prism maser object's read_cloud_end_stokes method and retrieves tau_idx
            self.masers[theta].read_cloud_end_stokes( filename, verbose = verbose )
            
            # If this is the first value of theta, saves tau index to top level object attribute
            if i == 0:
                self.tau_idx = self.masers[theta].tau_idx
            
            # If not, checks if the new tau_idx is consistent with that set by the file for the first theta
            elif self.tau_idx != self.masers[theta].tau_idx:
                print('MASER_V_THETA.READ_CLOUD_END_STOKES WARNING:  tau_idx for theta = {0} {1} does not match value from theta = {2} {1}'.format(\
                                                                        theta, self.units, self.thetas[0] ) )
                print('                                          Value for theta = {0} {1} : {2}    Value for theta = {3} {1} : {4}'.format(\
                                                                        theta, self.units, self.masers[theta].tau_idx, self.thetas[0], self.tau_idx ))
    
    def calc_R( self, Gamma = None, summed = True, taufs = None, \
    				  ext = 'fits', saveas = None, overwrite = False, verbose = False ):
        """
        Calculates the stimulated emission rate with respect to the loss rate, R/Gamma, at the end of 
        the cloud for each maser object in the masers dictionary attribute for one or more given total 
        optical depths. Includes loading in the deltas object from the output file for that total 
        optical depth for each maser object.
        
        If Gamma is specified, will return the stimulated emission rate, R, in inverse seconds, at
        cloud end is returned as an array with shape ( N_theta, N_tauf ).
        
        If Gamma is not specified, returned array (same shape) will instead by the unitless ratio 
        of stimulated emission rate to loss rate, R/Gamma.
        
        Optional Parameters:
            
            Gamma           Float or None
            
                                [ default = None ]
                                
                                The loss rate in inverse seconds. If provided, will calculate and 
                                return the stimulated emission rate, R. If set to None, will 
                                calculate and return R/Gamma. 
                                
                                Note: either way, any saved fits file contains the values of 
                                R/Gamma. Gamma value is only used to scale values returned.
                                
            summed			Boolean True/False
            					
            					[ default = True ]
            					
            					Whether to calculate R or R/Gamma summed over all n angular 
            					frequency bins (if True), or only return/save the values calculated 
            					in the line center angular frequency bin (if False).
            					
            					Note: the line center bin is determined as the bin at which the
            					value of omegabar is closest to zero. If the omegabar array was
            					selected in such a way that there is not a bin at omegabar = 0, this
            					will not truly be the line center stimulated emission rate.
            
            taufs           Float, List/Array of Floats, or None
            
                                [ default = None ]
                                
                                The value(s) of tauf for which R will be calculated. If None are
                                provided, will calculate for all tauf values in object attribute,
                                taufs.
            
            ext             String ('fits' or 'txt')
            
                                [ Default = 'fits' ]
                                
                                The file extension of the output inversion (deltas) solutions in 
                                the outpath. All maser objects should use the same file extension.
            
            saveas          String or None
            
                                [ Default = None ]
                                
                                If provided as string, will save the calculated stimulated emission
                                rate, R, to a fits file in the top-level object attribute, outpath,
                                with the file name given in the string. String name given by saveas 
                                does not need to end in a '.fits' suffix, though it can.
                                
                                The resulting fits file will have 2 extensions - the 0th extension
                                contains a header with basic information on the results, the tau bin,
                                and the values of tauf (as the data stored in the extension), while 
                                the 1st extension contains the data array of calculated R/Gamma 
                                values (unitless). Any provided value of Gamma will be stored in the
                                0th extension header. A key 'SUMMED' will also be saved in the 0th 
                                extension header indicating whether the results contained were summed
                                over all angular frequency bins or calculated at line center.
            
            overwrite       Boolean
            
                                [ Default = False ]
                                
                                Whether to overwrite any existing fits file with the same name when 
                                creating the output (True) or not (False). Used only if saveas is not
                                None.
                                
            verbose         Boolean
            
                                [ default = False ]
                                
                                Whether to print out progress during calculation.
        
        Other Object Attributes Used:
            
            theta      The angle between the magnetic field and line of sight in radians.
            costheta   cos( self.theta )
            sintheta   sin( self.theta )
            phi        The sky-plane angle in radians.
            costwophi  cos( 2 * self.phi )
            sintwophi  sin( 2 * self.phi )
            etap       The squared ratio of the + dipole moment to the 0th dipole moment.
            etam       The squared ratio of the - dipole moment to the 0th dipole moment.
            omegabar   The array of angular frequencies.
            k          The number of frequency bins spanned by the Zeeman shift, delta omega
            
        Returns:
            
            outarray        NumPy Array
            
                                2-dimensional array with shape ( number_of_theta, number_of_tauf).
                                
                                If Gamma was specified, values contained are stimulated emission
                                rate, R, in inverse seconds, for each (theta,tauf) combination.
                                
                                If Gamma was not specified, values contained are unitless values
                                of R / Gamma.
                                
                                If summed=True, the values have been summed over all angular
                                frequency bins. If summed=False, the values are only those in the
                                angular frequency bin that is closest to line center.
        """
        
        #### Processing tauf to make sure it's a numpy array ####
        
        # If it's not provided, uses object attribute as default
        if taufs is None:
            taufs = self.taufs
        
        # If it's a list or tuple, turns into a numpy array
        elif isinstance(taufs,list) or isinstance(taufs,tuple):
            taufs = np.array( taufs )
        
        # Otherwise, assumes it's a single value and tries to turn into a length-1 numpy array
        elif not isinstance( taufs, np.ndarray ):
            taufs = np.array([ taufs ])
        
        
        
        
        
        
        #### Iterates through taufs and theta values to calculate R/Gamma ####
        
        # Initializes empty 2D array to populate with calculated R/Gamma values
        RG_tauf_theta = np.array([])
        
        # Iterates through theta/maser object attributes
        for theta in self.thetas:
            
            # Initializes empty 1D numpy array to populate with R/Gamma at a given theta as a function of tauf
            RG_v_tauf = np.array([])
            if verbose:
                print('Loading delta and calculating R/Gamma for theta = {0} {1}...'.format(theta, self.units))
            
            # Iterates through tauf values and populates RG_v_tauf array
            for tauf in taufs:
                if verbose:
                    print( '  -- tauf = {0}'.format(tauf) )
                
                # Reads in deltas array for tauf value and sets as object's deltas attribute
                self.masers[theta].deltas = self.masers[theta].readin( tauf, ext=ext, updatepars=False )
            
                # Makes sure that the tauf object attributes are up to date
                self.masers[theta].update_tauf( float(tauf) )
            
                # Calculates the stimulated emission rate and adds to RG_v_tauf array
                #     Does not implement Gamma yet.
                RG_v_tauf = np.append( RG_v_tauf, \
                					   self.masers[theta].calc_R( Gamma = None, summed = summed, verbose = False, sep = False ) )
            
            # Once R/Gamma for all tauf values for that theta have been calculated...
            # Clears out deltas attribute from maser object to conserve memory
            del self.masers[theta].deltas
            
            # Reshapes and adds RG_v_tauf to RG_tauf_theta array
            RG_v_tauf = RG_v_tauf.reshape( RG_v_tauf.size, 1)
            if RG_tauf_theta.size > 0:
                RG_tauf_theta = np.hstack(( RG_tauf_theta, RG_v_tauf ))
            else:
                RG_tauf_theta = np.array( RG_v_tauf )
            
        
        
        
        
        
        
        #### Saves, if requested ####
        
        # Saves, if requested
        if saveas is not None and isinstance( saveas, str ):
            
            # Makes path for file to save and makes sure the file name ends in .fits extension
            savepath = '{0}{1}'.format( self.outpath, saveas )
            if not savepath.lower().endswith('.fits'):
                savepath = '{0}.fits'.format(savepath)
            
            # Makes primary HDU with no data
            prime_hdu = fits.PrimaryHDU()
            
            # Populates primary header with info these stokes arrays
            prime_hdu.header['SUMMED'] = ( summed, 'Calculation summed over ang freq bins?' )
            prime_hdu.header['AFmin'] = ( self.omegabar[0+self.k], 'Angular Freq Min for Stokes Arrays [s^-1]' )
            prime_hdu.header['AFmax'] = ( self.omegabar[-1-self.k], 'Angular Freq Max for Stokes Arrays [s^-1]' )
            prime_hdu.header['AFres'] = ( self.omegabar[1]-self.omegabar[0], 'Angular Freq Resolution [s^-1]' )
            prime_hdu.header['AFbins'] = ( self.omegabar.size, 'Total Angular Freq Bins' )
            prime_hdu.header['AFdata'] = ( self.omegabar.size - 2*self.k, 'Angular Freq Bins for Stokes Data' )
            prime_hdu.header['k'] = ( self.k, 'Zeeman splitting [bins]' )
            prime_hdu.header['taures'] = ( self.fracLOS.size, 'Number of Tau Resolution Bins along LoS' )
            prime_hdu.header['taufN'] = ( self.taufs.size, 'Number of Optical Depths' )
            prime_hdu.header['taufmin'] = ( self.taufs[0], 'Min of Optical Depths' )
            prime_hdu.header['taufmax'] = ( self.taufs[-1], 'Max of Optical Depths' )
            prime_hdu.header['gOmega'] = ( float(self.k) * (self.omegabar[1]-self.omegabar[0] / pi ), 'Full Zeeman spliting rate [s^-1]' )
            
            
            # Populates primary header with other info about calculation
            prime_hdu.header['cloud'] = ( self.cloud, 'number of rays' )
            prime_hdu.header['Doppler'] = ( self.W, 'Doppler width [s^-1]' )
            prime_hdu.header['Zeeman'] = ( (self.omegabar[1]-self.omegabar[0])*float(self.k), 'SS Zeeman splitting [s^-1]' )
            prime_hdu.header['phi'] = ( self.phi, 'Sky angle [rad]' )
            prime_hdu.header['etap'] = ( self.etap, '|d^+|^2 / |d^0|^2' )
            prime_hdu.header['etam'] = ( self.etam, '|d^-|^2 / |d^0|^2' )
            prime_hdu.header['alphap'] = ( self.alphap, 'P^+ / P^0' )
            prime_hdu.header['alpham'] = ( self.alpham, 'P^- / P^0' )
            prime_hdu.header['i0'] = ( self.iquv0[0], 'Ray 1 initial Stokes i' )
            prime_hdu.header['q0'] = ( self.iquv0[1], 'Ray 1 initial Stokes q' )
            prime_hdu.header['u0'] = ( self.iquv0[2], 'Ray 1 initial Stokes u' )
            prime_hdu.header['v0'] = ( self.iquv0[3], 'Ray 1 initial Stokes v' )
            if self.cloud == 2:
                prime_hdu.header['iF'] = ( self.iquvF[0], 'Ray 2 initial Stokes i' )
                prime_hdu.header['qF'] = ( self.iquvF[1], 'Ray 2 initial Stokes q' )
                prime_hdu.header['uF'] = ( self.iquvF[2], 'Ray 2 initial Stokes u' )
                prime_hdu.header['vF'] = ( self.iquvF[3], 'Ray 2 initial Stokes v' )
            prime_hdu.header['endfill'] = ( self.endfill, 'Mode for handling freq edges' )
            prime_hdu.header['farcoeff'] = ( self.far_coeff, '-gamma_qu/cos(theta)' )
            prime_hdu.header['nexp'] = ( self.n, 'Number of Expansion Terms' )
            prime_hdu.header['ftol'] = ( self.ftol, 'Tolerance for convergence' )
            if self.fccalc:
                prime_hdu.header['ne'] = ( self.ne, 'Electron number density [m^-3]' )
                prime_hdu.header['AF0'] = ( self.freq0, 'Angular frequency at line center [s^-1]' )
                prime_hdu.header['Gamma'] = ( self.Gamma, 'Loss rate [s^-1]' )
                prime_hdu.header['B'] = ( self.B, 'Magnetic field strength [G]' )
                prime_hdu.header['A0'] = ( self.A0, 'Einstein A coeff [s^-1]' )
                prime_hdu.header['P0'] = ( self.P0, 'Pi Pump rate [m^-3 s^-1]' )
            
            # Saves array of taufs values as data of primary hdu
            prime_hdu.data = taufs
               
            # Makes HDU for data extension
            ext1 = fits.ImageHDU( RG_tauf_theta.astype( np.float64 ) )
            ext1.name = 'RdGamma'
            
            # Makes HDU list with each hdu as an extension
            hdu = fits.HDUList([ prime_hdu, ext1 ])
            
            # Writes hdulist to file
            hdu.writeto( savepath, overwrite = overwrite )
            
            # Prints feedback if requested
            if verbose:
                print('Stimulated emission file {0} written.'.format( savepath ) )
            
        
        #### If Gamma was specified, converts R/Gamma values to R ####
        
        if Gamma is not None:
            R_tauf_theta = RG_tauf_theta * Gamma
            outarray = R_tauf_theta
        else:
            outarray = RG_tauf_theta
        
        
        #### Returns whichever array ####
        
        return outarray
            
    def read_R(self, filename, Gamma = None, verbose = True ):
        """
        Reads in fits file created by method calc_R using saveas option and returns the stimulated
        emission rate array, R, (if Gamma is specified), or the ratio of the stimulated emission rate to
        the loss rate, R/Gamma (if Gamma is not specified) as a function of theta and tauf.
        
        
        
        Required Parameters:
            
            filename        String
                                Name of the fits file to which the calculated R values were saved by calc_R
                                method. Assumed to be in path specified by object attribute outpath.
        
        Optional Parameters:
                                
            
            Gamma           Float or None
                                [ default = None ]
                                The loss rate in inverse seconds. If provided, will return the stimulated 
                                emission rate, R. If set to None, will return R/Gamma. 
                                
                                Note: assumes fits file being read contains the values of 
                                R/Gamma. Gamma value is only used to scale values returned.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        
        Updates Attributes:
            
            taufs
        
        Returns:
            
            outarray        NumPy Array
                                2-dimensional array with shape ( number_of_theta, number_of_tauf ).
                                
                                If Gamma was specified, values contained are stimulated emission
                                rate, R, in inverse seconds, for each (theta,tauf) combination.
                                
                                If Gamma was not specified, values contained are unitless values
                                of R / Gamma.
        
        Other Functionality:
            
            Checks other attributes stored in fits file header against object attributes, and raises a 
            warning if there is disagreement (but will still run).
        
        """
        
        # File assumed to be within the outpath
        filepath = '{0}{1}'.format( self.outpath, filename )
        if not os.path.exists(filepath):
            
            # If file is not in the outpath, checks from current directory
            if os.path.exists(filename):
                print('    Warning: File {0} not found in outpath {1}, but found in current directory.'.format(filename, self.outpath))
                print('             Using file in current directory.')
                filepath = filename
            
            # if file is not in outpath or current directory
            else:
                raise FileNotFoundError('Files {0} or {1} not found.'.format(filepath, filename))
        
        # Opens file in such a way that it will close automatically when over
        with fits.open(filepath) as hdu:
            
            
            
            #### Checks parameters from file against object attributes ####
            if verbose:
            
                # Makes warning string templates
                warning_temp_line = '    MASER.READ_R WARNING:  Parameter {0} in file does not match object attribute. (Diff : {1:.2e})'
            
                # Starts with k, since that'll affect the omegabar arrays
                if hdu[0].header['k'] != self.k:
                    print( warning_temp_line1.format( 'k' ) )
                    print( warning_temp_line2.format( hdu[0].header['k'], self.k, hdu[0].header['k'] - self.k ) )
            
                # Then checks omegabar values
                check_dict = OrderedDict([ ( 'AFmin' , self.omegabar[0+self.k] ), \
                                           ( 'AFmax' , self.omegabar[-1-self.k] ), \
                                           ( 'AFres' , self.omegabar[1]-self.omegabar[0] ), \
                                           ( 'AFbins', self.omegabar.size ), \
                                           ( 'AFdata', self.omegabar.size - 2*self.k ) ])
                for key in check_dict.keys():
                    if hdu[0].header[key] != check_dict[key]:
                        print( warning_temp_line.format( key, hdu[0].header[key] - check_dict[key] ) )
            
                # Checks tau and tauf values
                #    tauf used to be called beta, so if tauf key is not in the hdu header, looks for the old
                #    beta key
                check_dict = OrderedDict([ ( 'taures' , self.fracLOS.size ), \
                                           ( 'taufN'  , self.taufs.size ), \
                                           ( 'taufmin', self.taufs[0] ), \
                                           ( 'taufmax', self.taufs[-1] ) ])
                for key in check_dict.keys():
                    hdu_key = str(key)
                    if hdu_key not in hdu[0].header.keys():
                        hdu_key = hdu_key.replace( 'tauf', 'beta' )
                    if hdu[0].header[hdu_key] != check_dict[key]:
                        print( warning_temp_line.format( key , hdu[0].header[hdu_key] - check_dict[key]) )
            
                # Checks other values present for all sims
                check_dict = OrderedDict([ ( 'cloud'   , self.cloud ), \
                                           ( 'Doppler' , self.W ), \
                                           ( 'Zeeman'  , (self.omegabar[1]-self.omegabar[0])*float(self.k) ), \
                                           ( 'phi'     , self.phi ), \
                                           ( 'etap'    , self.etap ), \
                                           ( 'etam'    , self.etam ), \
                                           ( 'alphap'  , self.alphap ), \
                                           ( 'alpham'  , self.alpham ), \
                                           ( 'i0'      , self.iquv0[0] ), \
                                           ( 'q0'      , self.iquv0[1] ), \
                                           ( 'u0'      , self.iquv0[2] ), \
                                           ( 'v0'      , self.iquv0[3] ), \
                                           ( 'endfill' , self.endfill ), \
                                           ( 'farcoeff', self.far_coeff ), \
                                           ( 'nexp'    , self.n ), \
                                           ( 'ftol'    , self.ftol ) ])
                for key in check_dict.keys():
                    if hdu[0].header[key] != check_dict[key]:
                        print( warning_temp_line.format( key, hdu[0].header[key] - check_dict[key] ) )
            
                # Checking fcalc values if set for both
                if 'ne' in hdu[0].header.keys() and self.fcalc:
                    check_dict = OrderedDict([ ( 'ne'   , self.ne ), \
                                               ( 'AF0'  , self.freq0 ), \
                                               ( 'Gamma', self.Gamma ), \
                                               ( 'B'    , self.B ), \
                                               ( 'A0'   , self.A0 ), \
                                               ( 'P0'   , self.P0 ) ])
                    for key in check_dict.keys():
                        if hdu[0].header[key] != check_dict[key]:
                            print( warning_temp_line.format( key, hdu[0].header[key] - check_dict[key] ) )
            
                # Checking values for 2nd ray if both are bi-directional
                if hdu[0].header['cloud'] == 2 and self.cloud == 2:
                    check_dict = OrderedDict([ ( 'iF'      , self.iquvF[0] ), \
                                               ( 'qF'      , self.iquvF[1] ), \
                                               ( 'uF'      , self.iquvF[2] ), \
                                               ( 'vF'      , self.iquvF[3] ) ])
                    for key in check_dict.keys():
                        if hdu[0].header[key] != check_dict[key]:
                            print( warning_temp_line.format( key, hdu[0].header[key] - check_dict[key] ) )
            
            
            
            
            #### Retrieves other values associated with R calculation ####
            
            # Saves array of taufs values
            self.taufs = hdu[0].data
            
            # Retrieves R_tauf_theta
            RG_tauf_theta = hdu[1].data
            
            
            
            #### Prints note saying if read-in values were summed or not ####
            
            # Sets brief key regarding type of data that will be returned for feedback
            if Gamma is not None:
                valtype = 'R'
            else:
                valtype = 'R/Gamma'
            
            # If SUMMED key actually in header
            if 'SUMMED' in hdu[0].header.keys():
                summed = hdu[0].header['SUMMED']
                if summed:
                    print('Values of {0} retrieved from {1} were summed over all angular frequency bins.'.format(\
                            valtype, filename))
                else:
                    print('Values of {0} retrieved from {1} are values at the line center bin only.'.format(\
                            valtype, filename))
		    
			# If files written before SUMMED key added
            else:
                print('Values of {0} from {1} do not have associated SUMMED key.'.format(valtype, filename))
                print('    They were probably created before this key was added, and are therefore likely')
                print('    summed over all wavelength bins (or are out of date).')
            
        
        #### If Gamma was specified, converts R/Gamma values to R ####
        
        if Gamma is not None:
            R_tauf_theta = RG_tauf_theta * Gamma
            outarray = R_tauf_theta
        else:
            outarray = RG_tauf_theta
        
        
        
        #### Returns whichever array ####
        
        return outarray
        
        
        
        
            
    
    ### Functions for plotting figures ###
    
    def plot_mlevpa( self, tauf = None, overplot_gkk = False, label = None, label_loc = 'left', ml_max = None, \
                        legend_loc = 3, legend_cols = 1, R_tauf_theta = None, figname = None, show=True ):
        """
        Plots m_l and EVPA, both vs. theta, in two windows. Does so for a single total optical depth, tauf, 
        or a list of optical depths. 
        
        Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
        of tauf values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
        
        Can label curves with log(R/gOmega) value instead of tauf value, but requires R_tauf_theta to have been
        calculated and/or read in with calc_R/read_R for either the full tauf values in the taufs attribute or
        the subset of tauf values provided on plot_mlevpa method call here.
            
        Optional Parameters:
            
            tauf            None, Float, or List of Floats 
                                [ Default = None ]
                                The values or values of total optical depth to be plotted. If None, plots all
                                values of tauf in self.taufs object attribute
            
            overplot_gkk    Boolean
                                [ Default = False ]
                                If True, will overplot the GKK functional form on the ml(theta) subplot.
            
            label           None or String
                                [ Default = None ]
                                Text to label inside plot.
                        
            label_loc       String: 'left', 'right', 'upperleft', 'upperright', 'lowerleft', or 'lowerright' 
                                [ Default = 'left' ]
                                The corner of the plot in which the label (if provided) will be placed. Here,
                                'upper' and 'lower' refer to which subplot the label will be placed in, not the
                                corner within the subplot. Note that 'left' and 'right' are shortened options for 
                                'upperleft' and 'upperright'. Not case sensitive.
                                Will probably want 'left' for lower optical depths and 'right' for higher optical
                                depths.
                                
            ml_max          Float or None
                                [ Default = None ]
                                Upper shown on the ml plot y-axis (ml). If None, will use the automatic plot
                                value scaled by matplotlib. (Mainly useful if you want to set the y-limit used 
                                by the figure to be the same as other figures.)
            
            legend_loc      Integer
                                [ Default = 3 ]
                                Location of the plot legend in the EVPA subfigure. Default (legend_loc=3) puts 
                                legend in lower left corner. Based on matplotlib legend location options. Other
                                common options are upper left (legend_loc=2), upper right (legend_loc=1), and
                                lower right (legend_loc=4).
                                
            legend_cols     Integer
                                [ Default = 1 ]
                                Number of columns in the legend.
            
            R_tauf_theta    None or 2D NumPy array
                                [ Default = None ]
                                If None, will label each curve by its tauf value. If a 2D NumPy array of
                                calculated R values is provided, will label each curve by its mean log(R/gOmega) 
                                value, and print out the variation in log(R/gOmega) for each curve as a function 
                                of theta. The array provided must have shape (Ntaufs, Nthetas), where Ntaufs is
                                *either* the number of tauf values in the taufs object attribute *or* the number
                                of tauf values provided by tauf key here.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
        """
        #### First, checks values ####
        method_name = 'MASER_V_THETA.PLOT_MLEVPA'
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Checks that label_loc is one of the valid options and converts to lower case
        label_loc = label_loc.lower()
        if label is not None and label_loc not in [ 'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright' ]:
            err_msg = "{0}: Accepted values for label_loc are:\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright'."
            raise ValueError(err_msg)
        
            
        
        # Makes template of error message for attribute checks
        attr_missing_msg1 = method_name + ': Object attribute {0} does not exist.'
        attr_missing_msg2 = method_name + ': Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_shape_msg    = method_name + ': Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                        ' '*(12+len(method_name)+2) + 'is not consistent with attributes taufs, omegabar, and k.\n' + \
                        ' '*(12+len(method_name)+2) + 'Attribute {0} should be NumPy array of shape ( {3}, {4} ).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Ntaufs = self.taufs.size
        Nfreq  = self.omegabar.size - 2 * self.k
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for theta in self.thetas:
            
            for req_att in required_attributes:
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # If it does exist and it's not tau_idx, makes sure that the array shape is correct
                elif req_att != 'tau_idx':
                    if self.masers[theta].__dict__[req_att].shape != ( Ntaufs, Nfreq ):
                        raise ValueError( attr_shape_msg.format(req_att, theta, self.units, Ntaufs, Nfreq) )
        
        
        
        
        
        
        #### Does some processing on requested tauf value ####
        
        # If none is provided, assume all tauf values in taufs attribute are desired
        if tauf is None:
            tauf = list( self.taufs )
            tauf_idxs = list( np.arange( len(tauf) ) )
        else:
        
            # If tauf provided is single value, not list, makes it into len-1 list
            if isinstance( tauf, float ) or isinstance( tauf, int ):
                tauf = [ float(tauf) ]
            
            # Determines indices for each tauf value in taufs object attribute
            tauf_idxs = []
            for bval in tauf:
                if float(bval) in self.taufs:
                    tauf_idxs.append( np.where( self.taufs == float(bval) )[0][0] )
                else:
                    err_msg = '{0}: Requested tauf value, {1}, not in taufs object attribute.\n'.format(method_name, bval) + \
                              ' '*(12+len(method_name)+2) + \
                              'Please make sure that cloud_end_stokes attributes have been generated or read for\n' + \
                              ' '*(12+len(method_name)+2) + \
                              'the desired tauf values before calling this method.'
                    raise ValueError(err_msg)
        
        
        
        
        
        
        # If R_tauf_theta is provided, checks it and creates labels
        if R_tauf_theta is not None:
        
            # Calculates gOmega and converts from angular frequency to frequency
            gOmega = float(self.k) * float( self.omegabar[1]-self.omegabar[0] )  / pi 
            
            # First, makes sure it's a numpy array
            if not isinstance( R_tauf_theta, np.ndarray ):
                err_msg = method_name + ': R_tauf_theta must be a NumPy array, if provided.'
                raise TypeError( err_msg )
            
            # Then, makes sure it's 2D
            elif R_tauf_theta.ndim != 2:
                err_msg = method_name + ': R_tauf_theta, if provided, must be a 2-dimensional NumPy array. (Current dimensions: {0})'
                raise ValueError( err_msg.format(  R_tauf_theta.ndim ) )
            
            # Then checks number of values along theta axis; should have shape ( Ntaufs, Nthetas )
            elif R_tauf_theta.shape[1] != self.thetas.size:
                err_msg = method_name + ': Number of values along theta (1st) axis in R_tauf_theta must equal the number of thetas in the object attribute.\n'+\
                        ' '*(12+len(method_name)+2) + 'Size of thetas array: {0},   Values along theta axis in R_tauf_theta: {1}'
                raise ValueError( err_msg.format( self.thetas.size, R_tauf_theta.shape[1] ) )
            
            # Then checks if number of values along tauf axis are the same size as tauf array
            elif R_tauf_theta.shape[0] == len(tauf):
                
                # Calculates array of log(R/gOmega); should still have shape ( Ntauf, Ntheta )
                logRpts = np.log10( R_tauf_theta / gOmega )
            
            # If number of values along tauf axis different from tauf array but same as taufs attribute, 
            #   calculates logRpts as pared down from original R_tauf_theta array to only include 
            #   requested tauf values
            elif R_tauf_theta.shape[0] == self.taufs.size:
                logRpts = np.log10( R_tauf_theta[tauf_idxs] / gOmega )
                
            # If neither are true, raises an error
            else:
                err_msg = method_name + ': Number of values along tauf (0th) axis in R_tauf_theta must equal the number of tauf values provided OR.\n'+\
                        ' '*(12+len(method_name)+2) + 'the number of tauf values in the taufs object attribute.\n' + \
                        ' '*(12+len(method_name)+2) + 'taufs provided: {0},  Size of taufs array: {1},   Values along tauf axis in R_tauf_theta: {2}'
                raise ValueError( err_msg.format( len(tauf), self.taufs.size, R_tauf_theta.shape[0] ) )
                
            # If no error was raised, calculates the mean log(R/gOmega) vs theta for each included tauf value
            mean_logR = np.mean( logRpts, axis=1 )
            
            # Calculates range of logR/gOmega across theta for each tauf
            range_logR = np.array([  np.max(logRpts[i,:]) - np.min(logRpts[i,:]) for i in range(logRpts.shape[0])  ])
            
            # Calculates variation as a percent of the mean
            logR_percent_change = np.abs( 100. * range_logR / mean_logR )
            
            # Prints output of percent variation across theta for each value of tauf
            print('Percent Change in log(R/gOmega) as a function of theta:')
            for i in range(len(tauf)):
                print('  tauf = {0: >5}  --  log(R/gOmega) = {1: >5}  --  {2:.2e} %'.format( tauf[i], round(mean_logR[i],1), logR_percent_change[i] ))
            
            # Create curve labels
            curve_labels = [ r'$\log(R/g\Omega) = $' + str(round(x,1)) for x in mean_logR  ]
        
        # If R_tauf_theta not provided, label is just the value of tauf
        else:
            curve_labels = tauf
            
                
                
            
            
        
        
        
        
        #### Determine the colors and markers ####
        
        if len( tauf ) <= 7:
            color_list  = color_sets[ 7][:len(tauf)]
            marker_list = marker_sets[7][:len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        elif len(tauf) in [8,9]:
            color_list  = color_sets[ len(tauf)]
            marker_list = marker_sets[len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        else:
            color_list  = list( islice( cycle( color_sets[ 8] ), len(tauf) ))
            marker_list = list( islice( cycle( marker_sets[9] ), len(tauf) ))
            fill_template = [ 'full', ]*8
            fill_template.extend( ['none',]*8 )
            fill_list   = list( islice( cycle( fill_template ), len(tauf) ))
        
        
        
        
        #### Determine the index of the line center frequency ####
        
        # Makes sure that all maser objects in masers dictionary have same k value
        ks_for_theta = np.unique( np.array([ self.masers[ theta ].k for theta in self.thetas ]) )
        if ks_for_theta.size > 1:
            err_msg = '{0}: Maser objects in masers dictionary must all have same value of k.\n'.format(method_name) + \
                      ' '*(12+len(method_name)+2) + '({0} values found.)'.format( ks_for_theta.size )
            raise ValueError( err_msg )
        
        # Checks top-level object k value to make sure it's consistent with these
        if self.k not in ks_for_theta:
            err_msg = '{0}: Maser_v_theta object must have the same value of k as objects in masers dictionary.'.format(method_name)
            raise ValueError( err_msg )
        
        # Actually sets aside index of line center frequency
        jcenter = int( Nfreq / 2 )
        
        
        
        
        #### Figure axes and labels ####
        
        # Creates figure window
        fig, ax = P.subplots(nrows=2, ncols=1, sharex=True, figsize = (5.5,4.5))
        fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.91 )
        
        # Iterates through requested tauf values for plot and plots them
        for i in range(len(tauf)):
            
            # Gets the index that corresponds to this total optical depth in the stacked arrays
            tauf_idx = tauf_idxs[i]
            
            # Makes the lists of line center ml and evpa values to plot
            plot_ml   = [ self.masers[theta].stacked_ml[   tauf_idx , jcenter ] for theta in self.thetas ]
            plot_evpa = [ ( self.masers[theta].stacked_evpa[ tauf_idx , jcenter ] + pi ) % pi for theta in self.thetas ]
            # Actually plots with corresponding color/marker/fill
            ax[0].plot( self.thetas, plot_ml  , marker = marker_list[i], \
                                                color = color_list[i], fillstyle = fill_list[i] )
            ax[1].plot( self.thetas, plot_evpa, marker = marker_list[i], \
                                                color = color_list[i], fillstyle = fill_list[i], label=curve_labels[i] )
        
        # Overplot GKK functional form, if requested
        if overplot_gkk:
            gkk_thetas = np.linspace( self.thetas[0], self.thetas[-1], 1001 )
            gkk_ml, gkk_evpa = gkk(gkk_thetas, units = self.units)
            ax[0].plot( gkk_thetas, np.abs(gkk_ml), 'k--' )
        
        # Y-axis limits and label of evpa plot
        ymin1, ymax1 = -pi/16., 17.*pi/16.
        ax[1].set_ylim( ymin1, ymax1 )
        ax[1].set_ylabel(r'$\chi$ [radians]')
        
        # Puts the ticks of the evpa plot in units of pi radians
        ticks = [ 0, 0.25*pi, 0.5*pi, 0.75*pi, pi]
        tick_labels=['0', '', r'$\pi$/2', '', r'$\pi$']
        ax[1].set_yticks( ticks )
        ax[1].set_yticklabels( tick_labels )
        
        
        
        #### Figure axes and labels ####
        
        # X-axis limits and label depend on the object's units used for theta
        if self.units in ['degrees','deg','d']:
            P.xlim( 0 , 90 )
            ax[1].set_xlabel(r'$\theta$ [$^{\circ}$]')
        else:
            P.xlim( 0 , 0.5*pi )
            ax[1].set_xlabel(r'$\theta$ [radians]')
            
        # Y-axis limits and label of ml plot
        if ml_max is None:
            ax[0].set_ylim(bottom=0)
        else:
            ax[0].set_ylim(0, ml_max)
        ax[0].set_ylabel(r'$m_l$')
        
        # Use scientific notation for y-axis of m_l plot if small
        ymin, ymax = ax[0].get_ylim()
        if ymax <= 1e-2:
            format_label_string_with_exponent( fig, ax[0], axis='y' )
        
        # Y-axis limits and label of evpa plot
        ymin1, ymax1 = -pi/16., 17.*pi/16.
        ax[1].set_ylim( ymin1, ymax1 )
        ax[1].set_ylabel(r'$\chi$ [radians]')
        
        # Puts the ticks of the evpa plot in units of pi radians
        ticks = [ 0, 0.25*pi, 0.5*pi, 0.75*pi, pi]
        tick_labels=['0', '', r'$\pi$/2', '', r'$\pi$']
        ax[1].set_yticks( ticks )
        ax[1].set_yticklabels( tick_labels )
        
        # Make the legend
        if R_tauf_theta is None:
            ax[1].legend(loc=legend_loc, fontsize='small', ncol=legend_cols)
        else:
            ax[1].legend(loc=legend_loc, fontsize='x-small', ncol=legend_cols)
    
        # Sets plot label, if requested.
        if label is not None:
            if label_loc in ['left','upperleft']:
                ax[0].text( 90.*0.02, ymax - (ymax -ymin )*0.05, label, ha='left', va='top')
            elif label_loc in ['right','upperright']:
                ax[0].text( 90.*0.98, ymax - (ymax -ymin )*0.05, label, ha='right', va='top')
            elif label_loc == 'lowerleft':
                ax[1].text( 90.*0.02, ymax1- (ymax1-ymin1)*0.05, label, ha='left', va='top')
            elif label_loc == 'lowerright':
                ax[1].text( 90.*0.98, ymax1- (ymax1-ymin1)*0.05, label, ha='right', va='top')
        
        
        
        
        #### Saves/Shows Figure ####
        
        # Saves figure if requested
        if figname is not None:
            try:
                fig.savefig( figname )
            except:
                print('Unable to save figure to {0}'.format(figname))
    
        # Finally, shows the plot, if requested
        if show:
            P.show()
        else:
            P.close()
    
    def plot_v_didv( self, freqoff, freq0, tauf = None, pbnorm = False, label = None, label_loc = 'left', ylims = None, \
                        legend_loc = 3, legend_cols = 1, figname = None, show = True ):
        """
        Plots stokv / ( d stoki / d velocity ) vs. cos(theta). Does so for a single total optical depth or a list 
        of optical depths, tauf.
        
        Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
        of tauf values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
        
        Required Parameters:
            
            freqoff         Integer
                                Offset index with respect to the central frequency to be plotted. Note: Calling 
                                with freqoff = 0 will likely result in division by zero error. For best results, 
                                supply non-zero values.
            
            freq0           Float
                                Rest frequency of the line, in Hz. Should correspond to the frequency at which
                                omegabar = 0.  
            
        Optional Parameters:
            
            tauf            None, Float, or List of Floats 
                                [ Default = None ]
                                The values or values of total optical depth to be plotted. If None, plots all
                                values of tauf in self.taufs object attribute
                                
            pbnorm          Boolean
                                [ Default = False ]
                                Whether or not to normalize the y-axis by the Zeeman splitting in velocity space, 
                                pB, (if True) or not (if False). If performed, will retrieve the Zeeman splitting 
                                in angular frequency from the spacing the angular frequency grid, omegabar, times 
                                the number of bins that span the Zeeman splitting, k. This will be then converted 
                                to velocity space before being used to normalize the y-axis values of the plot, 
                                providing a unitless y-axis. If normalization is not performed, y-axis will be in 
                                units of m/s.
                        
            label          None or String
                                [ Default = None ]
                                Text to label inside plot.
                        
            label_loc      String: 'left' or 'right'
                                [ Default = 'left' ]
                                The corner of the plot in which the label (if provided) will be placed.
                                Will probably want 'left' for lower optical depths and 'right' for higher optical
                                depths.
                                
            ylims           Length-2 list or tuple or None
                                [ Default = None ]
                                Limits for y-axis in plot. If None, determines automatically.
            
            legend_loc      Integer
                                [ Default = 3 ]
                                Location of the plot legend in the EVPA subfigure. Default (legend_loc=3) puts 
                                legend in lower left corner. Based on matplotlib legend location options. Other
                                common options are upper left (legend_loc=2), upper right (legend_loc=1), and
                                lower right (legend_loc=4).
                                
            legend_cols     Integer
                                [ Default = 1 ]
                                Number of columns in the legend.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
        """
        
        #### First, checks values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg =  "MASER_V_THETA.PLOT_V_DIDV ERROR:    Setting show = False without specifying a file name for the plot will result in no\n" + \
                ' '*12+"                                    figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Checks that label_loc is one of the valid options and converts to lower case
        label_loc = label_loc.lower()
        if label is not None and label_loc not in [ 'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright' ]:
            err_msg =  "MASER_V_THETA.PLOT_V_DIDV ERROR:    Accepted values for label_loc are:\n" + \
                ' '*12+"                                    'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright'"
            raise ValueError(err_msg)
        
            
        
        # Makes template of error message for attribute checks
        attr_missing_msg1 = 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Object attribute {0} does not exist.'
        attr_missing_msg2 = 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_ndim_msg     = 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Object attribute {0} for maser object with theta = {1} {2} must be\n' + \
                     ' '*16+'                                    2-dimensional. (Current number of dimensions = {3}).'
        attr_dim0_msg     = 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                     ' '*16+'                                    is not consistent with attributes omegabar and k.\n' + \
                     ' '*16+'                                    Should be 2D NumPy array with {3} values along 0-axis (currently {4}).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Ntaufs = self.taufs.size
        Nfreq  = self.omegabar.size - 2 * self.k
        
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for req_att in required_attributes:
            for theta in self.thetas:
                
                # Checks if the attribute exists
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # For those that are arrays, check their dimensions and freq values
                if req_att != 'tau_idx':
                
                    # Checks that the attribute is a 2d array
                    if self.masers[theta].__dict__[req_att].ndim != 2:
                        raise AttributeError( attr_ndim_msg.format(req_att, theta, self.units, self.masers[theta].__dict__[req_att].ndim ) )
                
                    # Checks number of values along 1st (frequency) axis
                    elif self.masers[theta].__dict__[req_att].shape[1] != Nfreq:
                        raise AttributeError( attr_dim0_msg.format(req_att, theta, self.units, Nfreq, self.masers[theta].__dict__[req_att].shape[0] ) )
            
            # Checks size of 0th (tauf) axis of those attribute arrays for every theta value
            if req_att != 'tau_idx':
                Ntauf_per_theta = np.array([ self.masers[theta].__dict__[req_att].shape[0] for theta in self.thetas ])
            
                # If not every maser object has the same number of taufs (and this hasn't been done for a previous 
                #   attribute), prints a warning and adjusts number of taufs
                if np.unique( Ntauf_per_theta ).size != 1 and Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_FREQ_THETA WARNING:    Optical depths in object attribute {0} not consistent across theta.\n' + \
                              '                                          Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
            
                # If every maser object does have the same number of taufs but its less than that expected from Ntaufs, 
                #   prints a warning and adjusts number of taufs
                elif  Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_FREQ_THETA WARNING:    Optical depths in object attribute {0} not consistent with taufs attribute.\n' + \
                              '                                          Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
        
        
        
        
        
        #### Does some processing on requested tauf value ####
        
        # If none is provided, assume all tauf values in taufs attribute are desired
        if tauf is None:
            tauf = list( self.taufs )
            tauf_idxs = list( np.arange( len(tauf) ) )[ : np.array(ibmax).max() ]
        else:
        
            # If tauf provided is single value, not list, makes it into len-1 list
            if isinstance( tauf, float ) or isinstance( tauf, int ):
                tauf = [ float(tauf) ]
            
            # Determines indices for each tauf value in taufs object attribute
            tauf_idxs = []
            for bval in tauf:
                if float(bval) in self.taufs:
                    tauf_idxs.append( np.where( self.taufs == float(bval) )[0][0] )
                else:
                    err_msg = 'Requested tauf value, {0}, not in taufs object attribute.\n'.format(bval) + \
                              '    Please make sure that cloud_end_stokes attributes have been generated or read for\n' + \
                              '    the desired tauf values before calling this method.'
                    raise ValueError(err_msg)
        
        
        
        
        #### Determine the colors and markers ####
        
        if len( tauf ) <= 7:
            color_list  = color_sets[ 7][:len(tauf)]
            marker_list = marker_sets[7][:len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        elif len(tauf) in [8,9]:
            color_list  = color_sets[ len(tauf)]
            marker_list = marker_sets[len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        else:
            color_list  = list( islice( cycle( color_sets[ 8] ), len(tauf) ))
            marker_list = list( islice( cycle( marker_sets[9] ), len(tauf) ))
            fill_template = [ 'full', ]*8
            fill_template.extend( ['none',]*8 )
            fill_list   = list( islice( cycle( fill_template ), len(tauf) ))
        
        
        
        
        #### Determine the frequency index ####
        
        # Makes sure that all maser objects in masers dictionary have same k value
        ks_for_theta = np.unique( np.array([ self.masers[ theta ].k for theta in self.thetas ]) )
        if ks_for_theta.size > 1:
            raise ValueError( 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Maser objects in masers dictionary must all have same value of k.' + \
                              ' ({0} values found.)'.format( ks_for_theta.size ) )
        
        # Checks top-level object k value to make sure it's consistent with these
        if self.k not in ks_for_theta:
            raise ValueError( 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Maser_v_theta object must have the same value of k as objects in masers dictionary.' )
        
        # Actually sets aside index of line center frequency
        jcenter = int( Nfreq / 2 )
        
        # Gets index for desired frequency bin to plot (freqoff)
        jplot = jcenter + freqoff
        
        
        
        
        #### Calculating velocity step ####
    
        # Converts angular frequency array, omegabar (s^-1), to frequency (wrt line center) array in Hz
        freqHz = self.omegabar[ self.k : -self.k ] / (2.*pi)
        
        # Converts frequency array to velocity in m/s
        velms_term = (1.0 + freqHz/freq0)
        velms = ( ( 1.0 - velms_term**2 ) / ( 1.0 + velms_term**2 ) ) * c # m/s
    
        # If normalization by pB is being performed, calculates it as the zeeman split in m/s
        #	i.e. pB = Delta velocity_Zeeman
        if pbnorm:
        
            # Retrieves the zeeman splitting in angular frequency; units of s^-1
            zeeman_AF = ( self.omegabar[1] - self.omegabar[0] ) * self.k
        
            # Converts to zeeman splitting in velocity space to get pB = zeeman_V; units of m/s
            norm = zeeman_AF * c / ( 2.0 * pi * freq0 )
    
        # If no normalization, sets normalization factor to 1.0
        else:
            norm = 1.0
    
        # Calculates difference in velocity spanning 2 bins centered on jplot
        dv = velms[ jplot+1 ] - velms[ jplot-1 ]
        
        
        
        
        #### Actually plots figure ####
        
        # Creates figure window
        fig, ax = P.subplots(figsize = (4.5,4.5))
        fig.subplots_adjust( hspace=0, left=0.2,bottom=0.13,right=0.95,top=0.91 )
    
        # Gets list of costhetas 
        if self.units in ['degrees','deg','d']:
            plot_costheta = np.cos( self.thetas * pi / 180. )
        else:
            plot_costheta = np.cos( self.thetas )
            
        # Initializes flag to say if we're flipping the y-axis
        flip = None
        
        # Iterates through requested tauf values for plot and plots them
        for i, bval in enumerate(tauf):
            
            # Gets the index that corresponds to this total optical depth in the stacked arrays
            tauf_idx = tauf_idxs[i]
            
            # Gets array of stokes v values to plot, one for each theta
            plot_stokv = np.array([ self.masers[theta].stacked_stokv[    tauf_idx , jplot   ] for theta in self.thetas ])
            
            # Gets array of di/dvel for each maser object
            plot_stokvi_m = np.array([ self.masers[theta].stacked_stoki[ tauf_idx , jplot-1 ] for theta in self.thetas ])
            plot_stokvi_p = np.array([ self.masers[theta].stacked_stoki[ tauf_idx , jplot+1 ] for theta in self.thetas ])
            didv = ( plot_stokvi_p - plot_stokvi_m ) / dv
            
            # Divides stokes v by didvel for plotting
            plot_vdidvel = plot_stokv / didv
            
            # Figures out if we need to flip this, if flag hasn't been set yet
            if flip is None:
                if ( plot_vdidvel <= 0.0 ).all():
                    flip = -1.0
                else:
                    flip = 1.0
            
            # Actually plots curve
            ax.plot( plot_costheta, flip*plot_vdidvel/norm, marker = marker_list[i], color = color_list[i], \
                                                                                fillstyle = fill_list[i], label = bval)
        
        #### Figure axes and labels ####
        
        # X-axis limits and label
        ax.set_xlim(0,1)
        ax.set_xlabel(r'$\cos \theta$')
        
        # Y-axis limits can be provided on call
        if ylims is not None:
            ymin, ymax = ylims
        elif np.nanmin( flip*plot_vdidvel ) >= 0.0:
            ymin = 0
            ymax = None
        else:
            ymin = None
            ymax = None
        ax.set_ylim(ymin, ymax)
        
        #### Figure axes and labels ####
        
        # X-axis limits and label
        ax.set_xlim(0,1)
        ax.set_xlabel(r'$\cos \theta$')
        
        # Y-axis limits can be provided on call
        if ylims is not None:
            ymin, ymax = ylims
        elif np.nanmin( flip*plot_vdidvel ) >= 0.0:
            ymin = 0
            ymax = None
        else:
            ymin = None
            ymax = None
        ax.set_ylim(ymin, ymax)
        
        # Y-axis label depends if axis flipped and if pB normalization occurred
        if flip == 1.0 and not pbnorm:
            ylabel = r'$v / ( \partial i / \partial$v$)$ [m s$^{-1}$]'
        elif flip == - 1.0 and not pbnorm:
            ylabel = r'$ - v / ( \partial i / \partial$v$)$ [m s$^{-1}$]'
        elif flip == 1.0 and pbnorm:
            ylabel = r'$v / ( p B \partial i / \partial$v$)$'
        elif flip == - 1.0 and pbnorm:
            ylabel = r'$ - v / ( p B \partial i / \partial$v$)$'
        ax.set_ylabel( ylabel )
        
        # Use scientific notation for y-axis if values small
        ymin, ymax = ax.get_ylim()
        if ymax <= 1e-2:
            ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
        
        # Legend
        ax.legend(loc=legend_loc, fontsize='small', ncol=legend_cols)
    
        # Makes frequency label
        dfreq = freqHz[jplot]
        if dfreq >= 0.0:
            freqsign = '+'
        else:
            freqsign = '-'
        if abs(dfreq) < 1000.0:
            freqlabel = '{0}{1} Hz'.format( freqsign, round(abs(dfreq)) )
        elif abs(dfreq) >= 1e3 and abs(dfreq) < 1e6:
            freqlabel = '{0}{1} kHz'.format( freqsign, round(abs(dfreq)*1e-3) )
        elif abs(dfreq) >= 1e6 and abs(dfreq) < 1e9:
            freqlabel = '{0}{1} MHz'.format( freqsign, round(abs(dfreq)*1e-6) )
        elif abs(dfreq) >= 1e9:
            freqlabel = '{0}{1} GHz'.format( freqsign, round(abs(dfreq)*1e-9) )
    
        # Adds freq to label, if provided; if not, sets label just as frequency
        if label is not None:
            label = freqlabel + '\n' + label
        else:
            label = freqlabel

        
        # Sets plot label, if requested.
        if label_loc == 'left':
            ax.text( 0.02, ymax - (ymax-ymin)*0.05, label, ha='left', va='top', fontsize='large')
        elif label_loc == 'right':
            ax.text( 0.98, ymax - (ymax-ymin)*0.05, label, ha='right', va='top', fontsize='large')
        
        
        
        #### Saves/Shows Figure ####
        
        # Saves figure if requested
        if figname is not None:
            try:
                fig.savefig( figname )
            except:
                print('Unable to save figure to {0}'.format(figname))
    
        # Finally, shows the plot, if requested
        if show:
            P.show()
        else:
            P.close()
    
    def plot_freq_theta( self, plotvalue, tauf, convert_freq = True, interp = 'cubic', subtitle = None, \
                        figname = None, show = True, verbose = True ):
        """
        Plots desired value vs. frequency (x-axis) and total optical depth of the cloud (y-axis).
    
        Can be used to plot mc, ml, evpa, stoki, stokq, stoku, stokv, fracq (stokq/stoki), or fracu (stoku/stoki).
        
        Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
        of tauf values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
        
        Required Parameters:
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), or 'fracu' for (stoku/stoki).
            
            tauf            Float or Integer
                                Value of the total optical depth to be plotted. Must be in taufs attribute.
            
        Optional Parameters:
            
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False).
            
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'.
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        #### First, checks values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg =  "MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Setting show = False without specifying a file name for the plot will result in no\n" + \
                ' '*16+"                                        figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Makes template of error message for attribute checks
        attr_missing_msg1 = 'MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Object attribute {0} does not exist.'
        attr_missing_msg2 = 'MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_ndim_msg     = 'MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Object attribute {0} for maser object with theta = {1} {2} must be\n' + \
                     ' '*16+'                                        2-dimensional. (Current number of dimensions = {3}).'
        attr_dim0_msg     = 'MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                     ' '*16+'                                        is not consistent with attributes omegabar and k.\n' + \
                     ' '*16+'                                        Should be 2D NumPy array with {3} values along 0-axis (currently {4}).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Ntaufs = self.taufs.size
        Nfreq  = self.omegabar.size - 2 * self.k
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for req_att in required_attributes:
            for theta in self.thetas:
                
                # Checks if the attribute exists
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # For those that are arrays, check their dimensions and freq values
                if req_att != 'tau_idx':
                
                    # Checks that the attribute is a 2d array
                    if self.masers[theta].__dict__[req_att].ndim != 2:
                        raise AttributeError( attr_ndim_msg.format(req_att, theta, self.units, self.masers[theta].__dict__[req_att].ndim ) )
                
                    # Checks number of values along 1st (frequency) axis
                    elif self.masers[theta].__dict__[req_att].shape[1] != Nfreq:
                        raise AttributeError( attr_dim0_msg.format(req_att, theta, self.units, Nfreq, self.masers[theta].__dict__[req_att].shape[0] ) )
            
            # Checks size of 0th (tauf) axis of those attribute arrays for every theta value
            if req_att != 'tau_idx':
                Ntauf_per_theta = np.array([ self.masers[theta].__dict__[req_att].shape[0] for theta in self.thetas ])
            
                # If not every maser object has the same number of taufs (and this hasn't been done for a previous 
                #   attribute), prints a warning and adjusts number of taufs
                if np.unique( Ntauf_per_theta ).size != 1 and Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_FREQ_THETA WARNING:    Optical depths in object attribute {0} not consistent across theta.\n' + \
                              '                                          Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
            
                # If every maser object does have the same number of taufs but its less than that expected from Ntaufs, 
                #   prints a warning and adjusts number of taufs
                elif  Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_FREQ_THETA WARNING:    Optical depths in object attribute {0} not consistent with taufs attribute.\n' + \
                              '                                          Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
        
        # Makes sure that specified plotvalue is allowed:
        if plotvalue not in ['mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', 'fracu']:
            err_msg = "MASER_V_THETA.PLOT_FREQ_THETA ERROR:    plotvalue '{0}' not recognized.\n".format(plotvalue) + \
                      ' '*12+"                                        Allowed values are 'mc', 'ml', 'evpa', 'stoki', 'stokq',\n" + \
                      ' '*12+"                                        'stoku', 'stokv', 'fracq', or 'fracu'."
            raise ValueError(err_msg)
            
        
        
        
        
        
        
        #### Processing tauf ####
        
        # Makes sure tauf is a float
        tauf = float(tauf)
        
        # Finds index of tauf in taufs array
        if tauf in self.taufs[:Ntaufs]:
            ib = np.where( self.taufs == tauf )[0][0]
        elif tauf not in self.taufs:
            err_msg = "MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Requested tauf value, {0}, not in taufs attribute array.\n".format(tauf) 
            raise ValueError(err_msg)
        else:
            err_msg = "MASER_V_THETA.PLOT_FREQ_THETA ERROR:    Requested tauf value, {0}, not available for all thetas.\n".format(tauf) 
            raise ValueError(err_msg)
        
        
        #### Determines frequency extent for plot and converts, if requested ####
        
        # Gets ANGULAR frequency range for x-axis of plot; used for imshow extent, not plot limits
        dfreq = self.omegabar[1] - self.omegabar[0]
        freqmin = self.omegabar[self.k] - dfreq/2.
        freqmax = self.omegabar[-self.k-1] + dfreq/2.
    
        # Converts these to frequency, if requested
        if convert_freq:
            dfreq = dfreq / ( 2.*pi )
            freqmin = freqmin / ( 2.*pi )
            freqmax = freqmax / ( 2.*pi )
        
            # Converts to MHz
            dfreq *= 1e-6
            freqmin *= 1e-6
            freqmax *= 1e-6
        
            # Generates axis label for frequency, while we're here
            xlabel = r'$\nu$ [MHz]'
    
        # If no conversion requested, just generates axis label
        else:
            xlabel = r'$\varpi$ [s$^{-1}$]'
            
            
        
        #### Determining which array to plot ####
        
        # Start with Stokes i
        if plotvalue == 'stoki':
            
            # Set aside array with plotable range
            temparray = np.array([  self.masers[theta].stacked_stoki[ ib , : ] for theta in self.thetas  ])
            
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes i at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'viridis'
        
        # Next, Stokes q
        elif plotvalue == 'stokq':
        
            # Set aside array with plotable range
            temparray = np.array([  self.masers[theta].stacked_stokq[ ib , : ] for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Next, Stokes u
        elif plotvalue == 'stoku':
        
            # Set aside array with plotable range
            temparray = np.array([  self.masers[theta].stacked_stoku[ ib , : ] for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Next, Stokes v
        elif plotvalue == 'stokv':
        
            # Set aside array with plotable range
            temparray = np.array([  self.masers[theta].stacked_stokv[ ib , : ] for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes v at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Next, fractional stokes q
        elif plotvalue == 'fracq':
        
            # Set aside array with plotable range (q/i)
            temparray = np.array([  self.masers[theta].stacked_stokq[ ib , : ] / self.masers[theta].stacked_stoki[ ib , : ]  \
                                                                                                for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q/i at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Next, fractional stokes u
        elif plotvalue == 'fracu':
        
            # Set aside array with plotable range (u/i)
            temparray = np.array([  self.masers[theta].stacked_stoku[ ib , : ] / self.masers[theta].stacked_stoki[ ib , : ]  \
                                                                                                for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u/i at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Next, does ml
        elif plotvalue == 'ml':
            
            # Array to plot is just stacked_ml
            temparray = np.array([  self.masers[theta].stacked_ml[ ib , : ] for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l$ at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'viridis'
        
        # Next, does mc
        elif plotvalue == 'mc':
            
            # Array to plot is just stacked_mc
            temparray = np.array([  self.masers[theta].stacked_mc[ ib , : ] for theta in self.thetas  ])
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_c$ at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
        
        # Finally, does evpa
        elif plotvalue == 'evpa':
        
            # Array to plot is just stacked_evpa, but makes sure phase wrapped between 0 and pi
            temparray = ( np.array([  self.masers[theta].stacked_evpa[ ib , : ] for theta in self.thetas  ]) + pi ) % pi
        
            # Sets plot title & colormap, while we're here
            fig_title = r'EVPA at Cloud End with $\tau_f=$'+str(tauf)
            cmap = 'RdBu'
            
            
        
        
        
        
        #### Regridding array for smooth distribution of theta solutions ####
        
        if temparray.size != 0:
            
            if verbose:
                print('Regridding data...')
            
            # Ravels temparray to prepare for regridding
            temparray = np.ravel( temparray )
        
            # Creates grid of existing frequency/theta points
            freqpts, thetapts = np.meshgrid( self.omegabar[self.k : -self.k], self.thetas )
            freqpts  = np.ravel( freqpts )
            thetapts = np.ravel( thetapts )
            points  = np.vstack(( freqpts, thetapts )).T
            
            # Creates grid of desired theta values and regrids
            #   Assumed frequency is already equi-spaced so doesn't change.
            thgoal = np.linspace( self.thetas.min(), self.thetas.max(), num=36 )
            freqgrid, thetagrid = np.meshgrid( self.omegabar[self.k : -self.k], thgoal )
            
            # Sets aside theta resolution for later use
            dtheta = thgoal[1] - thgoal[0]
            
            # Re-grids data array with desired interpolation
            zs = griddata( points, temparray, (freqgrid, thetagrid), method=interp)
            
            
            
        
        
        
            #### Figures out min and max for color ####
            
            if verbose:
                print('Plotting figure...')
            
            vmax = np.nanmax( np.abs( zs[ : , 2*self.k : -2*self.k ] ) )
            if plotvalue == 'evpa':
                vmin = 0.0
                vmax = pi
            elif cmap == 'RdBu':
                vmin = -1.0 * vmax
            else:
                vmin = 0.0
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            P.close()
            fig, ax = P.subplots(nrows=1, ncols=1, figsize = (5.5,4.5))
            fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.88 )
            im = ax.imshow( zs, aspect='auto', vmin = vmin, vmax = vmax, cmap = cmap, origin='lower', \
                      extent = [freqmin, freqmax, self.thetas.min() - dtheta/2., self.thetas.max() + dtheta/2.] )
            
            # Axis limits 
            ax.set_xlim( freqmin, freqmax )
            ax.set_ylim( self.thetas.min() , self.thetas.max() )
            
            # If frequency conversion not requested, makes x-axis frequency ticks in scientific notation
            if not convert_freq:
                ax.set_ticklabel_format( axis='x', style = 'sci', scilimits = (0,0) )
            
            # X-axis label
            ax.set_xlabel(xlabel)
            
            # Y-axis label depends on theta units
            if self.units in ['degrees','deg','d']:
                ax.set_ylabel(r'$\theta$ [$^{\circ}$]')
            else:
                ax.set_ylabel(r'$\theta$ [radians]')
            
            # Plot title
            if subtitle is None:
                ax.set_title( fig_title )
            else:
                ax.set_title( fig_title + '\n' + subtitle )
            
            # Colorbar
            cbar = fig.colorbar( im, ax=ax )
            if vmax <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
        
            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                    if verbose:
                        print('Saved figure to {0}.'.format(figname))
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
    
    def plot_freq_tau( self, theta, plotvalue, taufmax = None, plottaufmax = None, convert_freq = True, \
                        tau_scale = 'linear', interp = 'cubic', subtitle = None, figname = None, show = True, verbose = True ):
        """
        Plots desired value vs. frequency (x-axis) and total optical depth of the cloud (y-axis) for a given 
        maser class object with a single value of theta.
    
        Can be used to plot mc, ml, evpa, stoki, stokq, stoku, stokv, fracq (stokq/stoki), or fracu (stoku/stoki).
        
        Intended to be run *after* stokes at a given point in cloud have been read in for a range of tauf values
        with cloud_end_stokes method.
        
        Required Parameters:
            
            theta           Float
                                The value of theta (in units specified on initialization of this object) for
                                which the plot is desired. Used as the key to access the desired maser object
                                in the masers dictionary attribute.
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), or 'fracu' for (stoku/stoki).
            
        Optional Parameters:
            
            taufmax         None or Float 
                                [ Default = None ]
                                Maximum value of tauf to show data for in the plot. If None, plots all available 
                                data.
                                
            plottaufmax     Float or None
                                [ Default = None ]
                                Y-limit (optical depth) shown on the plot axes. If None, will be the same as
                                taufmax. (Only useful if you want to set the y-limit used by the figure to be
                                the same as other figures despite not having tauf up to that value for this
                                parameter set.)
                        
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False).
            
            tau_scale       String ('log' or 'linear')
                                [ Default = 'linear' ]
                                Scale for the y-axis (total optical depth) on the plot.
                                
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'.
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        
        # Checks that theta is in the thetas array
        if float(theta) not in self.thetas:
            errmsg = 'MASER_V_THETA.PLOT_FREQ_TAU ERROR:    Maser object does not exist for maser object with theta = {0} {1}.'
            raise ValueError( errmsg.format( theta, self.unit ) )
        
        # If it is a valid key, runs plot_freq_tau for that object
        else:
            self.masers[ float(theta) ].plot_freq_tau( plotvalue, taufmax = taufmax, plottaufmax = plottaufmax, \
                                                       convert_freq = convert_freq, tau_scale = tau_scale, interp = interp, \
                                                       subtitle = subtitle, figname = figname, show = show, verbose = verbose )
    
    def plot_theta_tau( self, plotvalue, freqoff = 0, taufmax = None, plottaufmax = None, contours = False, \
                        convert_freq = True, tau_scale = 'linear', interp = 'cubic', subtitle = None, label = None, \
                        label_loc = 'left', figname = None, show = True, verbose = True ):
        """
        Plots desired value at some offset from line center vs. theta (x-axis) and total optical depth (y-axis).
        
        Intended to be run *after* stokes at a given point in cloud have been read in for a range of tauf values
        with cloud_end_stokes method.
        
        Required Parameters:
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), 'fracu' for (stoku/stoki), 'quv' for sqrt(q^2+u^2)/i-v/i, 
                                'mlmc' for (ml-mc), or 'vmetric' for ( Vmax - Vmin ) / Imax.
                                Note: 'vmetric' plotvalue only available for freqoff = None. 
            
        Optional Parameters:
            
            freqoff         Integer, list of Integers, or None
                                [ Default = 0 ]
                                If a single integer, value gives the offset index (with respect to the central  
                                frequency in omegabar) to be plotted.
                                If a list of integers, individual values should be the same as in the case of a 
                                single integer, but plotted output will be summed across those wavelength bins.
                                If None, calculates the desired plotvalue from the peak stokes values across 
                                angular frequency for each theta and tauf. 
                                NOTE: None is NOT the same as 0!!!
            
            taufmax         None or Float 
                                [ Default = None ]
                                Maximum value of tauf to show data for in the plot. If None, plots all available 
                                data.
                                
            plottaufmax     Float or None
                                [ Default = None ]
                                Y-limit (optical depth) shown on the plot axes. If None, will be the same as
                                taufmax. (Only useful if you want to set the y-limit used by the figure to be
                                the same as other figures despite not having tauf up to that value for this
                                parameter set.)
                        
            contours        Boolean
                                [ Default = False ]
                                Whether (True) or not (False) to overplot contours on the image.
                        
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False).
            
            tau_scale       String ('log' or 'linear')
                                [ Default = 'linear' ]
                                Scale for the y-axis (total optical depth) on the plot.
                                
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'.
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                        
            label          None or String
                                [ Default = None ]
                                Y-limit (optical depth) shown on the plot axes. If None, will be the same as
                                taufmax. (Only useful if you want to set the y-limit used by the figure to be
                                the same as other figures despite not having tauf up to that value for this
                                parameter set.)
                        
            label_loc      String: 'left' or 'right'
                                [ Default = 'left' ]
                                The (lower) corner of the plot in which the label (if provided) will be placed.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        #### First, checks values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg =  "MASER_V_THETA.PLOT_THETA_TAU ERROR:    Setting show = False without specifying a file name for the plot will result in no\n" + \
                ' '*16+"                                       figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        

        # Makes template of error message for attribute checks
        attr_missing_msg1 = 'MASER_V_THETA.PLOT_THETA_TAU ERROR:    Object attribute {0} does not exist.'
        attr_missing_msg2 = 'MASER_V_THETA.PLOT_THETA_TAU ERROR:    Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_ndim_msg     = 'MASER_V_THETA.PLOT_THETA_TAU ERROR:    Object attribute {0} for maser object with theta = {1} {2} must be\n' + \
                     ' '*16+'                                        2-dimensional. (Current number of dimensions = {3}).'
        attr_dim0_msg     = 'MASER_V_THETA.PLOT_THETA_TAU ERROR:    Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                     ' '*16+'                                        is not consistent with attributes omegabar and k.\n' + \
                     ' '*16+'                                        Should be 2D NumPy array with {3} values along 0-axis (currently {4}).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Ntaufs = self.taufs.size
        Nfreq  = self.omegabar.size - 2 * self.k
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for req_att in required_attributes:
            for theta in self.thetas:
                
                # Checks if the attribute exists
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # For those that are arrays, check their dimensions and freq values
                if req_att != 'tau_idx':
                
                    # Checks that the attribute is a 2d array
                    if self.masers[theta].__dict__[req_att].ndim != 2:
                        raise AttributeError( attr_ndim_msg.format(req_att, theta, self.units, self.masers[theta].__dict__[req_att].ndim ) )
                
                    # Checks number of values along 1st (frequency) axis
                    elif self.masers[theta].__dict__[req_att].shape[1] != Nfreq:
                        raise AttributeError( attr_dim0_msg.format(req_att, theta, self.units, Nfreq, self.masers[theta].__dict__[req_att].shape[0] ) )
            
            # Checks size of 0th (tauf) axis of those attribute arrays for every theta value
            if req_att != 'tau_idx':
                Ntauf_per_theta = np.array([ self.masers[theta].__dict__[req_att].shape[0] for theta in self.thetas ])
            
                # If not every maser object has the same number of taufs (and this hasn't been done for a previous 
                #   attribute), prints a warning and adjusts number of taufs
                if np.unique( Ntauf_per_theta ).size != 1 and Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_THETA_TAU WARNING:    Optical depths in object attribute {0} not consistent across theta.\n' + \
                              '                                         Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
            
                # If every maser object does have the same number of taufs but its less than that expected from Ntaufs, 
                #   prints a warning and adjusts number of taufs
                elif  Ntauf_per_theta.min() != Ntaufs:
                        print('MASER_V_THETA.PLOT_THETA_TAU WARNING:    Optical depths in object attribute {0} not consistent with taufs attribute.\n' + \
                              '                                         Highest value of optical depth present for all theta is {1}. ({2} values.)'.format( \
                              req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                        Ntaufs = Ntauf_per_theta.min()
        
        # Makes sure that specified plotvalue is allowed:
        if plotvalue not in ['mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', 'fracu', 'vmetric', 'quv', 'mlmc']:
            err_msg =  "MASER_V_THETA.PLOT_THETA_TAU ERROR:    plotvalue '{0}' not recognized.\n".format(plotvalue) + \
                ' '*12+"                                       Allowed values are 'mc', 'ml', 'evpa', 'stoki', 'stokq',\n" + \
                ' '*12+"                                       'stoku', 'stokv', 'fracq', or 'fracu'."
            raise ValueError(err_msg)
        
        # Checks that tau_scale is an acceptable value
        tau_scale = tau_scale.lower()
        if tau_scale not in ['linear','log']:
            err_msg = "tau_scale '{0}' not recognized.\n".format(tau_scale) + \
                      "        Allowed values are 'linear' or 'log'."
            raise ValueError(err_msg)

        
        # If using default taufmax, just uses last one with stokes for all theta values present
        elif taufmax is None:
            ibmax = Ntaufs - 1
            taufmax = self.taufs[ibmax]
        

        
        
        
        #### Processing defaults for taufmax and plottaufmax ####
        
        # Finds index of maximum tauf present for all ml arrays
        if taufmax is not None:
        
            # Finds index of taufmax in taufs array
            if taufmax in self.taufs[:Ntaufs]:
                ibmax = np.where( self.taufs == taufmax )[0][0]
            elif taufmax not in self.taufs:
                err_msg = "MASER_V_THETA.PLOT_THETA_TAU ERROR:    Requested tauf value, {0}, not in taufs attribute array.\n".format(taufmax) 
                raise ValueError(err_msg)
            else:
                err_msg = "MASER_V_THETA.PLOT_THETA_TAU ERROR:    Requested tauf value, {0}, not available for all thetas.\n".format(taufmax) 
                raise ValueError(err_msg)
        
        # If using default taufmax, just uses last one with stokes for all theta values present
        elif taufmax is None:
            ibmax = Ntaufs - 1
            taufmax = self.taufs[ibmax]
        
        # Sets plottaufmax, if not set
        if plottaufmax is None:
            plottaufmax = taufmax
        

        
        
        
        
        #### Determine the frequency index ####
        
        # Makes sure that all maser objects in masers dictionary have same k value
        ks_for_theta = np.unique( np.array([ self.masers[ theta ].k for theta in self.thetas ]) )
        if ks_for_theta.size > 1:
            raise ValueError( 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Maser objects in masers dictionary must all have same value of k.' + \
                              ' ({0} values found.)'.format( ks_for_theta.size ) )
        
        # Checks top-level object k value to make sure it's consistent with these
        if self.k not in ks_for_theta:
            raise ValueError( 'MASER_V_THETA.PLOT_V_DIDV ERROR:    Maser_v_theta object must have the same value of k as objects in masers dictionary.' )
        
        # Actually sets aside index of line center frequency
        jcenter = int( Nfreq / 2 )
            
        
        
        #### Determining which array to plot ####
        
        # Start with Stokes i
        if plotvalue == 'stoki':
            
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    thetaline = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
            
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes i'
            cmap = 'viridis'
        
        # Next, Stokes q
        elif plotvalue == 'stokq':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q'
            cmap = 'RdBu'
        
        # Next, Stokes u
        elif plotvalue == 'stoku':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u'
            cmap = 'RdBu'
        
        # Next, Stokes v
        elif plotvalue == 'stokv':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    min_stokv = np.min( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    thetaline = np.array([ ( max_stokv[j] - min_stokv[j] ) for j in range(ibmax+1) ])
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Range in Stokes v'
                    cmap = 'viridis'
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + freqoff ]
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Stokes v'
                    cmap = 'RdBu'
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Stokes v'
                    cmap = 'RdBu'
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
        # Next, fractional stokes q
        elif plotvalue == 'fracq':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ] / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) / np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) 
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q/i'
            cmap = 'RdBu'
        
        # Next, fractional stokes u
        elif plotvalue == 'fracu':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ] / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) / np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) 
                thetaline = thetaline.reshape( thetaline.size, 1)
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u/i'
            cmap = 'RdBu'
        
        # Next, does ml
        elif plotvalue == 'ml':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_ml[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_ml[ : ibmax+1 , jcenter + freqoff ]
                else:
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = np.sqrt( qsum**2 + usum**2 ) / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l$'
            cmap = 'viridis'
        
        # Next, does mc
        elif plotvalue == 'mc':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    
                    # Calculation if we want peak mc to be the mc where stokes v is max, not the maximum value of mc
                    #max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    #peak_idx = np.array([ np.where( self.masers[theta].stacked_stokv[j,:] == x )[0][0] for j,x in enumerate(max_stokv) ])
                    #thetaline = np.array([ max_stokv[j] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                    
                    # Calculation if we want the peak mc to mean the actual max mc
                    thetaline = np.max( self.masers[theta].stacked_mc[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'viridis'
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta       ].stacked_mc[ : ibmax+1 , jcenter + freqoff ]
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'RdBu'
                else:
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = vsum / isum
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'RdBu'
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
        # Finally, does evpa
        elif plotvalue == 'evpa':
        
            # Build array with plotable range
            # Note: np.arctan2 returns values between -pi and +pi, with 0.5 * np.arctan2 returning values between
            #       -pi/2 and +pi/2. Applying + pi % pi to this changes the range to 0 - pi, flipping only the negative
            #       values to the new quadrant.
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_evpa[ j, peak_idx[j] ] + pi for j in range(ibmax+1) ]) % pi
                elif isinstance(freqoff,int):
                    thetaline = ( self.masers[ theta      ].stacked_evpa[ : ibmax+1 , jcenter + freqoff ] + pi ) % pi
                else:
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = 0.5 * np.arctan2( usum, qsum )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'EVPA'
            cmap = 'RdBu'
        
        # New option: vmetric plots (Vmax - Vmin) / Imax (only available with freqoff=None)
        elif plotvalue == 'vmetric':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                min_stokv = np.min( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                thetaline = np.array([ ( max_stokv[j] - min_stokv[j] ) / max_stoki[j] for j in range(ibmax+1) ])
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
    
            # Sets plot title & colormap, while we're here
            fig_title = r'$(V_{max} - V_{min}) / I_{max}$'
            cmap = 'viridis'
    
        # New option: quv plots sqrt( q**2 + u**2 )/i - v/i
        elif plotvalue == 'quv':
    
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx  = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    qupeak    = np.sqrt( np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] for j in range(ibmax+1) ])**2 + \
                                         np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] for j in range(ibmax+1) ])**2 )
                    thetaline = ( qupeak - max_stokv ) / max_stoki
                elif isinstance(freqoff,int):
                    thetaline = np.sqrt( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ]**2 + \
                                         self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ]**2 ) - \
                                         self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + freqoff ]
                    thetaline = thetaline / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = np.sqrt( qsum**2 + usum**2 ) - vsum
                    thetaline = thetaline / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
    
            # Sets plot title & colormap, while we're here
            fig_title = r'$(q^2 + u^2)^{1/2}/i - v/i$'
            cmap = 'RdBu'
    
        # New option: mlmc plots ml - mc
        elif plotvalue == 'mlmc':
    
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx  = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    max_mc    = np.max( self.masers[theta].stacked_mc[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    ml_peak   = np.array([ self.masers[ theta ].stacked_ml[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                    thetaline = ml_peak - max_mc
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_ml[ : ibmax+1 , jcenter + freqoff ] - \
                                self.masers[ theta ].stacked_mc[ : ibmax+1 , jcenter + freqoff ]
                else:
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = ( np.sqrt( qsum**2 + usum**2 ) - vsum ) / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
    
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l - m_c$'
            cmap = 'RdBu'
            
            
        
        
        
        
        #### Regridding array for smooth distribution of theta and tauf solutions ####
        
        if temparray.size != 0:
            
            if verbose:
                print('Regridding data...')
            
            # Ravels temparray to prepare for regridding
            # Before ravel, has shape (tauf, theta)
            temparray = np.ravel( temparray )
        
            # Creates grid of existing theta and tauf values
            # Resulting arrays have shape ( tauf, theta ) when meshgrid of (theta, tauf)
            thetapts, taufpts = np.meshgrid( self.thetas, self.taufs[:ibmax+1] )
            thetapts = np.ravel( thetapts )
            taufpts  = np.ravel( taufpts )
            points   = np.vstack(( thetapts, taufpts )).T
            
            # Creates grid of desired theta and tauf values and regrids
            #   Again, thetagrid and taufgrid have shape ( 1001, 36 )
            thetagoal = np.linspace( self.thetas.min(), self.thetas.max(), num=36 )
            
            # Creates grid of desired total optical depths (in requested scale) and regrids. 
            if tau_scale == 'log':
                taufgoal = np.logspace( log10(self.taufs[0]), log10(taufmax), 1001)
            else:
                taufgoal = np.linspace( self.taufs[0], taufmax, 1001)
            
            # Creates grids of each value
            thetagrid, taufgrid = np.meshgrid( thetagoal, taufgoal )
            
            # Re-grids data array with desired interpolation
            # Resulting zs has shape ( Nthetagoal, Ntaufgoal ), like thetagrid and taufgrid
            zs = griddata( points, temparray, (thetagrid, taufgrid), method=interp)
            
            
            
        
        
        
            #### Figures out min and max for color ####
            
            if verbose:
                print('Plotting figure...')
            
            # zs shape is (taufgoal, thetagoal). no freq edge effects
            vmax = np.nanmax(np.abs( zs ))
            if plotvalue == 'evpa':
                vmin = 0.0
                vmax = pi
            elif cmap == 'RdBu':
                vmin = -1.0 * vmax
            else:
                vmin = 0.0
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            P.close()
            fig, ax = P.subplots(nrows=1, ncols=1, figsize = (5.5,4.5))
            fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.88 )
                
            dtheta = thetagoal[1] - thetagoal[0]
            
            # Plots image
            if tau_scale == 'log':
                # Get correct bin ends on y-scale
                dy = log10(taufgoal[1]/taufgoal[0])
                #print('dy = {0}'.format(dy))
                theta_plot = np.linspace( thetagoal[0]-dtheta/2., thetagoal[-1]+dtheta/2., thetagoal.size+1)
                tauf_plot  = np.logspace( log10(self.taufs[0]) - dy/2. , log10(taufmax) + dy/2., taufgoal.size+1 )
                #print('theta_plot from {0} to {1}'.format(theta_plot[0], theta_plot[-1]))
                #print('tauf_plot from {0} to {1}'.format(tauf_plot[0], tauf_plot[-1]))
                im = ax.pcolormesh( theta_plot, tauf_plot, zs, vmin = vmin, vmax = vmax, cmap = cmap )
                ax.set_yscale( 'log' )
            else:
                dtauf  = taufgoal[1]  - taufgoal[0]
                im = ax.imshow( zs, aspect='auto', origin='lower', vmin = vmin, vmax = vmax, cmap = cmap, \
                                extent = [thetagoal[0] - dtheta/2., thetagoal[-1] + dtheta/2., \
                                taufgoal[0] - dtauf/2., taufgoal[-1] + dtauf/2.] )
            
            
        
            # Adds contour, if requested
            if contours and plotvalue != 'evpa':
                levels = gen_contour_levels( vmin, vmax, min_contours = 4 )
                contour_color = [ 'w', ]*len(levels)
                if cmap == 'RdBu':
                    #print(levels)
                    contour_color[ levels.index(0.0) ] = 'lightgray'
                _cs2 = ax.contour( thetagrid, taufgrid, zs, levels=levels, origin='lower', colors=contour_color )
            
            # If making contours for evpa plot, central contour will need to be darker
            elif contours:
                levels_degrees = [ 30, 60, 90, 120, 150 ]
                levels = [ float(x) * pi / 180. for x in levels_degrees ]
                _cs2 = ax.contour( thetagrid, taufgrid, zs, levels=levels, origin='lower', \
                    colors=['w','w','lightgray','w','w'] )
            
            # Axis limits
            ax.set_xlim( self.thetas.min(), self.thetas.max() )
            ax.set_ylim( self.taufs[0], plottaufmax )
            
            # Axis labels; x-axis label depends on theta units
            if self.units in ['degrees','deg','d']:
                ax.set_xlabel(r'$\theta$ [$^{\circ}$]')
            else:
                ax.set_xlabel(r'$\theta$ [radians]')
            ax.set_ylabel(r'Total $\tau_f$')
            
            # Makes colorbar; will have ticks at overplotted contour lines, if contours requested
            if interp is not None:
                
                # Makes just the colorbar with correct ticks
                if plotvalue == 'evpa':
                    levels_degrees = [ 0, 30, 60, 90, 120, 150, 180 ]
                    cbar = fig.colorbar( im, ax=ax, ticks = [ float(x)*pi/180. for x in levels_degrees ] )
                    tick_labels = ['0', '', '', r'$\pi$/2', '', '', r'$\pi$']
                    cbar.ax.set_yticklabels( tick_labels )
                    if contours:
                        cbar.add_lines( _cs2 )
                    
                elif contours:
                    cbar = fig.colorbar( im, ax=ax, ticks = levels )
                    cbar.add_lines( _cs2 )
                
                else:
                    cbar = fig.colorbar( im, ax=ax )
                    
            
            # If no interpolation, can only make contour plot, so just labels contours on figure, no colorbar
            else:
                if plotvalue == 'evpa':
                    mk_clabel = lambda crad : str( round( crad * 180. / pi ) )
                    ax.clabel(_cs2, fontsize=9, inline=1, fmt=mk_clabel, colors='k' )
                    cbar = None
                else:
                    ax.clabel(_cs2, fontsize=9, inline=1, fmt='%.1e', colors='k' )
                    cbar = None
            
            # Converts colorbar ticks to scientific notation if they're small; won't affect evpa plots
            if cbar is not None and vmax <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
            
            # Determine plot title; depends on frequency with respect to line center
            if isinstance(freqoff,int) and freqoff == 0:
                freqtitle = 'Line Center'
            elif freqoff is None:
                freqtitle = 'Peak'
            elif isinstance(freqoff,int):
                # Gets frequency offset in ANGULAR freq (s^-1)
                dfreq = self.omegabar[ jcenter + freqoff ] - self.omegabar[jcenter]
            
                 # If frequency conversion to MHz is requested, converts and generates freq_string label
                if convert_freq:
            
                    # Converts from angular frequency to frequency & Creates label
                    dfreq = dfreq / (2.*pi)
                    if dfreq < 1000.0:
                        freqtitle = '{0} Hz from Line Center'.format( round(dfreq,0) )
                    elif dfreq >= 1e3 and dfreq < 1e6:
                        freqtitle = '{0} kHz from Line Center'.format( round(dfreq*1e-3,0) )
                    elif dfreq >= 1e6 and dfreq < 1e9:
                        freqtitle = '{0} MHz from Line Center'.format( round(dfreq*1e-6,0) )
                    elif dfreq >= 1e9:
                        freqtitle = '{0} GHz from Line Center'.format( round(dfreq*1e-9,0) )
        
                # If no frequency conversion requested, makes frequency label
                else:
                    freqtitle = r'$\varpi =$ {0:.2e}'.format(dfreq) + r' s$^{-1}$ from Line Center'
            else:
                # Gets frequency offset in ANGULAR freq (s^-1)
                dfreq = self.omegabar[ jcenter + np.max(freqoff) ] - self.omegabar[ jcenter + np.min(freqoff) ]
                midfreq = ( self.omegabar[ jcenter + np.max(freqoff) + self.k ] + self.omegabar[ jcenter + np.min(freqoff) + self.k ] ) / 2.
            
                 # If frequency conversion to MHz is requested, converts and generates freq_string label
                if convert_freq:
            
                    # Converts from angular frequency to frequency & Creates label
                    dfreq = dfreq / (2.*pi)
                    midfreq = midfreq / (2.*pi)
                    if abs(midfreq) < 1e-7:
                        freqtitle = 'at Line Center'
                    elif midfreq < 1000.0:
                        freqtitle = '{0} Hz from Line Center'.format( round(midfreq,0) )
                    elif midfreq >= 1e3 and midfreq < 1e6:
                        freqtitle = '{0} kHz from Line Center'.format( round(midfreq*1e-3,0) )
                    elif midfreq >= 1e6 and midfreq < 1e9:
                        freqtitle = '{0} MHz from Line Center'.format( round(midfreq*1e-6,0) )
                    elif midfreq >= 1e9:
                        freqtitle = '{0} GHz from Line Center'.format( round(midfreq*1e-9,0) )
                    
                    if dfreq < 1000.0:
                        freqtitle += ' ({0} Hz wide)'.format( round(dfreq,0) )
                    elif dfreq >= 1e3 and dfreq < 1e6:
                        freqtitle += ' ({0} kHz wide)'.format( round(dfreq*1e-3,0) )
                    elif dfreq >= 1e6 and dfreq < 1e9:
                        freqtitle += ' ({0} MHz wide)'.format( round(dfreq*1e-6,0) )
                    elif dfreq >= 1e9:
                        freqtitle += ' ({0} GHz wide)'.format( round(dfreq*1e-9,0) )
        
                # If no frequency conversion requested, makes frequency label
                else:
                    freqtitle = r'$\varpi =$ {0:.2e}'.format(midfreq) + r' s$^{-1}$ from Line Center (' + \
                                '{0:.2e}'.format(dfreq) + r' s$^{-1}$ wide)'
            
            # Combines and adds plot title
            if subtitle is None:
                P.title( fig_title + r' at Cloud End, ' + freqtitle + '\n' )
            else:
                P.title( fig_title + r' at Cloud End, ' + freqtitle + '\n' + subtitle )
        
            # Sets plot label, if requested.
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if label is not None:
                margin_size = 0.02
                if tau_scale == 'linear':
                    ytext = ymin + (ymax-ymin)*margin_size
                else:
                    ytext = ymin * 10.**( margin_size * log10(ymax/ymin) )
                if label_loc == 'left':
                    ax.text( xmin + (xmax-xmin)*margin_size, ytext, label, ha='left', va='bottom', fontsize='large')
                elif label_loc == 'right':
                    ax.text( xmin + (xmax-xmin)*(1.-margin_size),ytext, label, ha='right', va='bottom', fontsize='large')
            
            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                    if verbose:
                        print('Saved figure to {0}.'.format(figname))
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
    
    def plot_mlmc( self, freqoff, tauf = None, label = None, label_loc = 'left', legend_cols = 1, \
                        figname = None, show = True ):
        """
        Plots ml (top) and mc (bottom) vs. cos(theta) at some offset frequency from the line center 
        (freqoff) for a selection of optical depths (tauf) in two vertical subplots.
        
        Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
        of tauf values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
        
        Required Parameters:
            
            freqoff         Integer
                                Offset index with respect to the central frequency to be plotted. Note: Calling 
                                with freqoff = 0 will likely result in division by zero error. For best results, 
                                supply non-zero values.
            
        Optional Parameters:
            
            tauf            None, Float, or List of Floats 
                                [ Default = None ]
                                The values or values of total optical depth to be plotted. If None, plots all
                                values of tauf in self.taufs object attribute
                        
            label          None or String
                                [ Default = None ]
                                Text to label inside plot.
                        
            label_loc      String: 'left' or 'right'
                                [ Default = 'left' ]
                                The corner of the upper (ml) plot in which the label (if provided) will be 
                                placed.
                                Will probably want 'left' for lower optical depths and 'right' for higher optical
                                depths.
                                
            legend_cols     Integer
                                [ Default = 1 ]
                                Number of columns in the legend.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
        """
        #### First, checks values ####
        method_name = 'MASER_V_THETA.PLOT_MLMC'
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Checks that label_loc is one of the valid options and converts to lower case
        label_loc = label_loc.lower()
        if label is not None and label_loc not in [ 'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright' ]:
            err_msg = "{0}: Accepted values for label_loc are:\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "'left', 'right', 'upperleft', 'upperright', 'lowerleft', 'lowerright'."
            raise ValueError(err_msg)
        
            
        
        # Makes template of error message for attribute checks
        attr_missing_msg1 = method_name + ': Object attribute {0} does not exist.'
        attr_missing_msg2 = method_name + ': Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_shape_msg    = method_name + ': Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                        ' '*(12+len(method_name)+2) + 'is not consistent with attributes taufs, omegabar, and k.\n' + \
                        ' '*(12+len(method_name)+2) + 'Attribute {0} should be NumPy array of shape ( {3}, {4} ).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Ntaufs = self.taufs.size
        Nfreq  = self.omegabar.size - 2 * self.k
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for theta in self.thetas:
            
            for req_att in required_attributes:
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # If it does exist and it's not tau_idx, makes sure that the array shape is correct
                elif req_att != 'tau_idx':
                    if self.masers[theta].__dict__[req_att].shape != ( Ntaufs, Nfreq ):
                        raise ValueError( attr_shape_msg.format(req_att, theta, self.units, Ntaufs, Nfreq) )
        
        
        
        
        #### Does some processing on requested tauf value ####
        
        # If none is provided, assume all tauf values in taufs attribute are desired
        if tauf is None:
            tauf = list( self.taufs )
            tauf_idxs = list( np.arange( len(tauf) ) )
        else:
        
            # If tauf provided is single value, not list, makes it into len-1 list
            if isinstance( tauf, float ) or isinstance( tauf, int ):
                tauf = [ float(tauf) ]
            
            # Determines indices for each tauf value in taufs object attribute
            tauf_idxs = []
            for bval in tauf:
                if float(bval) in self.taufs:
                    tauf_idxs.append( np.where( self.taufs == float(bval) )[0][0] )
                else:
                    err_msg = '{0}: Requested tauf value, {1}, not in taufs object attribute.\n'.format(method_name, bval) + \
                              ' '*(12+len(method_name)+2) + \
                              'Please make sure that cloud_end_stokes attributes have been generated or read for\n' + \
                              ' '*(12+len(method_name)+2) + \
                              '    the desired tauf values before calling this method.'
                    raise ValueError(err_msg)
        
        
        
        
        #### Determine the colors and markers ####
        
        if len( tauf ) <= 7:
            color_list  = color_sets[ 7][:len(tauf)]
            marker_list = marker_sets[7][:len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        elif len(tauf) in [8,9]:
            color_list  = color_sets[ len(tauf)]
            marker_list = marker_sets[len(tauf)]
            fill_list = [ 'full', ] * len(tauf)
        else:
            color_list  = list( islice( cycle( color_sets[ 8] ), len(tauf) ))
            marker_list = list( islice( cycle( marker_sets[9] ), len(tauf) ))
            fill_template = [ 'full', ]*8
            fill_template.extend( ['none',]*8 )
            fill_list   = list( islice( cycle( fill_template ), len(tauf) ))
        
        
        
        
        #### Determine the frequency index ####
        
        # Makes sure that all maser objects in masers dictionary have same k value
        ks_for_theta = np.unique( np.array([ self.masers[ theta ].k for theta in self.thetas ]) )
        if ks_for_theta.size > 1:
            err_msg = '{0}: Maser objects in masers dictionary must all have same value of k.\n'.format(method_name) + \
                      ' '*(12+len(method_name)+2) + '({0} values found.)'.format( ks_for_theta.size )
            raise ValueError( err_msg )
        
        # Checks top-level object k value to make sure it's consistent with these
        if self.k not in ks_for_theta:
            err_msg = method_name + ': Maser_v_theta object must have the same value of k as objects in masers dictionary.'
            raise ValueError( err_msg )
        
        # Actually sets aside index of line center frequency
        jcenter = int( Nfreq / 2 )
        
        
        
        
        #### Actually plots ####
        
        # Calculates cos theta, which is used for the x-axis
        if self.units in ['degrees', 'deg', 'd']:
            costheta = np.cos( self.thetas * pi / 180. )
        else:
            costheta = np.cos( self.thetas )
        
        # Creates figure
        fig, ax = P.subplots(nrows=2, ncols=1, sharex=True, figsize = (5.5,4.5))
        fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.91 )
        
        # Begins iterating through the number of optical depths specified for plotting
        for i, bval in enumerate(tauf):
            
            # Gets the index that corresponds to this total optical depth in the stacked arrays
            tauf_idx = tauf_idxs[i]
            
            # Makes the lists of line center ml and mc values to plot
            plot_ml = [ self.masers[theta].stacked_ml[ tauf_idx , jcenter + freqoff ] for theta in self.thetas ]
            plot_mc = [ self.masers[theta].stacked_mc[ tauf_idx , jcenter + freqoff ] for theta in self.thetas ]
            
            # Actually plots with corresponding color/marker/fill
            ax[0].plot( costheta, plot_ml, marker = marker_list[i], \
                                                color = color_list[i], fillstyle = fill_list[i] )
            ax[1].plot( costheta, plot_mc, marker = marker_list[i], \
                                                color = color_list[i], fillstyle = fill_list[i], label=bval )
        
        
        
        
        
        #### Figure axes and labels ####
        
        # X-axis is cos(theta) so no units
        P.xlim( 0, 1 )
        ax[1].set_xlabel(r'$\cos \theta$')
            
        # Y-axis labels for both plots, making sure there's enough room on left hand side
        ax[0].set_ylabel(r'$m_l$')
        ax[1].set_ylabel(r'$m_c$')
        ax[1].set_xlabel(r'$\cos \theta$')
        P.subplots_adjust(left=0.15)
        
        # Use scientific notation for y-axis of both subplots, if small
        for i in range(len(ax)):
            ymin, ymax = ax[i].get_ylim()
            if ymax <= 1e-2:
                format_label_string_with_exponent( fig, ax[i], axis='y' )
        
        # Make the legend
        ax[1].legend(fontsize='small', ncol=legend_cols)
    
        # Makes frequency label
        d_angfreq = self.omegabar[ jcenter + freqoff ] - self.omegabar[jcenter]
        d_freq_Hz = d_angfreq / (2.*pi)
        if d_freq_Hz >= 0.0:
            freqsign = '+'
        else:
            freqsign = '-'
        if abs(d_freq_Hz) < 1000.0:
            freqlabel = '{0}{1} Hz'.format( freqsign, round(abs(d_freq_Hz)) )
        elif abs(d_freq_Hz) >= 1e3 and abs(dfreq) < 1e6:
            freqlabel = '{0}{1} kHz'.format( freqsign, round(abs(d_freq_Hz)*1e-3) )
        elif abs(d_freq_Hz) >= 1e6 and abs(dfreq) < 1e9:
            freqlabel = '{0}{1} MHz'.format( freqsign, round(abs(d_freq_Hz)*1e-6) )
        elif abs(d_freq_Hz) >= 1e9:
            freqlabel = '{0}{1} GHz'.format( freqsign, round(abs(d_freq_Hz)*1e-9) )
        
        # Adds freq to label, if provided; if not, sets label just as frequency
        if label is not None:
            label = freqlabel + '\n' + label
        else:
            label = freqlabel
    
        # Sets plot label, if requested.
        ymin, ymax = ax[0].get_ylim()
        if label_loc.lower() == 'left':
            ax[0].text( 0.02, ymax - (ymax -ymin )*0.05, label, ha='left', va='top', fontsize='large')
        elif label_loc.lower() == 'right':
            ax[0].text( 0.98, ymax - (ymax -ymin )*0.05, label, ha='right', va='top', fontsize='large')
        
        
        
        
        #### Saves/Shows Figure ####
        
        # Saves figure if requested
        if figname is not None:
            try:
                fig.savefig( figname )
            except:
                print('Unable to save figure to {0}'.format(figname))
    
        # Finally, shows the plot, if requested
        if show:
            P.show()
        else:
            P.close()
        
    def plot_R( self, R_tauf_theta, norm = False, aslog = False, taufmax = None, plottaufmax = None, \
                        interp = 'cubic', fig_title = None, figname = None, show = True, verbose = True ):
        """
        Plots provided stimulated emission rate, R, vs. theta (x-axis) and total optical depth (y-axis).
        
        R plotted may be scaled by gOmega (the total Zeeman splitting rate) or Gamma (the loss rate, 
        must be provided) or plotted on its own. Any of these may be optionally plotted on log scale.
        
        Required Parameters:
            
            R_tauf_theta    2D NumPy Array
                                The stimulated emission rate, in inverse seconds, at the cloud end.
                                2-dimensional array with shape ( number_of_tauf, number_of_theta ).
                                Calculated R in each dimension should correspond to the object 
                                attributes, taufs and thetas.
            
        Optional Parameters:
            
            norm            False, String ('gOmega'), or Float/Integer
                                [ Default = False ]
                                Indicates any normalization that occurs to R before plotting. 
                                If False, no normalization is done.
                                If 'gOmega', plots R / gOmega, where gOmega is the full width of the
                                Zeeman splitting (across all substates), calculated as 
                                2*k*omegabar_bin_width, and converted to frequency.
                                If Float or Integer, plots R / Gamma, where Gamma is the loss rate,
                                and assumes that the value provided for norm is the loss rate, 
                                Gamma, in inverse seconds.
            
            aslog           Boolean True/False
                                [ Default = False ]
                                If True, will plot the base-10 log of R (normalized first by any 
                                value indicated with keyword, norm).
            
            taufmax         None or Float 
                                [ Default = None ]
                                Maximum value of tauf to show data for in the plot. If None, plots 
                                all available data.
                                
            plottaufmax     Float or None
                                [ Default = None ]
                                Y-limit (optical depth) shown on the plot axes. If None, will be the 
                                same as taufmax. (Only useful if you want to set the y-limit used by
                                the figure to be the same as other figures despite not having tauf up 
                                to that value for this parameter set.)
                                
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'.
                                
            fig_title       None or String 
                                [ Default = None ]
                                If not None, provided string will be added as figure title.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        
        method_name = 'MASER_V_THETA.PLOT_R'
        
        
        
        
        
        
        
        #### Processing defaults for taufmax and plottaufmax ####
        
        # Finds index of maximum tauf present for all ml arrays
        if taufmax is not None:
        
            # Finds index of taufmax in taufs array
            if taufmax in self.taufs:
                ibmax = np.where( self.taufs == taufmax )[0][0]
                Ntaufs = ibmax + 1
            else:
                err_msg = method_name + ": Requested tauf value, {0}, not in taufs attribute array.\n".format(taufmax) 
                raise ValueError(err_msg)
        
        # If using default taufmax, just uses last one with stokes for all theta values present
        else:
            Ntaufs = self.taufs.size
            ibmax = Ntaufs - 1
            taufmax = self.taufs[ibmax]
        
        # Sets default plottaufmax, if not set
        if plottaufmax is None:
            plottaufmax = taufmax
        
        
        
        
        #### First, checks values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Checks if taufs attribute exists; thetas must unless deleted since it's created by the init
        if 'taufs' not in self.__dict__.keys():
            attr_missing_msg  = method_name + ': Object attribute {0} does not exist.'
            raise AttributeError( attr_missing_msg.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the 
        #    R_tauf_theta array
        Nthetas = self.thetas.size
        Ntaufs  = self.taufs.size
        
        # Checks that the R_tauf_theta is a Numpy Array
        if not isinstance( R_tauf_theta, np.ndarray ):
            R_type_msg = method_name + ': R_tauf_theta must be a NumPy array.'
            raise TypeError( R_type_msg )
        
        # If it is a numpy array, makes sure it's 2-dimensional
        elif R_tauf_theta.ndim != 2:
            R_type_msg = method_name + ': R_tauf_theta must be a 2-dimensional NumPy Array. (Current dimensions: {0}).'
            raise ValueError( R_type_msg.format( R_tauf_theta.ndim ) )
        
        # If it is 2-dimensional, checks the axis sizes, starting with the tauf axis
        elif R_tauf_theta.shape[0] != Ntaufs:
            R_shape_tauf_msg  = method_name + ': Shape of R_tauf_theta is not consistent with taufs object attribute.\n' + \
                                ' '*(12+len(method_name)+2) + \
                                'R_tauf_theta should be NumPy array of shape ( {0}, {1} ). (Current shape ( {2}, {3} ).)'
            raise ValueError( R_shape_tauf_msg.format( Ntaufs, Nthetas, *R_tauf_theta.shape ) )
        
        # then checks theta axis
        elif R_tauf_theta.shape[1] != Nthetas:
            R_shape_theta_msg = method_name + ': Shape of R_tauf_theta is not consistent with thetas object attribute.\n' + \
                                ' '*(12+len(method_name)+2) + \
                                'R_tauf_theta should be NumPy array of shape ( {0}, {1} ). (Current shape ( {2}, {3} ).)'
            raise ValueError( R_shape_theta_msg.format(  Ntaufs, Nthetas,*R_tauf_theta.shape ) )
        
        
        # Checks that value provided for norm is accepted
        # First, assumes that if None, they meant False
        if norm is None:
            norm = False
        # If a string is provided, checks if it's gomega and makes lower case (not case sensitive)
        elif isinstance( norm, str ):
            if norm.lower() == 'gomega':
                norm = norm.lower()
            # If it isn't tries to convert to a float
            else:
                try:
                    norm = float( norm )
                except:
                    norm_string_msg = method_name + ': String provided for norm not recognized.\n' + \
                                ' '*(12+len(method_name)+2) + \
                                "Accepted values are False (Boolean), 'gOmega' (string), or a Float or Integer."
                    raise ValueError( norm_string_msg )
        # Otherwise, if not False, makes sure it's a float
        elif norm:
            norm = float(norm)
        
        
        
        
        
        
        
        
        
        
        #### Applies Normalization ####
        
        # First, does it if there's no normalization
        if not norm:
            
            # Array to plot is just R_tauf_theta
            temparray = R_tauf_theta
            
            # Sets aside the name of what we're plotting for the title and its units
            plotvalue = 'R'
            plotunits = r's$^{-1}$'
            
        
        # Does the same if we're normalizing by gOmega
        elif isinstance( norm, str ):
            
            # Calculates gOmega and converts from angular frequency to frequency
            gOmega = float(self.k) * float( self.omegabar[1]-self.omegabar[0] )  / pi 
            
            # Array to plot is R_tauf_theta divided by that value
            temparray = R_tauf_theta / gOmega
            
            # Sets aside the name of what we're plotting for the title and its units
            plotvalue = r'R/g$\Omega$'
            plotunits = None
        
        
        # Finally, if a value is provided for norm, assume's it's Gamma in inverse seconds
        else:
            
            # Array to plot is R_tauf_theta divided by that value
            temparray = R_tauf_theta / norm
            
            # Sets aside the name of what we're plotting for the title and its units
            plotvalue = r'R/$\Gamma$'
            plotunits = None
            
        
        
        
        
        
        
        
        
        #### Applies log scaling, if requested ####
        
        if aslog:
            
            # Takes the base-10 log of the array to plot
            temparray = np.log10( temparray )
            
            # Updates the plotted value name
            plotvalue = r'log(' + plotvalue + r')'
            
            # Updates the units, if any
            if plotunits is not None:
                plotunits = r'log(' + plotunits + r')'
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        #### Regridding array for smooth distribution of theta and tauf solutions ####
        
        if temparray.size != 0:
            
            if verbose:
                print('Regridding data...')
            
            # Ravels temparray to prepare for regridding
            # Before ravel, has shape (tauf, theta)
            temparray = np.ravel( temparray )
        
            # Creates grid of existing (theta, tauf) points
            # Resulting arrays have shape ( tauf, theta ) when meshgrid of (theta, tauf)
            thetapts, taufpts = np.meshgrid( self.thetas, self.taufs[:ibmax+1] )
            thetapts = np.ravel( thetapts )
            taufpts  = np.ravel( taufpts )
            points   = np.vstack(( thetapts, taufpts )).T
            
            # Creates grid of desired theta and tauf values and regrids
            #   Again, thetagrid and taufgrid have shape ( 1001, 36 )
            thetagoal = np.linspace( self.thetas.min(), self.thetas.max(), num=36 )
            taufgoal  = np.linspace( self.taufs[0], taufmax, 1001)
            thetagrid, taufgrid = np.meshgrid( thetagoal, taufgoal )
            
            # Sets aside resolution of each for later use
            dtheta = thetagoal[1] - thetagoal[0]
            dtauf  = taufgoal[1]  - taufgoal[0]
            
            # Re-grids data array with desired interpolation
            # Resulting zs has shape ( Nthetagoal, Ntaufgoal ), like thetagrid and taufgrid
            zs = griddata( points, temparray, (thetagrid, taufgrid), method=interp)
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            # Color map
            cmap = 'viridis'
            
            # Clears any existing figures and makes plot
            P.close()
            P.imshow( zs, aspect='auto', origin='lower', cmap = cmap, \
                      extent = [self.thetas.min() - dtheta/2., self.thetas.max() + dtheta/2., \
                      self.taufs[0] - dtauf/2., taufmax + dtauf/2.] )
            
            # Axis limits
            P.xlim( self.thetas.min(), self.thetas.max() )
            P.ylim( self.taufs[0], plottaufmax )
            
            # Axis labels; x-axis label depends on theta units
            if self.units in ['degrees','deg','d']:
                P.xlabel(r'$\theta$ [$^{\circ}$]')
            else:
                P.xlabel(r'$\theta$ [radians]')
            P.ylabel(r'Total $\tau_f$')
            
            # Colorbar
            cbar = P.colorbar()
            if np.nanmax(zs) <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
            
            # Combines and sets as colorbar label
            if plotunits is not None:
                unit_str = ' [ ' + plotunits + ' ]'
            else:
                unit_str = ''
            cbar.set_label( plotvalue + unit_str )
            
            # If plot title requested, sets
            if fig_title is not None:
                P.title( fig_title )
        
            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                    if verbose:
                        print('Saved figure to {0}.'.format(figname))
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
    
    def plot_theta_logR( self, plotvalue,  R_tauf_theta, freqoff = 0, taufmax = None, ylims = None, contours = False, \
                        convert_freq = True, interp = 'cubic', summed = False, subtitle = None, \
                        figname = None, show = True, verbose = True ):
        """
        Plots desired value at some offset from line center vs. theta (x-axis) and log(R/gOmega) (y-axis).
        
        Can be used to plot mc, ml, evpa, stoki, stokq, stoku, stokv, fracq (stokq/stoki), fracu (stoku/stoki), 
        quv (sqrt(q^2+u^2)/i-v/i), mlmc (ml-mc), or vmetric ( Vmax - Vmin ) / Imax.
        
        Intended to be run *after* stokes at a given point in cloud have been read in for a range of tauf values
        with cloud_end_stokes method AND using R_tauf_theta array calculated with calc_R method.
        
        Required Parameters:
            
            plotvalue       String
                                What to plot. Options are 'mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 
                                'fracq' (for stokq/stoki), 'fracu' for (stoku/stoki), 'quv' for sqrt(q^2+u^2)/i-v/i, 
                                'mlmc' for (ml-mc), or 'vmetric' for ( Vmax - Vmin ) / Imax.
                                Note: 'vmetric' plotvalue only available for freqoff = None. 
            
            R_tauf_theta    2D NumPy Array
                                The stimulated emission rate, in inverse seconds, at the cloud end. 2-dimensional 
                                array with shape ( number_of_tauf, number_of_theta ). Calculated R in each 
                                dimension should correspond to the object attributes, taufs and thetas.
            
        Optional Parameters:
            
            freqoff         Integer, list of Integers, or None
                                [ Default = 0 ]
                                If a single integer, value gives the offset index (with respect to the central  
                                frequency in omegabar) to be plotted.
                                If a list of integers, individual values should be the same as in the case of a 
                                single integer, but plotted output will be summed across those wavelength bins.
                                If None, calculates the desired plotvalue from the peak stokes values across 
                                angular frequency for each theta and log(R/gOmega). 
                                NOTE: None is NOT the same as 0!!!
            
            taufmax         None or Float 
                                [ Default = None ]
                                Maximum value of tauf to show data for in the plot. If None, plots all available 
                                data.
                                
            ylims           Length-2 tuple/List or None
                                [ Default = None ]
                                The y-axis (log(R/gOmega)) limits (ymin, ymax) for the plot. If None, uses
                                matplotlib default based on data.
            
            contours        Boolean
                                [ Default = False ]
                                Whether (True) or not (False) to overplot contours on the image.
                        
            convert_freq    Boolean 
                                [ Default = True ]
                                Whether to convert omegabar from angular frequency (s^-1) to frequency (MHz) 
                                (if True) or not (if False). Only used for labelling frequency plotted.
                                
            interp          String
                                [ Default = 'cubic' ]
                                Type of interpolation to use for image re-gridding. Default is 'cubic'. Other 
                                options are 'linear' and 'nearest'. 
                                Note: 'nearest' will not properly cut off interpolation past valid range of 
                                log(R/gOmega) values. Not recommended.
                                
            summed          Boolean
                                [ Default = False ]
                                Whether the R provided is summed over all n (True) or only calculated at line
                                center (False). Only affects the y-axis label; if summed is False, R will 
                                appear as R_0.
                                
            subtitle        None or String 
                                [ Default = None ]
                                If not None, provided string will be added to title as a second line. Intended 
                                to be used to indicate faraday polarization values used if not 0.
                                
            figname         None or String
                                [ Default = None ]
                                If a string is provided, figure will be saved with the provided file path/name. 
                                Note: this is the path from the working directory, NOT within the outpath of 
                                the object. 
                                If None, figure will be shown but not saved.
            
            show            Boolean
                                [ Default = True ]
                                Whether to show the figure or just close after saving (if figname provided).
                                Note: If this is set to False, you must set figname to be a string to which
                                the resulting figure will be saved; otherwise, the plot will disappear unseen.
                                
            verbose         Boolean
                                [ Default = True ]
                                Whether to print feedback to terminal at various stages of the process.
        """
        
        method_name = 'MASER_V_THETA.PLOT_THETA_LOGR'
        
        
        
        
        #### Processing default for taufmax ####
        
        # Finds index of maximum tauf present for all ml arrays
        if taufmax is not None:
        
            # Finds index of taufmax in taufs array
            if taufmax in self.taufs:
                ibmax = np.where( self.taufs == taufmax )[0][0]
                Ntaufs = ibmax + 1
            else:
                err_msg = method_name + ": Requested tauf value, {0}, not in taufs attribute array.\n".format(taufmax) 
                raise ValueError(err_msg)
        
        # If using default taufmax, just uses last one with stokes for all theta values present
        elif taufmax is None:
            Ntaufs = self.taufs.size
            ibmax = Ntaufs - 1
            taufmax = self.taufs[ibmax]
            
            
            
            
        #### First, checks values ####
        
        #### First, checks values ####
        
        # Makes sure that, if show is False, a figname has been specified
        if show is False and figname is None:
            err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(method_name) + \
                      " "*(12+len(method_name)+2) + \
                      "figure produced. Please either set show = True or provide a figname for the plot."
            raise ValueError(err_msg)
        
        # Makes template of error message for attribute checks
        attr_missing_msg1 =  method_name + ': Object attribute {0} does not exist.'
        attr_missing_msg2 =  method_name + ': Object attribute {0} does not exist for maser object with theta = {1} {2}.'
        attr_ndim_msg     =  method_name + ': Object attribute {0} for maser object with theta = {1} {2} must be\n' + \
                     ' '*(12+len(method_name)+2) + '2-dimensional. (Current number of dimensions = {3}).'
        attr_dim0_msg     =  method_name + ': Shape of object attribute {0} for maser object with theta = {1} {2}\n' + \
                     ' '*(12+len(method_name)+2) + 'is not consistent with attributes omegabar and k.\n' + \
                     ' '*(12+len(method_name)+2) + 'Should be 2D NumPy array with {3} values along 0-axis (currently {4}).'
        
        # Iterates through required keywords to make sure the attributes exist; checks top level object first
        for req_attr in ['tau_idx', 'taufs']:
            if req_attr not in self.__dict__.keys():
                raise AttributeError( attr_missing_msg1.format(req_attr) )
            
        # Since taufs attribute must exist if we made it here, figures out the expected dimensions of the attribute 
        #   arrays
        Nfreq  = self.omegabar.size - 2 * self.k
        Nthetas = self.thetas.size
        
        # Then checks maser objects in masers dictionary
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa', 'tau_idx' ]
        for req_att in required_attributes:
            for theta in self.thetas:
                
                # Checks if the attribute exists
                if req_att not in self.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg2.format(req_att, theta, self.units) )
                
                # For those that are arrays, check their dimensions and freq values
                if req_att != 'tau_idx':
                
                    # Checks that the attribute is a 2d array
                    if self.masers[theta].__dict__[req_att].ndim != 2:
                        raise AttributeError( attr_ndim_msg.format(req_att, theta, self.units, self.masers[theta].__dict__[req_att].ndim ) )
                
                    # Checks number of values along 1st (frequency) axis
                    elif self.masers[theta].__dict__[req_att].shape[1] != Nfreq:
                        raise AttributeError( attr_dim0_msg.format(req_att, theta, self.units, Nfreq, self.masers[theta].__dict__[req_att].shape[0] ) )
            
            # Checks size of 0th (tauf) axis of those attribute arrays for every theta value
            if req_att != 'tau_idx':
                Ntauf_per_theta = np.array([ self.masers[theta].__dict__[req_att].shape[0] for theta in self.thetas ])
            
                # If not every maser object has the same number of taufs (and this hasn't been done for a previous 
                #   attribute), prints a warning and adjusts number of taufs
                if np.unique( Ntauf_per_theta ).size != 1 and Ntauf_per_theta.min() != Ntaufs:
                    warn_msg = method_name + ' WARNING: Optical depths in object attribute {0} not consistent across theta.\n' + \
                                ' '*(len(method_name)+10) + 'Highest value of optical depth present for all theta is {1}. ({2} values.)'
                    print( warn_msg.format( req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                    Ntaufs = Ntauf_per_theta.min()
        
                # If every maser object does have the same number of taufs but its less than that expected from Ntaufs, 
                #   prints a warning and adjusts number of taufs
                elif  Ntauf_per_theta.min() != Ntaufs:
                    warn_msg = method_name + ' WARNING: Optical depths in object attribute {0} not consistent with taufs attribute.\n' + \
                                ' '*(len(method_name)+10) + 'Highest value of optical depth present for all theta is {1}. ({2} values.)'
                    print( warn_msg.format( req_att, self.taufs[ Ntauf_per_theta.min() - 1 ], Ntauf_per_theta.min() ))
                    Ntaufs = Ntauf_per_theta.min()
        
        # Makes sure that specified plotvalue is allowed:
        if plotvalue not in ['mc', 'ml', 'evpa', 'stoki', 'stokq', 'stoku', 'stokv', 'fracq', 'fracu', 'vmetric', 'quv', 'mlmc']:
            err_msg = method_name + ": plotvalue '{0}' not recognized.\n".format(plotvalue) + \
                        ' '*(12+len(method_name)+2) + "Allowed values are 'mc', 'ml', 'evpa', 'stoki', 'stokq',\n" + \
                        ' '*(12+len(method_name)+2) + "'stoku', 'stokv', 'fracq', or 'fracu'."
            raise ValueError(err_msg)
        
        # Checks that the R_tauf_theta is a Numpy Array
        if not isinstance( R_tauf_theta, np.ndarray ):
            R_type_msg = method_name + ': R_tauf_theta must be a NumPy array.'
            raise TypeError( R_type_msg )
        
        # If it is a numpy array, makes sure it's 2-dimensional
        elif R_tauf_theta.ndim != 2:
            R_type_msg = method_name + ': R_tauf_theta must be a 2-dimensional NumPy Array. (Current dimensions: {0}).'
            raise ValueError( R_type_msg.format( R_tauf_theta.ndim ) )
        
        # If it is 2-dimensional, checks the axis sizes, starting with the tauf axis
        elif R_tauf_theta.shape[0] != Ntaufs:
            R_shape_tauf_msg  = method_name + ': Shape of R_tauf_theta is not consistent with taufs object attribute.\n' + \
                                ' '*(12+len(method_name)+2) + \
                                'R_tauf_theta should be NumPy array of shape ( {0}, {1} ). (Current shape ( {2}, {3} ).)'
            raise ValueError( R_shape_tauf_msg.format( Ntaufs, Nthetas, *R_tauf_theta.shape ) )
        
        # then checks theta axis
        elif R_tauf_theta.shape[1] != Nthetas:
            R_shape_theta_msg = method_name + ': Shape of R_tauf_theta is not consistent with thetas object attribute.\n' + \
                                ' '*(12+len(method_name)+2) + \
                                'R_tauf_theta should be NumPy array of shape ( {0}, {1} ). (Current shape ( {2}, {3} ).)'
            raise ValueError( R_shape_theta_msg.format(  Ntaufs, Nthetas,*R_tauf_theta.shape ) )
        
        # Checks format of provided ylims
        if ylims is not None:
            
            # If it's a list-like object, makes sure it's a list and checks length
            if isinstance( ylims, list ) or isinstance( ylims, tuple ) or isinstance( ylims, np.ndarray ):
                ylims = list( ylims )
                if len( ylims ) != 2:
                    err_msg = method_name + ': List provided for ylims must be length 2 ( ymin, ymax ). Current length: {0}'
                    raise ValueError( err_msg.format( len(ylims) ) )
            
            # If it's not a list, raises an error
            else:
                err_msg = method_name + ': Value provided for ylims must be either None or length-2 list/tuple/NumPy array.'
                raise ValueError( err_msg )
        
        
        
        
        
        
        
        
        
        #### Determine the frequency index ####
        
        # Makes sure that all maser objects in masers dictionary have same k value
        ks_for_theta = np.unique( np.array([ self.masers[ theta ].k for theta in self.thetas ]) )
        if ks_for_theta.size > 1:
            raise ValueError( method_name + ': Maser objects in masers dictionary must all have same value of k.' + \
                              ' ({0} values found.)'.format( ks_for_theta.size ) )
        
        # Checks top-level object k value to make sure it's consistent with these
        if self.k not in ks_for_theta:
            raise ValueError( method_name + ': Maser_v_theta object must have the same value of k as objects in masers dictionary.' )
        
        # Actually sets aside index of line center frequency
        jcenter = int( Nfreq / 2 )
            
        
        
        #### Determining which array to plot ####
        
        # Start with Stokes i
        if plotvalue == 'stoki':
            
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    thetaline = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
            
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes i'
            cmap = 'viridis'
        
        # Next, Stokes q
        elif plotvalue == 'stokq':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q'
            cmap = 'RdBu'
        
        # Next, Stokes u
        elif plotvalue == 'stoku':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u'
            cmap = 'RdBu'
        
        # Next, Stokes v
        elif plotvalue == 'stokv':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    min_stokv = np.min( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    thetaline = np.array([ ( max_stokv[j] - min_stokv[j] ) for j in range(ibmax+1) ])
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Range in Stokes v'
                    cmap = 'viridis'
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + freqoff ]
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Stokes v'
                    cmap = 'RdBu'
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    # Sets plot title & colormap, while we're here
                    fig_title = r'Stokes v'
                    cmap = 'RdBu'
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
        # Next, fractional stokes q
        elif plotvalue == 'fracq':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ] / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) / np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) 
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes q/i'
            cmap = 'RdBu'
        
        # Next, fractional stokes u
        elif plotvalue == 'fracu':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ] / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    thetaline = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) / np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 ) 
                thetaline = thetaline.reshape( thetaline.size, 1)
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'Stokes u/i'
            cmap = 'RdBu'
        
        # Next, does ml
        elif plotvalue == 'ml':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_ml[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_ml[ : ibmax+1 , jcenter + freqoff ]
                else:
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = np.sqrt( qsum**2 + usum**2 ) / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l$'
            cmap = 'viridis'
        
        # Next, does mc
        elif plotvalue == 'mc':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    
                    # Calculation if we want peak mc to be the mc where stokes v is max, not the maximum value of mc
                    #max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    #peak_idx = np.array([ np.where( self.masers[theta].stacked_stokv[j,:] == x )[0][0] for j,x in enumerate(max_stokv) ])
                    #thetaline = np.array([ max_stokv[j] / self.masers[ theta ].stacked_stoki[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                    
                    # Calculation if we want the peak mc to mean the actual max mc
                    thetaline = np.max( self.masers[theta].stacked_mc[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'viridis'
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta       ].stacked_mc[ : ibmax+1 , jcenter + freqoff ]
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'RdBu'
                else:
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = vsum / isum
                    # Sets plot title & colormap, while we're here
                    fig_title = r'$m_c$'
                    cmap = 'RdBu'
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
        # Finally, does evpa
        elif plotvalue == 'evpa':
        
            # Build array with plotable range
            # Note: np.arctan2 returns values between -pi and +pi, with 0.5 * np.arctan2 returning values between
            #       -pi/2 and +pi/2. Applying + pi % pi to this changes the range to 0 - pi, flipping only the negative
            #       values to the new quadrant.
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    thetaline = np.array([ self.masers[ theta ].stacked_evpa[ j, peak_idx[j] ] + pi for j in range(ibmax+1) ]) % pi
                elif isinstance(freqoff,int):
                    thetaline = ( self.masers[ theta      ].stacked_evpa[ : ibmax+1 , jcenter + freqoff ] + pi ) % pi
                else:
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = 0.5 * np.arctan2( usum, qsum )
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'EVPA'
            cmap = 'RdBu'
        
        # New option: vmetric plots (Vmax - Vmin) / Imax (only available with freqoff=None)
        elif plotvalue == 'vmetric':
            
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                min_stokv = np.min( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                thetaline = np.array([ ( max_stokv[j] - min_stokv[j] ) / max_stoki[j] for j in range(ibmax+1) ])
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$(V_{max} - V_{min}) / I_{max}$'
            cmap = 'viridis'
        
        # New option: quv plots sqrt( q**2 + u**2 )/i - v/i
        elif plotvalue == 'quv':
    
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx  = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    max_stokv = np.max( self.masers[theta].stacked_stokv[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    qupeak    = np.sqrt( np.array([ self.masers[ theta ].stacked_stokq[ j, peak_idx[j] ] for j in range(ibmax+1) ])**2 + \
                                         np.array([ self.masers[ theta ].stacked_stoku[ j, peak_idx[j] ] for j in range(ibmax+1) ])**2 )
                    thetaline = ( qupeak - max_stokv ) / max_stoki
                elif isinstance(freqoff,int):
                    thetaline = np.sqrt( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + freqoff ]**2 + \
                                         self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + freqoff ]**2 ) - \
                                         self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + freqoff ]
                    thetaline = thetaline / self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + freqoff ]
                else:
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = np.sqrt( qsum**2 + usum**2 ) - vsum
                    thetaline = thetaline / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
    
            # Sets plot title & colormap, while we're here
            fig_title = r'$(q^2 + u^2)^{1/2}/i - v/i$'
            cmap = 'RdBu'
        
        # New option: mlmc plots ml - mc
        elif plotvalue == 'mlmc':
        
            # Build array with plotable range
            for i,theta in enumerate(self.thetas):
                if freqoff is None:
                    max_stoki = np.max( self.masers[theta].stacked_stoki[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    peak_idx  = np.array([ np.where( self.masers[theta].stacked_stoki[j,:] == x )[0][0] for j,x in enumerate(max_stoki) ])
                    max_mc    = np.max( self.masers[theta].stacked_mc[ : ibmax+1 , 2*self.k:-2*self.k ], axis=1 )
                    ml_peak   = np.array([ self.masers[ theta ].stacked_ml[ j, peak_idx[j] ] for j in range(ibmax+1) ])
                    thetaline = ml_peak - max_mc
                elif isinstance(freqoff,int):
                    thetaline = self.masers[ theta ].stacked_ml[ : ibmax+1 , jcenter + freqoff ] - \
                                self.masers[ theta ].stacked_mc[ : ibmax+1 , jcenter + freqoff ]
                else:
                    isum = np.sum( self.masers[ theta ].stacked_stoki[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    qsum = np.sum( self.masers[ theta ].stacked_stokq[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    usum = np.sum( self.masers[ theta ].stacked_stoku[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    vsum = np.sum( self.masers[ theta ].stacked_stokv[ : ibmax+1 , jcenter + np.array(freqoff) ], axis=1 )
                    thetaline = ( np.sqrt( qsum**2 + usum**2 ) - vsum ) / isum
                thetaline = thetaline.reshape( thetaline.size, 1)
                if i != 0:
                    temparray = np.hstack(( temparray, thetaline ))
                else:
                    temparray = np.array( thetaline )
        
            # Sets plot title & colormap, while we're here
            fig_title = r'$m_l - m_c$'
            cmap = 'RdBu'
            
            
            
            
        
        
        
        
        #### Regridding array for smooth distribution of theta and tauf solutions ####
        
        if temparray.size != 0:
            
            # Calculates gOmega
            gOmega = float(self.k) * float( self.omegabar[1]-self.omegabar[0] )  / pi 
            
            # Calculates array of log(R/gOmega); should still have shape ( Ntauf, Ntheta )
            logRpts = np.log10( R_tauf_theta / gOmega )
            
            
            if interp is not None:
                if verbose:
                    print('Regridding data...')
            
                # Ravels temparray to prepare for regridding
                # Before ravel, has shape (tauf, theta)
                temparray = np.ravel( temparray )
        
                # Creates grid of existing (theta, tauf) points
                # Resulting arrays have shape ( tauf, theta ) when meshgrid of (theta, tauf)
                thetapts, taufpts = np.meshgrid( self.thetas, self.taufs[:ibmax+1] )
            
                # Creates point locations, but instead of taufpts values, want 
                #   values of log10(R_theta_tauf/gOmega)
                thetapts = np.ravel( thetapts )
                logRpts  = np.ravel( logRpts )
                points   = np.vstack(( thetapts, logRpts )).T
            
                # Creates grid of desired theta and log(R/gOmega) values and regrids
                #   Again, thetagrid and logRgrid have shape ( 1001, 36 )
                thetagoal = np.linspace( self.thetas.min(), self.thetas.max(), num=36 )
                logRgoal  = np.linspace( np.nanmin(logRpts), np.nanmax(logRpts), 1001)
                thetagrid, logRgrid = np.meshgrid( thetagoal, logRgoal )
            
                # Sets aside resolution of each for later use
                dtheta = thetagoal[1] - thetagoal[0]
                dlogR  = logRgoal[1]  - logRgoal[0]
            
                # Re-grids data array with desired interpolation
                # Resulting zs has shape ( Nthetagoal, NlogRgoal )
                zs = griddata( points, temparray, (thetagrid, logRgrid), method=interp)
            
            
            
            
            
            
            
        
        
                #### Figures out min and max for color ####
            
                if verbose:
                    print('Plotting figure...')
            
                # zs shape is (taufgoal, thetagoal). no freq edge effects
                vmax = np.nanmax(np.abs( zs ))
                if plotvalue == 'evpa':
                    vmin = 0.0
                    vmax = pi
                elif cmap == 'RdBu':
                    vmin = -1.0 * vmax
                else:
                    vmin = 0.0
            
            # Does the same if skipping interpolation
            else:
                if verbose:
                    print('Plotting figure...')
            
                # zs shape is (taufgoal, thetagoal). no freq edge effects
                vmax = np.nanmax(np.abs( temparray ))
                if plotvalue == 'evpa':
                    vmin = 0.0
                    vmax = pi
                elif cmap == 'RdBu':
                    vmin = -1.0 * vmax
                else:
                    vmin = 0.0
                
            
            
            
            
            
            
            
            
            
        
        
        
            #### Actually plotting ####
            
            # Makes figure
            P.close()
            fig, ax = P.subplots(nrows=1, ncols=1, figsize = (5.5,4.5))
            fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.88 )
                
            # If interpolation is being performed, plots image with optional overlayed (monochromatic) contours
            if interp is not None:
                
                # Plots image
                im = ax.imshow( zs, aspect='auto', origin='lower', vmin = vmin, vmax = vmax, cmap = cmap, \
                                extent = [self.thetas.min() - dtheta/2., self.thetas.max() + dtheta/2., \
                                logRgoal[0] - dlogR/2., logRgoal[-1] + dlogR/2.] )
        
                # Adds contour, if requested
                if contours and plotvalue != 'evpa':
                    levels = gen_contour_levels( vmin, vmax, min_contours = 4 )
                    contour_color = [ 'w', ]*len(levels)
                    if cmap == 'RdBu':
                        #print(levels)
                        contour_color[ levels.index(0.0) ] = 'lightgray'
                    _cs2 = ax.contour( thetagrid, logRgrid, zs, levels=levels, origin='lower', colors=contour_color )
                
                # If making contours for evpa plot, central contour will need to be black
                elif contours:
                    levels_degrees = [ 30, 60, 90, 120, 150 ]
                    levels = [ float(x) * pi / 180. for x in levels_degrees ]
                    _cs2 = ax.contour( thetagrid, logRgrid, zs, levels=levels, origin='lower', \
                        colors=['w','w','lightgray','w','w'] )
                    
                    
            # If no interpolation, plots contours only but in a colormap gradient with labels
            else:
                if plotvalue == 'evpa':
                    levels_degrees = [ 30, 60, 90, 120, 150 ]
                    levels = [ float(x) * pi / 180. for x in levels_degrees ]
                else:
                    levels = gen_contour_levels( vmin, vmax, min_contours = 4 )
                thetapts, taufpts = np.meshgrid( self.thetas, self.taufs[:ibmax+1] )
                _cs2 = ax.contour( thetapts, logRpts, temparray, levels=levels, origin='lower', cmap = 'viridis' )
                
            
            # Axis limits
            ax.set_xlim( self.thetas.min(), self.thetas.max() )
            if ylims is not None:
                ax.set_ylim( *ylims )
            else:
                ax.set_ylim( np.nanmin(logRpts), np.nanmax(logRpts) )
            
            # Axis labels; x-axis label depends on theta units
            if self.units in ['degrees','deg','d']:
                ax.set_xlabel(r'$\theta$ [$^{\circ}$]')
            else:
                ax.set_xlabel(r'$\theta$ [radians]')
            if summed:
                ax.set_ylabel(r'log( $R/g\Omega$ )')
            else:
                ax.set_ylabel(r'log( $R_0/g\Omega$ )')
            
            
            
            # Makes colorbar; will have ticks at overplotted contour lines, if contours requested
            if interp is not None:
                
                # Makes just the colorbar with correct ticks
                if plotvalue == 'evpa':
                    levels_degrees = [ 0, 30, 60, 90, 120, 150, 180 ]
                    cbar = fig.colorbar( im, ax=ax, ticks = [ float(x)*pi/180. for x in levels_degrees ] )
                    tick_labels = ['0', '', '', r'$\pi$/2', '', '', r'$\pi$']
                    cbar.ax.set_yticklabels( tick_labels )
                    if contours:
                        cbar.add_lines( _cs2 )
                    
                elif contours:
                    cbar = fig.colorbar( im, ax=ax, ticks = levels )
                    cbar.add_lines( _cs2 )
                
                else:
                    cbar = fig.colorbar( im, ax=ax )
            
            # If no interpolation, can only make contour plot, so just labels contours on figure, no colorbar
            else:
                if plotvalue == 'evpa':
                    mk_clabel = lambda crad : str( round( crad * 180. / pi ) )
                    ax.clabel(_cs2, fontsize=9, inline=1, fmt=mk_clabel, colors='k' )
                    cbar = None
                else:
                    ax.clabel(_cs2, fontsize=9, inline=1, fmt='%.1e', colors='k' )
                    cbar = None
            
            # Converts colorbar ticks to scientific notation if they're small; won't affect evpa plots
            if cbar is not None and vmax <= 1e-2:
                cbar.ax.ticklabel_format( axis='y', style = 'sci', scilimits = (0,0) )
            
            # Determine plot title; depends on frequency with respect to line center
            if isinstance(freqoff,int) and freqoff == 0:
                freqtitle = 'Line Center'
            elif freqoff is None:
                freqtitle = 'Peak'
            elif isinstance(freqoff,int):
                # Gets frequency offset in ANGULAR freq (s^-1)
                dfreq = self.omegabar[ jcenter + freqoff ] - self.omegabar[jcenter]
            
                 # If frequency conversion to MHz is requested, converts and generates freq_string label
                if convert_freq:
            
                    # Converts from angular frequency to frequency & Creates label
                    dfreq = dfreq / (2.*pi)
                    if dfreq < 1000.0:
                        freqtitle = '{0} Hz from Line Center'.format( round(dfreq,0) )
                    elif dfreq >= 1e3 and dfreq < 1e6:
                        freqtitle = '{0} kHz from Line Center'.format( round(dfreq*1e-3,0) )
                    elif dfreq >= 1e6 and dfreq < 1e9:
                        freqtitle = '{0} MHz from Line Center'.format( round(dfreq*1e-6,0) )
                    elif dfreq >= 1e9:
                        freqtitle = '{0} GHz from Line Center'.format( round(dfreq*1e-9,0) )
        
                # If no frequency conversion requested, makes frequency label
                else:
                    freqtitle = r'$\varpi =$ {0:.2e}'.format(dfreq) + r' s$^{-1}$ from Line Center'
            else:
                # Gets frequency offset in ANGULAR freq (s^-1)
                dfreq = self.omegabar[ jcenter + np.max(freqoff) ] - self.omegabar[ jcenter + np.min(freqoff) ]
                midfreq = ( self.omegabar[ jcenter + np.max(freqoff) + self.k ] + self.omegabar[ jcenter + np.min(freqoff) + self.k ] ) / 2.
            
                 # If frequency conversion to MHz is requested, converts and generates freq_string label
                if convert_freq:
                    # Converts from angular frequency to frequency & Creates label
                    dfreq = dfreq / (2.*pi)
                    midfreq = midfreq / (2.*pi)
                    if abs(midfreq) < 1e-7:
                        freqtitle = 'at Line Center'
                    elif midfreq < 1000.0:
                        freqtitle = '{0} Hz from Line Center'.format( round(midfreq,0) )
                    elif midfreq >= 1e3 and midfreq < 1e6:
                        freqtitle = '{0} kHz from Line Center'.format( round(midfreq*1e-3,0) )
                    elif midfreq >= 1e6 and midfreq < 1e9:
                        freqtitle = '{0} MHz from Line Center'.format( round(midfreq*1e-6,0) )
                    elif midfreq >= 1e9:
                        freqtitle = '{0} GHz from Line Center'.format( round(midfreq*1e-9,0) )
                    
                    if dfreq < 1000.0:
                        freqtitle += ' ({0} Hz wide)'.format( round(dfreq,0) )
                    elif dfreq >= 1e3 and dfreq < 1e6:
                        freqtitle += ' ({0} kHz wide)'.format( round(dfreq*1e-3,0) )
                    elif dfreq >= 1e6 and dfreq < 1e9:
                        freqtitle += ' ({0} MHz wide)'.format( round(dfreq*1e-6,0) )
                    elif dfreq >= 1e9:
                        freqtitle += ' ({0} GHz wide)'.format( round(dfreq*1e-9,0) )
        
                # If no frequency conversion requested, makes frequency label
                else:
                    freqtitle = r'$\varpi =$ {0:.2e}'.format(midfreq) + r' s$^{-1}$ from Line Center (' + \
                                '{0:.2e}'.format(dfreq) + r' s$^{-1}$ wide)'
                
            # Combines and adds plot title
            if subtitle is None:
                P.title( fig_title + r' at Cloud End, ' + freqtitle + '\n' )
            else:
                P.title( fig_title + r' at Cloud End, ' + freqtitle + '\n' + subtitle )
        

            # Saves figure if requested
            if figname is not None:
                try:
                    P.savefig( figname )
                    if verbose:
                        print('Saved figure to {0}.'.format(figname))
                except:
                    print('Unable to save figure to {0}'.format(figname))
    
            # Finally, shows the plot, if requested
            if show:
                P.show()
            else:
                P.close()
        
        
        
        
        