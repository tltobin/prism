# Last updated 4/12/2021
#    -> Renamed base class faraday --> maser.
#    -> Added keyword filename to base class maser (formerly faraday), with root of output file name to which the
#       inversion solutions will be saved. (Previously hard-coded as 'FaradayOut'.)
#    -> Moved physical constants to separate script const.py
#    -> Created new maser base class, _maser_base_, to handle parameter prioritization from call/parameter file/
#       defaults, with option to ignore specific parameters.
#         - Class has method, calc_far_coeff, which is referenced by others, but this class requires that W be handed
#           to it explicitly. Methods of the same name in higher level classes provide W from the object attribute of
#           the same name.
#         - Class also has lower level methods, _read_par_file_ and _process_key_, to read the parameter file using
#           configparser and find the value for a specific key given a config file section from configparser (if any) 
#           and instructions on the allowed data type and value(s) for the parameter.
#    -> Class for single run, maser (formerly faraday) updated to use _maser_base_ for parameter handling in __init__ 
#       and its calc_far_coeff method in the maser method of the same name.
#    -> Created new method, maser_v_theta, to handle sets of maser objects that vary only in theta.
#         - Also uses _maser_base_ class for parameter handling and calc_far_coeff base.
#         - maser_v_theta object has most attributes that a maser object would, except for theta. Instead, has thetas,
#           a NumPy array of the values of theta provided, as well as the associated output paths for each, outpaths.
#         - Allows theta to be specified in degrees or radians, instead of just radians, through the use of the 
#           parameter, units.
#         - The maser base class objects for each parameter set/theta are stored in the object attribute masers, a 
#           dictionary with the values of theta provided on object initialization (rounded to theta_precision) as keys.
#         - The calc_far_coeff, readin, and update_beta methods for the maser_v_theta object update all maser objects
#           in the masers attribute dictionary.




import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.optimize.nonlin import NoConvergence
from math import cos, sin, exp, pi, factorial
import os
import configparser
from collections import OrderedDict

from .utils import _default_, string_to_list, string_to_bool
from .const import e_charge, E0, me, c


# saves original numpy print options 
orig_print = np.get_printoptions()



############################### Primary object class for single parameter set ###############################

class maser(_maser_base_):
    def __init__(self, parfile = None, \
                    omegabar = _default_( None ), tau0 = _default_( None ), theta = _default_( None ), \
                    iquv0 = _default_( None ), W = _default_( None ), k = _default_( None ), \
                    phi = _default_( 0.0 ), n = _default_( 50 ), outpath = _default_( '' ), \
                    far_coeff = _default_( 0.0 ), etapm = _default_( 1.0 ), alphapm = _default_( 1.0 ), \
                    cloud = _default_( 1 ), iquvF = _default_( None ), \
                    betas = _default_( None ), resume = _default_( False ), lastdelta = None, \
                    verbose = _default_( True ), ftol = _default_( 6e-10 ), filename = _default_( 'FaradayOut' ), \
                    endfill = _default_( 'zero' ), trend = _default_( False ), lastdelta2 = None ):
        """
        Object for calculating the dimensionless population inversions for a given parameter set.
        
        Initializing the object establishes object attributes described below. It does not calculate
        the Faraday coefficient from provided terms (see calc_far_coeff method), find the best fit
        population inversions (see run method), or read in output files from previous runs (see readin
        method).
        
        
        Optional Parameters:
            
            parfile         String
                                If provided, gives the path and file name (from current directory) of
                                a parameter file containing values for any of the keywords accepted 
                                by this object class initialization. Values in this parameter file 
                                will override any default values. 
                                Parameter file ingestion also allows the specification of the
                                omegabar array by min, max, and stepsize, as well as the
                                specification of tau0 by number of resolution elements (both of which
                                are not currently supported when set explicitly on object 
                                initialization.)
            
            
                        --- Parameters that must be either provided or read from parameter file ---
                                            --- (see parfile parameter above) ---
            
            omegabar        NumPy Array
                                Array of angular frequencies [s^-1] relative to line center for each 
                                frequency bin. Should be 1D with length NV + 4k, where NV is the 
                                number of angular frequency bins *not* removed by edge effects when
                                calculating population inversions, delta. The 2k angular 
                                frequency bins on either end of the frequency range will be lost
                                during calculation, and will be accounted for by the edge handling
                                method specified by optional keyword endfill. Saved as object 
                                attribute, omegabar. NOTE: NumPy longdouble precision recommended.
            tau0            NumPy Array
                                Array establishing bin locations through the cloud. Values in array
                                should indicate the fraction of the cloud through which the ray has
                                passed, ranging from 0 to 1, inclusive. (If cloud=2, this indicates
                                the fraction of the cloud through which Ray 1 has passed.) Will be 
                                multiplied by the total optical depth (beta) for the cloud for
                                calculation. Saved as object attribute, tau0. NOTE: NumPy longdouble 
                                precision recommended.
            theta           Float
                                The angle between the magnetic field and the line of sight [radians].
                                If cloud=2, this will be taken as the theta for Ray 1, where Ray 2 
                                has theta_2 = - theta. Saved as object attribute, theta.
            iquv0           Length-4 NumPy Array
                                The initial values of (unitless) Stokes i, q, u, and v for the ray
                                prior to passing through the cloud. If cloud=2, these values are
                                used for Ray 1 only. Use optional parameter iquvF to set the 
                                corresponding values for Ray 2 (see below). Saved as object attribute,
                                iquv0.
            W               Float
                                The Doppler Width in angular frequency [s^-1]. Saved as object 
                                attribute, W. 
            k               Integer
                                The number of angular frequency bins in omegabar spanned by the 
                                Zeeman shift, delta omega. Saved as object attribute, k.
             
                               --- Important parameters with useful defaults for all applications ---
                                        
            phi             Float   
                                [ Default = 0.0 ]
                                The sky-plane angle [radians]. If cloud=2, this will be taken as the phi 
                                for Ray 1, where Ray 2 has phi_2 = - phi. Saved as object attribute, phi.
            n               Integer
                                [ Default = 50 ]
                                The number of terms in the LDI expansion. Counting begins at 0. Saves as 
                                object attribute, n.
            outpath         String
                                [ Default = '' ]
                                The directory path (from current directory) to which the output 
                                dimensionless inversions will be saved for each beta value. Saved as 
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
                                      
            betas           Float or NumPy Array
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
                                array tau0 to set the optical depth transversed the cloud at each 
                                spatial resolution point. Saved as object attribute, betas.
                                
                                      --- Parameters Needed when Resuming a Previous Solution ---
                                      
            resume          Boolean
                                [ Default = False ]
                                Whether the solving method, run, will begin with no prior information 
                                about the deltas array (resume=False) or if this is a continuation of a 
                                previous attempt at solving (resume=True). If the former, the initial 
                                guess for the first solution with betas[0] will be an array of ones. If 
                                you wish to continue a previous solving run, set resume to be True and 
                                optional parameter lastdelta (see below) to be the last known array of 
                                deltas, which will be used as the inital guess for the new betas[0] 
                                solution. May also use trend fitting to extrapolate an initial guess 
                                for deltas using lastdelta and lastdelta2. Saved as object attribute, 
                                resume.
            lastdelta       NumPy Array
                                [ Default = None ]
                                The array of deltas to be used as the initial guess (if there is no 
                                trend fitting) or the most recent delta solution (if there is trend 
                                fitting). Only used if resume = True. Array shape should be (T,NV+4k,3), 
                                where NV is the number of angular frequency bins *not* removed by edge 
                                effects and T is the number of tau bins (i.e. the length of tau0). 
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
                                For each total optical depth, beta, the inversion solutions will be
                                saved to a file named '[filename]_beta[beta]_dminus.[fits/txt]' in the 
                                directory specified by outpath.
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
                                effects and T is the number of tau bins (i.e. the length of tau0). The 
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
                                Array of tau0 * beta for a given total optical depth beta. During object
                                initialization, initializes this attribute by assuming beta=1, so tau
                                = tau0.
            dtau            Float
                                Bin width for tau attribute.
        
            
        """
        
        
        #### Uses _maser_base_ initialization to load parfile (if any) and the following attributes:
        ####     phi, n, outpath, far_coeff, etapm, alphapm, cloud, betas, resume, lastdelta, verbose, ftol, filename,
        ####     endfill, trend, lastdelta2
        ####     + fccalc, sintwophi, costwophi
        #### Saves config file as attribute conf, name of parfile as attribute parfile, and name of base section in
        ####     config file as attribute sect
        super().__init__( parfile = parfile, ignore = [], \
                    omegabar = omegabar, tau0 = tau0, theta = theta, iquv0 = iquv0, W = W, k = k, \
                    phi = phi, n = n, outpath = outpath, \
                    far_coeff = far_coeff, etapm = etapm, alphapm = alphapm, \
                    cloud = cloud, iquvF = iquvF, \
                    betas = betas, resume = resume, lastdelta = lastdelta, \
                    verbose = verbose, ftol = ftol, filename = filename, \
                    endfill = endfill, trend = trend, lastdelta2 = lastdelta2)
        
        #### Some extra work on betas attribute, setting beta, tau, and dtau
        
        # If betas is an array, sets beta, tau, and dtau attributes based on first value in array
        if isinstance( self.betas, np.ndarray ):
            self.update_beta( self.betas[0] )
        
        # If betas is None, sets beta, tau, and dtau attributes
        elif self.betas is None:
            self.beta = None
            self.tau  = None
            self.dtau = None
    
    
    ### Main functions used by the typical user ###
    
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
        
        # Begins iteration across beta values for solving
        for b in range( self.betas.size ):
            
            # If this is the first iteration, creates an initial guess array
            if b == 0:
                
                # If this is not resuming a previous run, that guess is an array of ones
                if not self.resume:
                    lastdelta = np.ones(( self.tau0.size, self.omegabar.size, 3 )).astype(np.longdouble)
                
                # Otherwise, sets that initial guess as the input resume array
                else:
                    lastdelta = self.lastdelta.astype(np.longdouble)
                    if self.trend == 'auto' or self.trend == True:
                        lastdelta2 = self.lastdelta2.astype(np.longdouble)
            
            # Sets beta, tau, and dtau attributes for current beta value
            self.update_beta(  self.betas[ b ] )
            
            
            # Prints feedback if requested
            if self.verbose  == True:
                print('PRISM.MASER: Beginning iteration {0} with beta={1}...'.format(b, self.beta ))
            
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
                    
                
                # If two betas haven't been iterated through yet, no trend to fit
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
                # self.write_deltas(self.beta, deltas_new, ext='fits', broken=True)
                break

            
            # Prints feedback if requested
            if self.verbose == True:
                print('PRISM.MASER: Iteration {0} with beta={1} Complete.'.format(b, self.beta))
                print('PRISM.MASER: Output data type', deltas_new.dtype)
                print('PRISM.MASER: Writing output to file...')
            
            # Saves resulting deltas with write_deltas method
            self.write_deltas(self.beta, deltas_new, ext='fits')
            
            # New deltas become old for the next iteration
            if self.trend == True or self.trend == 'auto':
                lastdelta2 = lastdelta.copy()
    
            lastdelta = deltas_new.copy()
            self.deltas = lastdelta
            del deltas_new
        
    def stokes(self, verbose = False ):
        """
        Program that calculates the dimensionless stokes values from the input dimensionless 
        inversion equations, delta.
       
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.tau0 is the array of 
        tau0 values that are the multiplied by beta and integrated over.
       
        self.cloud option can be 1 or 2 depending on whether the seed radiation is entering one (1) 
        or both (2) ends of the cloud. 
        
        self.iquv0 is an array of Stokes I, Q, U, and V values at tau[0]. This should be an array of 
        length 4 of the input stokes values for the light ray. If cloud=2, this will be the initial
        of ray 1.
        
        self.iquvF is the same as iquv0 but for the far end of the cloud corresponding to tau[T-1]. 
        This is only used for cloud=2.
        
        self.k is the number of frequency bins spanned by the Zeeman shift, delta omega.
        
        
        
        Returns an array with shape (T,NV+2k,4) of unitless Stokes values. Zeroth axis separates by
        optical depth, first axis separates by frequency, and second axis separates i, q, u, v.
        
        """
        
        # First separates out the plus, 0, and minus components of delta. These arrays have shape 
        #    (T, NV+4k)
        delta_m = self.deltas[:,:,0]
        delta_0 = self.deltas[:,:,1]
        delta_p = self.deltas[:,:,2]
        
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
    
    def calc_R(self, Gamma = None, verbose = False, sep = False ):
        """
        Program that calculates the stimulated emission rate, R, at the end of the cloud from the 
        input dimensionless inversion equations, delta.
        
        If the loss rate (Gamma, in inverse seconds) is not provided, only calculates the ratio of the
        stimulated emission rate to the loss rate. If the loss rate, Gamma, is provided, will return
        the stimulated emission rate, R, in inverse seconds.
        
        Optional Parameters:
            
            Gamma           Float or None [ default = None ]
                                The loss rate in inverse seconds. If provided, will calculate and 
                                return the stimulated emission rate, R. If set to None, will 
                                calculate and return R/Gamma. 
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
        """
        
        # Figures out beta value from tau and tau0 attributes
        beta = float(self.tau[-1]) / float(self.tau0[-1])
        
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
        domegabar = self.omegabar[1] - self.omegabar[0]
        Rzero  = ( 0.5*Rzero_n[0]  + Rzero_n[1:-1].sum()  + 0.5*Rzero_n[-1]  ) * domegabar
        Rplus  = ( 0.5*Rplus_n[0]  + Rplus_n[1:-1].sum()  + 0.5*Rplus_n[-1]  ) * domegabar
        Rminus = ( 0.5*Rminus_n[0] + Rminus_n[1:-1].sum() + 0.5*Rminus_n[-1] ) * domegabar
        
        # If we're summing the three rates, sums
        if not sep:
            outval = Rminus + Rzero + Rplus
        
        # If returning separate values, sets outval to be tuple
        else:
            outval = ( Rminus, Rzero, Rplus )
        
        # Returns
        return outval
            
    def update_beta( self, beta ):
        """
        Updates beta value (i.e. the total optical depth of the cloud multiplied by tau0).
        
        Updates object attributes self.beta, self.tau, and self.dtau.
        """
        
        # Saves new beta value as object attribute beta
        self.beta = beta
        
        # Scales tau array appropriately
        self.tau = self.tau0 * self.beta
    
        # Determines the spacing in tau
        self.dtau = self.tau[1] - self.tau[0]
            
    
    def readin(self, beta, ext='txt', updatepars = False ): 
        """
        Program to read in files generated by iterative root finding in __init__ function.
        
        Reads in the -, 0, and + delta arrays for the specified beta from the output directory.
        
        Returns array with shape (T,NV+4k,3) array of delta values. Zeroth axis separates by optical
        depth, tau, first axis separates by frequency, and second axis separates by transition for
        -, 0, and +, resp.
        """
        
        # Makes sure . not provided in requested extension
        if ext.startswith('.'):
            ext = ext[1:]
        
        # Reading in if text file
        if ext == 'txt':
        
            # Determines path names for each delta using desired extension
            dminus_path = '{0}{1}_beta{2}_dminus.{3}'.format(self.outpath, self.filename, beta, ext )
            dzero_path  = '{0}{1}_beta{2}_dzero.{3}'.format(self.outpath, self.filename, beta, ext )
            dplus_path  = '{0}{1}_beta{2}_dplus.{3}'.format(self.outpath, self.filename, beta, ext )
        
            # Reads in minus file
            dminus = np.genfromtxt( dminus_path )
        
            # Reads in zero file
            dzero  = np.genfromtxt( dzero_path )
        
            # Reads in plus file 
            dplus  = np.genfromtxt( dplus_path )
        
        # Reading in if fits file
        elif ext == 'fits':
        
            # Determines path names for single fits file
            outpath = '{0}{1}_beta{2}.{3}'.format(self.outpath, self.filename, beta, ext )
            
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
                
                # Reconstructs tau0 assuming tau0 is fraction of cloud transversed from 0 to 1
                self.tau0 = np.linspace( 0, 1, hdr['taubins'] ).astype(np.longdouble)
                
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
                
                # Saves beta and ftol
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
            
            # Closes fits file
            hdu.close()
        
        # Updates beta
        self.update_beta( float(beta) )
        
        return np.dstack(( dminus, dzero, dplus ))
    
    
        
    
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
                outpath_minus = '{0}{1}_beta{2}_dminus.{3}'.format(self.outpath, self.filename, self.beta, ext )
                outpath_zero  = '{0}{1}_beta{2}_dzero.{3}'.format(self.outpath, self.filename, self.beta, ext )
                outpath_plus  = '{0}{1}_beta{2}_dplus.{3}'.format(self.outpath, self.filename, self.beta, ext )
            else:
                outpath_minus = '{0}{1}_beta{2}_dminus_BROKEN.{3}'.format(self.outpath, self.filename, self.beta, ext )
                outpath_zero  = '{0}{1}_beta{2}_dzero_BROKEN.{3}'.format(self.outpath, self.filename, self.beta, ext )
                outpath_plus  = '{0}{1}_beta{2}_dplus_BROKEN.{3}'.format(self.outpath, self.filename, self.beta, ext )
            
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
                outpath = '{0}{1}_beta{2}.{3}'.format(self.outpath, self.filename, self.beta, ext )
            else:
                outpath = '{0}{1}_beta{2}_BROKEN.{3}'.format(self.outpath, self.filename, self.beta, ext )
            
            # Makes primary HDU with no data
            prime_hdu = fits.PrimaryHDU()
            
            # Populates primary header with sim info
            prime_hdu.header['cloud'] = ( self.cloud, 'number of rays' )
            prime_hdu.header['Doppler'] = ( self.W, 'Doppler width [s^-1]' )
            prime_hdu.header['Zeeman'] = ( (self.omegabar[1]-self.omegabar[0])*float(self.k), 'Zeeman splitting [s^-1]' )
            prime_hdu.header['k'] = ( self.k, 'Zeeman splitting [bins]' )
            prime_hdu.header['AFres'] = ( self.omegabar[1]-self.omegabar[0], 'Angular Freq Resolution [s^-1]' )
            prime_hdu.header['AFbins'] = ( self.omegabar.size, 'Angular Freq Resolution Bins' )
            prime_hdu.header['taubins'] = ( self.tau0.size, 'Number of Tau Resolution Bins' )
            prime_hdu.header['tau'] = ( self.tau[-1] / self.tau0[-1], 'Total Optical Depth' )
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
        lin = '  Tau resolution elements = {0}\n'.format( self.tau0.size )
        desc.write(lin)
        
        # Line for maximum beta
        lin = '  Beta = {0}\n'.format( self.betas )
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
        
        # Finally, prints full tau0 and omegabar arrays
        lin = 'Full Tau0 Array (size={0}):\n  {1}\n\n'.format(self.tau0.size, self.tau0 )
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
            for t in range( self.tau0.shape[0] ):
        
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
    
    def gain_matrix(self, delta, beta = None ):
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
        
        # If beta is provided along with delta array, sets
        if beta is not None:
            self.update_beta( beta )
        
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
    
    def integ_gain(self, delta, beta = None):
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
        # If beta is provided along with delta array, sets
        if beta is not None:
            self.update_beta( beta )
            
            
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
    
    def stokes_per_ray(self, delta, beta = None, verbose=False ):
        """
        Program that calculates the dimensionless stokes values from the input dimensionless 
        inversion equations, delta.
       
        The main input for this function, delta, should be a numpy array with dimensions 
        (T,NV+4k,3), where NV is the number of velocity bins and T is the number of tau bins. The 
        0th axis specifies values across different tau at constant frequency for a single 
        transition. The 1st axis specifies values across frequency at constant tau for a transition.
        The three rows along the 0th axis should be for delta^- (delta[0]), delta^0 (delta[1]), and 
        delta^+ (delta[2]).
        
        beta is a float that is multiplied by self.tau0 to determine the tau values that are
        integrated over
        
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.tau0 is the array of 
        tau0 values that are the multiplied by beta and integrated over.
       
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
        
        # If beta is provided along with delta array, sets
        if beta is not None:
            self.update_beta( beta )
    
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
    
    def LDI_terms(self, delta, beta = None, verbose=False ):
        """
        Program that calculates the dimensionless stokes values from the input dimensionless 
        inversion equations, delta.
       
        The main input for this function, delta, should be a numpy array with dimensions 
        (T,NV+4k,3), where NV is the number of velocity bins and T is the number of tau bins. The 
        0th axis specifies values across different tau at constant frequency for a single 
        transition. The 1st axis specifies values across frequency at constant tau for a transition.
        The three rows along the 0th axis should be for delta^- (delta[0]), delta^0 (delta[1]), and 
        delta^+ (delta[2]).
        
        beta is a float that is multiplied by self.tau0 to determine the tau values that are
        integrated over
        
        verbose [default=False]: Whether to print out the values in a specified wavelength bin at
        several points in the calculation for checking. To print, set verbose to the index of the 
        wavelength bin (in the original delta array) desired for printout.
        
        Values defined in __init__ used here: self.theta is the angle between the magnetic field and 
        line of sight. self.phi is the sky-plane angle. Their corresponding values, self.costheta, 
        self.sintheta, self.costwophi and self.sinthophi are also used. self.etap and self.etam are 
        eta^+ and eta^-, respectively, or the squared ratio of the + and - dipole matrix components 
        to that of the 0th. self.far_coeff is -gamma_QU / cos(theta). and self.tau0 is the array of 
        tau0 values that are the multiplied by beta and integrated over.
       
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
        
        # If beta is provided along with delta array, sets
        if beta is not None:
            self.update_beta( beta )
    
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
                                specification of tau0 by number of resolution elements (both of which
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
        ####     phi, n, outpath, far_coeff, etapm, alphapm, cloud, betas, resume, lastdelta, verbose, ftol, filename,
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
        
        
        #### Some extra work on betas attribute, setting beta, tau, and dtau
        
        # If betas is an array, sets beta, tau, and dtau attributes based on first value in array
        if isinstance( self.betas, np.ndarray ):
            self.update_beta( self.betas[0] )
        
        # If betas is None, sets beta, tau, and dtau attributes
        elif self.betas is None:
            self.update_beta( None )
        
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
    
    def readin(self, beta, as_attr, ext='txt', updatepars = False ): 
        """
        Program to read in files generated by iterative root finding for each maser object in the masers 
        dictionary attribute and saves them as an object attribute of the individual maser objects of 
        name as_attr. 
        
        For example, to load the beta = 2.0 delta arrays (saved as fits files) as the lastdelta
        attributes for each object, run
            
            self.readin( 2.0, 'lastdelta', ext='fits', updatepars = False )
        
        Required Parameters:
            
            beta            Float
                                
                                Value or an array of total optical depths for the cloud. Unitless.
                                Indicates which solution should be read in from the output path.
            
            as_attr         String
                                
                                The name of the attribute of each maser object in the masers dictionary
                                to which the corresponding deltas array will be saved.
            
        Optional Parameters:
            
            ext             String ('txt' or 'fits')
                                
                                [ Default = 'txt' ]
                                
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
                dminus_path = '{0}{1}_beta{2}_dminus.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, beta, ext )
                dzero_path  = '{0}{1}_beta{2}_dzero.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, beta, ext )
                dplus_path  = '{0}{1}_beta{2}_dplus.{3}'.format(self.masers[theta].outpath, self.masers[theta].filename, beta, ext )
        
                # Reads in files
                dminus = np.genfromtxt( dminus_path )
                dzero  = np.genfromtxt( dzero_path )
                dplus  = np.genfromtxt( dplus_path )
            
        
            # Reading in if fits file
            elif ext == 'fits':
        
                # Determines path names for single fits file
                outpath = '{0}{1}_beta{2}.{3}'.format(self.masers[theta].outpath, self.filename, beta, ext )
            
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
                
                    # Reconstructs tau0 assuming tau0 is fraction of cloud transversed from 0 to 1
                    self.masers[theta].tau0 = np.linspace( 0, 1, hdr['taubins'] ).astype(np.longdouble)
                
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
                
                    # Saves beta and ftol
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
            
            
            # Updates beta
            self.update_beta( float(beta) )
        
            # Sets as requested attribute
            self.masers[theta].__dict__[as_attr] = np.dstack(( dminus, dzero, dplus ))
    
    def update_beta( self, beta ):
        """
        Updates beta value (i.e. the total optical depth of the cloud multiplied by tau0).
        
        Updates object attributes self.beta, self.tau, and self.dtau.
        """
        
        if beta is not None:
        
            # Saves new beta value as object attribute beta
            self.beta = beta
        
            # Scales tau array appropriately
            self.tau = self.tau0 * self.beta
    
            # Determines the spacing in tau
            self.dtau = self.tau[1] - self.tau[0]
        
        else:
            self.beta = None
            self.tau  = None
            self.dtau = None
        
        # Sets for all lower level maser objects
        for theta in self.thetas:
            for key in ['beta','tau','dtau']:
                self.masers[ theta ].__dict__[key] = self.__dict__[key]
    

########################### Utility base class for par setting and faraday calc ############################

class _maser_base_:
    def __init__( self, parfile = None, ignore = [], \
                    omegabar = _default_( None ), tau0 = _default_( None ), theta = _default_( None ), \
                    iquv0 = _default_( None ), W = _default_( None ), k = _default_( None ), \
                    phi = _default_( 0.0 ), n = _default_( 50 ), outpath = _default_( '' ), \
                    far_coeff = _default_( 0.0 ), etapm = _default_( 1.0 ), alphapm = _default_( 1.0 ), \
                    cloud = _default_( 1 ), iquvF = _default_( None ), \
                    betas = _default_( None ), resume = _default_( False ), lastdelta = None, \
                    verbose = _default_( True ), ftol = _default_( 6e-10 ), filename = _default_( 'FaradayOut' ), \
                    endfill = _default_( 'zero' ), trend = _default_( False ), lastdelta2 = None):
        """
        Base class for handling parameters of maser and maser_v_theta. 
        
        Does not have 'beta', 'tau', 'dtau' attributes or the associated method to set them, self.update_beta.
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
                else:
                    print(omegabar_min, omegabar_max, d_omegabar)
                    raise ValueError( 'Keyword omegabar must be provided either on call or in parameter file either directly or\n' + \
                                        '    through the combination of (omegabar_min, omegabar_max, d_omegabar).' )
        
        
        ## Parameter tau0 -- NumPy array, required, can also be set by taures
        if 'tau0' not in ignore:
            self.tau0 = self._process_key_( 'tau0', tau0, self.conf[sect], \
                                allowed = { np.ndarray: [np.longdouble, None] }, convert = False , ignore_none = True  )
            
            # If tau0 is not set, checks for taures; int, conv allowed
            if self.tau0 is None:
                taures = self._process_key_( 'taures', _default_(None), self.conf[sect], \
                                                       allowed = { int: None }, convert = True , ignore_none = True  )
                if taures is not None:
                    self.tau0 = np.linspace( 0, 1, taures, dtype = np.longdouble )
                else:
                    raise ValueError( 'Keyword tau0 must be provided either on call or in parameter file either directly or\n' + \
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
        
        
        ## Parameter betas -- list/array, float, or nonetype, conversion allowed, can be specified in parfile, 
        #    none is valid 
        #  print('Starting betas...')  # Line for debugging
        if 'betas' not in ignore:
            self.betas = self._process_key_( 'betas', betas, self.conf[sect], \
                         allowed = OrderedDict([ (np.ndarray, [float,None]), (float, None), (None, None) ]), \
                         convert = True , ignore_none = False )
            
            # If betas is single value, converts it to a length-1 numpy array 
            if isinstance( self.betas, float ):
                self.betas = np.array([ self.betas ])
            
        
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
        Also uses the Doppler width in Hz given in object initialization.
        
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
        Utility function to process key values by priority: specified on call --> conf file (if provided) --> default.
        
        Can filter for allowed data types and values using allowed dictionary (optional keyword).
        
        Required parameters:
            
            keyname         String
                                The name of the parameter as it appears in the config file (if provided). Also used in
                                error messages.
                                
            keyvalue        (Any)
                                Value of the parameter set on function call. Method will distinguish if it is the
                                default parameter class or user-specified.
                                
            confsection     ConfigParser SectionProxy object, or None
                                To refer to a config file when the key is not specified directly on call, supply the
                                section of the relevent config file, as read in by configparser, here, eg:
                                
                                    >>> conf = configparser.ConfigParser(inline_comment_prefixes=['#']) 
                                    >>> conf.read( 'my_config_file.par')
                                    >>> self._process_key_( keyname, keyvalue, conf['Relevent Section Name'] )
                                    
                                If no config file is provided or desired for cross-referencing, enter None.
        
        Optional parameters:
            
            allowed         Dict or OrderedDict
                                [ Default = {} ]
                                Used to check that values provided by user and read from the parameter file (if 
                                applicable) are the appropriate data types/values for the parameter. Dictionary format
                                should have the allowed data types as dictionary keys. If only specific values for a
                                given data type are allowed, then the dictionary entry for that data type will point to
                                a list or tuple of the allowed values associated with that data type. If any values of
                                a given data type are allowed, the dictionary entry for that data type will be None.
                                
                                For example, to check a keyword that can be either a boolean or a string named 'auto':
                                
                                    >>> allowed = { bool: None, str: [ 'auto' ] }
                                
                                If there is an order in which the data types should be checked, use an OrderedDict 
                                instead of a Dict object for allowed:
                                    
                                    >>> # If we need to check if a string is None, but it can also be a string
                                    >>> from collections import OrderedDict
                                    >>> allowed = OrderedDict()
                                    >>> allowed[None] = None
                                    >>> allowed[str]  = None
                                
                                If the data type is a list or NumPy array, instead of specific values allowed for the
                                keyword, the allowed dictionary should point to a length-2 list with the allowed 
                                data types and length of the list. Either can be specified as None to remove 
                                constraints. Multiple data types can be specified with nesting, which will be taken in 
                                priority order:
                                    
                                    >>> # To require the value to be a length-4 list of floats
                                    >>> allowed = { list: [ float, 4 ] }
                                    >>>
                                    >>> # To require the value to be list of float or string of any length, with
                                    >>> #    floats preferred over strings
                                    >>> allowed = { list: [ ( float, str ), None ] }
                                
                                Note: If data type is specified for a list/array, values will be converted, if possible.
                                
                                List and NumPy array are considered interchangable for checking data type, but whichever 
                                is specified in the allowed dictionary is what the item will be returned as.
                                    
                                    >>> # The above examples will return a list object
                                    >>> # To repeat the len-4 list of floats returned as a numpy array
                                    >>> allowed = { np.ndarray: [ float, 4 ] }
                                    
                                Note 1: If no data type is specified for a list/array type object and data is read in 
                                        from the config file, the data type of the values in the list will be a string.
                                Note 2: The data type of a numpy array is numpy.ndarray, NOT numpy.array.
            
            ignore_none     Boolean 
                                [ Default = True ]
                                Whether to treat any parameters set as None in the config file as being unset (True) or
                                treat None as a viable parameter for the key (False). Only used if a confsection is 
                                provided.
            
            convert         Boolean
                                [ Default = False ]
                                Whether to try to convert any user-provided values into the data types, or simply check
                                if they have the correct data type. This only applies to values provided on object call,
                                not any read from the config file or built in defaults. Note: If convert is turned on,
                                and multiple data types are acceptable, you MUST use an OrderedDict for your allowed
                                values to ensure consistent type conversion.
            
        Returns:
            
           out_value        The value of the key, as determined by prioritizing on call -> conf file -> default.
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
        