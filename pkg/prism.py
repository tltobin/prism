# Last updated 12/27/2021 
#    - Added calc_R method to faraday class object to calculate the stimulated emission rate, R, for a given solution.
#    - Removed requirement to provide delta array to stokes method on call; now just uses attribute, self.deltas.
#    - Removed requirement to provide beta value to stokes method on call; now calculates by comparing tau attribute
#          to tau0 attribute.


import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.optimize.nonlin import NoConvergence
from math import cos, sin, exp, pi, factorial
import os
from glob import glob

# saves original numpy print options 
orig_print = np.get_printoptions()


class faraday:
    def __init__(self, omegabar, tau0, theta, iquv0, W, k, \
                    phi = 0.0, far_coeff = 0.0, etapm = 1.0, alphapm = 1.0, n = 50, \
                    outpath = '', endfill = 'zero', verbose=True, \
                    cloud = 1, iquvF = None, \
                    beta = None, ftol = 6e-10, \
                    resume=False, lastdelta=None, trend = False, lastdelta2=None ):
        """
        Object for calculating the dimensionless population inversions for a given parameter set.
        
        Initializing the object establishes object attributes described below. It does not calculate
        the Faraday coefficient from provided terms (see calc_far_coeff method), find the best fit
        population inversions (see run method), or read in output files from previous runs (see readin
        method).
        
        Required Parameters:
            
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
        
        Optional Parameters:
            
            
                                        --- Important parameters with useful defaults ---
                                        
            phi             Float   
                                [ Default = 0.0 ] The sky-plane angle [radians]. If cloud=2, this will 
                                be taken as the phi for Ray 1, where Ray 2 has phi_2 = - phi. Saved as
                                object attribute, phi.
            far_coeff       Float
                                [ Default = 0.0 ] Unitless value gives -gamma_QU / cos(theta). Can 
                                either be specified explicitly here or calculated from components 
                                later by calc_far_coeff method. Saved as object attribute, far_coeff.
            etapm           Float or Length-2 List or NumPy Array
                                Contains values [ eta^+ , eta^- ], where each corresponds to the
                                relative line strengths between the +/- transition (respectively) and 
                                the 0 transition. I.e. eta^+ = | dhat^+ |^2 / | dhat^0 |^2 and 
                                eta^- = | dhat^- |^2 / | dhat^0 |^2, where the dhat terms are the 
                                dipole moments of each transition. Unitless. If specified as a float 
                                instead of a len-2 list/array, will use the value specified as both 
                                eta^+ and eta^-. Saves eta^+ as object attribute etap and eta^- as 
                                object attribute etam. 
            alphapm         Float or Length-2 List or NumPy Array
                                Contains values [ alpha^+ , alpha^- ], where each corresponds to the
                                ratio between the pump rate of the +/- state, respectively, and that
                                of the 0 state. I.e. alpha^+ = P^+ / P^0 and alpha^- = P^- / P^0. 
                                Unitless. If specified as a float instead of a len-2 list/array, will 
                                use the value specified as both alpha^+ and alpha^-. Saves alpha^+ as 
                                object attribute alphap and alpha^- as object attribute alpha^-. 
            n               Integer
                                [ Default = 50 ] The number of terms in the LDI expansion. Counting 
                                begins at 0. Saves as object attribute, n.
            outpath         String
                                [ Default = '' ] The directory path (from current directory) to which
                                the output dimensionless inversions will be saved for each beta value.
                                Saved as object attribute, outpath.
            endfill         'fit' or 'zero'
                                [ Default = 'zero' ] Determines how 2k angular frequency bins on either 
                                end will be filled in the inversion calculation. If endfill='fit', a 4th 
                                order polynomial as a function of frequency will be fit to the output 
                                delta arrays at each optical depth, and the end values will be filled in 
                                according to their frequency. If endfill='zero', the end points will just 
                                be set to zero. Saved as object attribute, endfill.
            verbose         Boolean
                                [ Default = True ] Whether to print feedback to terminal at various 
                                stages of the process. Saved as object attribute, verbose.
            
                                    --- Parameters for unidirectional/bidirectional solutions ---
                                    
            cloud           1 or 2  
                                [ Default = 1 ] Switch indicating whether the solution should be 
                                calculated for a uni-directional cloud (cloud=1) or a bi-directional 
                                cloud (cloud=2). If cloud = 2, requires optional parameter iquvF to be 
                                set. Saved as object attribute, cloud.
            iquvF           Length-4 NumPy Array    
                                [ Default = None ] The initial values of (unitless) Stokes i, q, u, and 
                                v for Ray 2 prior to passing through the cloud. Only used if cloud=2. 
                                Saved as object attribute, iquvF, if cloud=2. 
                            
                                      --- Parameters Needed when Starting a New Solution ---
                                      
            beta            Float or NumPy Array
                                [ Default = None ] Value or an array of total optical depths for the 
                                cloud. Unitless.
                                IF INITIALIZING A NEW CALCULATION, SET THIS TO NOT BE NONE. If
                                reading in prior calculation, setting this is not required. 
                                Length should be N, where N is the total number of solutions that will 
                                be calculated. Values should be specified in increasing order, as 
                                the solution for each optical depth will be used as the initial guess 
                                for the next, and computation time increases with increasing optical
                                depth. If no prior results are available, recommend starting with the
                                first value in this array < 1.0. Each value will be multiplied by
                                array tau0 to set the optical depth transversed the cloud at each 
                                spatial resolution point. Saved as object attribute, beta.
            ftol            Float
                                [ Default = 6e-10 ] The tolerance in minimizing the delta residuals 
                                used to determine convergence on a new solution. Passed directly to
                                scipy.optimize.newton_krylov as its parameter, f_tol. Saved as object 
                                attribute, ftol.
                                
                                      --- Parameters Needed when Resuming a Previous Solution ---
                                      
            resume          Boolean
                                [ Default = False ] Whether the solving method, run, will begin with no 
                                prior information about the deltas array (resume=False) or if this is a 
                                continuation of a previous attempt at solving (resume=True). If the 
                                former, the initial guess for the first solution with beta[0] will be an 
                                array of ones. If you wish to continue a previous solving run, set resume 
                                to be True and optional parameter lastdelta (see below) to be the last 
                                known array of deltas, which will be used as the inital guess for the new 
                                beta[0] solution. May also use trend fitting to extrapolate an initial 
                                guess for deltas using lastdelta and lastdelta2. Saved as object 
                                attribute, resume.
            lastdelta       NumPy Array
                                [ Default = None ] The array of deltas to be used as the initial guess (if
                                there is no trend fitting) or the most recent delta solution (if there is 
                                trend fitting). Only used if resume = True. Array shape should be 
                                (T,NV+4k,3), where NV is the number of angular frequency bins *not* 
                                removed by edge effects and T is the number of tau bins (i.e. the length 
                                of tau0). The three rows along the 2nd axis divide the population
                                inversions by transition, with lastdelta[:,:,0] = delta^-, 
                                lastdelta[:,:,1] = delta^0, and lastdelta[:,:,2] = delta^+. Saved as
                                object attribute, lastdelta.
            trend           Boolean or 'auto'
                                [ Default = False ] Keyword indicating whether to use one previous fitted 
                                delta (lastdelta) as the initial guess for the next iteration (if trend = 
                                False), or use the two fitted previous deltas (lastdelta and lastdelta2) 
                                to generate an initial guess for the next iteration (if trend = True). For 
                                the latter, it uses lastdelta + ( lastdelta2 - lastdelta )/2 as the initial 
                                guess. Finally, if trend = 'auto', calculates the residual using both 
                                lastdelta and lastdelta + ( lastdelta2 - lastdelta )/2 before starting the 
                                zero-finding using the one with the lower residual as the initial guess.
                                Saved as object attribute, trend.
            lastdelta2      NumPy Array
                                [ Default = None ] The second most recent delta solution. Only used if 
                                resume = True AND trend = 'auto' or True. See optional parameter trend
                                above for details on usage. If specified, array shape should be (T,NV+4k,3), 
                                where NV is the number of angular frequency bins *not* removed by edge 
                                effects and T is the number of tau bins (i.e. the length of tau0). The three 
                                rows along the 2nd axis divide the population inversions by transition, with 
                                lastdelta2[:,:,0] = delta^-,  lastdelta2[:,:,1] = delta^0, and 
                                lastdelta2[:,:,2] = delta^+. Saved as object attribute, lastdelta2.
                                
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
        
        #### Saves required parameters as object attributes ####
        self.omegabar   = omegabar
        self.tau0       = tau0
        self.theta      = theta
        self.iquv0      = iquv0
        self.W          = W
        self.k          = k
        
        
        #### Saving required parameters that have useful defaults ####
        
        # Some easy ones here for phi and far_coeff
        self.phi        = phi 
        self.far_coeff  = far_coeff
        
        # etapm can be set as float or len-2; uses to set etap and etam
        if isinstance( etapm, float ) or isinstance( etapm, int ):
            self.etap = float( etapm )
            self.etam = float( etapm )
        else:
            self.etap, self.etam = etapm
            
        # alphapm can be set as float or len-2; uses to set alphap and alpham
        if isinstance( alphapm, float ) or isinstance( alphapm, int ):
            self.alphap = float( alphapm )
            self.alpham = float( alphapm )
        else:
            self.alphap, self.alpham = alphapm
        
        # n is saved directly
        self.n         = n
        
        # Makes sure outpath ends with a /, if specified as non-empty string, before saving as attribute
        if len(outpath) > 0 and not outpath.endswith('/'):
            outpath += '/'
        self.outpath    = outpath
        
        # endfill and verbose set directly
        self.endfill = endfill
        self.verbose = verbose
        
        
        #### Saving pars for uni/bi-directional solutions ####
        # iquvF only saved if cloud is 2
        self.cloud      = cloud
        if self.cloud == 2:
            self.iquvF  = iquvF
        
        
        #### Saving pars for starting new solutions ####
        if isinstance( beta, int ) or isinstance( beta, float ):
            self.beta = np.array([ beta ])
        else:
            self.beta    = beta
        self.ftol    = ftol
        
        
        #### Saving pars for resuming previous solutions ####
        self.resume  = resume
        self.lastdelta = lastdelta
        self.trend   = trend
        self.lastdelta2 = lastdelta2
           
        
            
        #### Sets some additional attributes ####
        
        # Sets boolean saying that far_coeff has not (yet) been calculated by calc_far_coeff
        self.fccalc = False
        
        # Begins by preemptively calculating sin and cos theta and sin and cos of 2 phi
        self.sintheta = sin(self.theta)
        self.costheta = cos(self.theta)
        self.sintwophi = sin(2.*self.phi)
        self.costwophi = cos(2.*self.phi)
        
        # Sets initial tau array
        self.tau = self.tau0
        
        # Determines the spacing in tau
        self.dtau = self.tau[1] - self.tau[0]
        
    
    
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
        
        # First checks if mode is cm
        if mode == 'cm':
            
            # If in cm mode, converts to SI
            ne = ne * 10.**6
            P0 = P0 * 10.**6
        
        # Sets values for necessary constants
        e_charge = 1.6021766208 * 10.**-19    # Elementary charge in C
        E0 = 8.854187817620 * 10.**-12        # Vacuum permittivity in F/m
        me = 9.10938356 * 10.**-31            # Electron mass in kg
        c = 2.99792458 * 10.**8               # Speed of light in m/s
        
        # Calculates small w from big W 
        w = self.W * c / ( 2.*pi*freq0 )     # Width parameter in velocity space in m/s
        
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
        print('FARADAY:  Output Path', path )
        
        # Writes sim description file if requested
        if sim_desc:
            self.write_desc(path)
        
        # If checkfile should be written out, begins array of output gamma values to print
        check_out = np.array([])
        
        # Begins iteration across beta values for solving
        for b in range( self.beta.size ):
            
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
            
            # Determines appropriate multiplier
            bval = self.beta[ b ]
            
            # Sets tau array
            self.tau = self.tau0 * bval
        
            # Determines the spacing in tau
            self.dtau = self.tau[1] - self.tau[0]
            
            # Prints feedback if requested
            if self.verbose  == True:
                print('FARADAY: Beginning iteration {0} with beta={1}...'.format(b, bval))
            
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
                #print('FARADAY: Input delta data type ',end=" ")
                #print(self.deltas.dtype)
                deltas_new = optimize.newton_krylov( self.inversion_resid, initdelta, maxiter=maxiter, \
                                                          f_tol = self.ftol, verbose=self.verbose )
            
            # If solver converged on a non-solution, prints an error and shunts last delta to a
            #     text file before breaking
            except Exception as e:
                print('ERROR: Solution not found in {0} iterations or otherwise broken.'.format( maxiter ))
                print(e)
                
                # Saves resulting deltas with write_deltas method and breaks
                # self.write_deltas(bval, deltas_new, ext='fits', broken=True)
                break

            
            # Prints feedback if requested
            if self.verbose == True:
                print('FARADAY: Iteration {0} with beta={1} Complete.'.format(b, bval))
                print('FARADAY: Output data type', deltas_new.dtype)
                print('FARADAY: Writing output to file...')
            
            # Saves resulting deltas with write_deltas method
            self.write_deltas(bval, deltas_new, ext='fits')
            
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
        
        # Figures out beta value from tau and tau0 attributes
        beta = float(self.tau[-1]) / float(self.tau0[-1])
        
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
        
        # Determines tau values and dtau for this calculation
        tau = beta * self.tau0
        dtau = tau[1] - tau[0]
        
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
                   - self.stokq[ -1, -1*k : k ] * self.costwophi - self.stoku[ -1, k : -1*k ] * self.sintwophi )
        Rplus_n  = Gamma * self.etap * ( cos2thetap1 * self.stoki[ -1, : -2*k ] - 2. * self.stokv[ -1, : -2*k ] * self.costheta \
                   + ( self.costwophi * self.stokq[ -1, : -2*k ] + self.sintwophi * self.stoku[ -1, : -2*k ] ) * self.sintheta * self.sintheta )
        Rminus_n = Gamma * self.etam * ( cos2thetap1 * self.stoki[ -1, 2*k : ]  + 2. * self.stokv[ -1, 2*k : ] * self.costheta \
                   + ( self.costwophi * self.stokq[ -1, 2*k :  ] + self.sintwophi * self.stoku[ -1, 2*k :  ] ) * self.sintheta * self.sintheta )
        
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
            

    
    def readin(self, beta, ext='txt', updatepars = False): 
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
            dminus_path = '{0}FaradayOut_beta{1}_dminus.{2}'.format(self.outpath, beta, ext )
            dzero_path  = '{0}FaradayOut_beta{1}_dzero.{2}'.format(self.outpath, beta, ext )
            dplus_path  = '{0}FaradayOut_beta{1}_dplus.{2}'.format(self.outpath, beta, ext )
        
            # Reads in minus file
            dminus = np.genfromtxt( dminus_path )
        
            # Reads in zero file
            dzero  = np.genfromtxt( dzero_path )
        
            # Reads in plus file 
            dplus  = np.genfromtxt( dplus_path )
        
        # Reading in if fits file
        elif ext == 'fits':
        
            # Determines path names for single fits file
            outpath = '{0}FaradayOut_beta{1}.{2}'.format(self.outpath, beta, ext )
            
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
        
        # Sets tau array
        self.tau = self.tau0 * float(beta)
        
        # Determines the spacing in tau
        self.dtau = self.tau[1] - self.tau[0]
        
        return np.dstack(( dminus, dzero, dplus ))
    
    
        
    
    ### Functions for writing output ###

    def write_deltas(self, beta, deltas, ext='txt', broken=False):
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
                outpath_minus = '{0}FaradayOut_beta{1}_dminus.{2}'.format(self.outpath, beta, ext )
                outpath_zero  = '{0}FaradayOut_beta{1}_dzero.{2}'.format(self.outpath, beta, ext )
                outpath_plus  = '{0}FaradayOut_beta{1}_dplus.{2}'.format(self.outpath, beta, ext )
            else:
                outpath_minus = '{0}FaradayOut_beta{1}_dminus_BROKEN.{2}'.format(self.outpath, beta, ext )
                outpath_zero  = '{0}FaradayOut_beta{1}_dzero_BROKEN.{2}'.format(self.outpath, beta, ext )
                outpath_plus  = '{0}FaradayOut_beta{1}_dplus_BROKEN.{2}'.format(self.outpath, beta, ext )
            
            # Writes output to text file with numpy savetxt
            np.savetxt(outpath_minus,deltas[:,:,0],fmt='%.18f')
            np.savetxt(outpath_zero, deltas[:,:,1],fmt='%.18f')
            np.savetxt(outpath_plus, deltas[:,:,2],fmt='%.18f')
            
            # Prints feedback if requested
            if self.verbose == True:
                print('Done.')
                print('FARADAY: Output files written:')
                print('FARADAY:     {0}'.format(outpath_minus))
                print('FARADAY:     {0}'.format(outpath_zero))
                print('FARADAY:     {0}'.format(outpath_plus))
        
        # Writing fits file if requested
        elif ext == 'fits':
            
            # Creates single path for output for delta minus, 0, and plus
            if not broken:
                outpath = '{0}FaradayOut_beta{1}.{2}'.format(self.outpath, beta, ext )
            else:
                outpath = '{0}FaradayOut_beta{1}_BROKEN.{2}'.format(self.outpath, beta, ext )
            
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
                print('FARADAY: Output file {0} written.'.format( outpath ) )
    
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
        lin = '  Beta = {0}\n'.format( self.beta )
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
    
    def gain_matrix(self, delta ):
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
    
    def integ_gain(self, delta):
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
    
    def stokes_per_ray(self, delta, beta, verbose=False ):
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
        
        # Determines tau values and dtau for this calculation
        tau = beta * self.tau0
        dtau = tau[1] - tau[0]
        
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
    
    def LDI_terms(self, delta, beta, verbose=False ):
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
        
        # Determines tau values and dtau for this calculation
        tau = beta * self.tau0
        dtau = tau[1] - tau[0]
        
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
    
    
    
