# Last updated 5/19/2021
# - Made some text smaller
# - plot_all_v_gkk:
#    - Renamed labels option to curve_labels
#    - Replaced figure title with in-figure labelling
#    - Added ability to specify colors and markers per curve, instead of defaults
#    - Added option to turn off gkk curve overplot



# Basic imports
import matplotlib.pyplot as P
import numpy as np
from itertools import cycle, islice
from math import pi

from .comparison import gkk

# Importing and setting up default color maps
try:
    import tol_colors as tol
    cset = tol.tol_cset(colorset='muted')
    color_sets = { 7: [cset.indigo, cset.purple, cset.rose, cset.olive, cset.green, cset.teal, cset.cyan ], 
                   8: [cset.indigo, cset.purple, cset.rose, cset.sand, cset.olive, cset.green, cset.teal, cset.cyan ],
                   9: [cset.indigo, cset.purple, cset.wine, cset.rose, cset.sand, cset.olive, cset.green, cset.teal, cset.cyan ]
                   }
except ImportError:
    color_sets = { 7: ['#332288', '#AA4499', '#CC6677', '#999933', '#117733', '#44AA99', '#88CCEE'],
                   8: ['#332288', '#AA4499', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE'], 
                   9: ['#332288', '#AA4499', '#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE']
                   }

# Creates a dictionary of default marker values
marker_sets = { 7: ['o','s','X','d','^','*','v'],
               8: ['o','s','X','d','^','*','v','P'],
               9: ['o','s','X','d','^','*','v','P','p']
               }


#################### High-level plotting functions for multiple maser_v_theta objects ######################

def plot_mcpeak( maserlist, labels, figname = None, show = True ):
    """
    Plots the *maximum* value of mc along frequency as a function of total optical depth (beta) for a set
    of prism.maser objects (maserlist).
    
    Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
    of beta values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
    
    Required Parameters:
        
        maserlist       List
                            A list of prism.maser class objects. Each object should have the associated
                            stokes arrays calculated (with method cloud_end_stokes) or read in (with method
                            read_cloud_end_stokes) to the appropriate object attributes (eg. stacked_stoki, 
                            etc.).
                    
        labels          List of Strings
                            Ahould be a list with the same length as maserlist, containing the legend label 
                            for each maser class object in maserlist.
        
    Optional Parameters:
                            
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
    
    #### First, checks values and formats some defaults ####
        
    func_name = 'PLOT_MCPEAK'
        
    # Makes sure that, if show is False, a figname has been specified
    if show is False and figname is None:
        err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(func_name) + \
                  " "*(12+len(func_name)+2) + \
                  "figure produced. Please either set show = True or provide a figname for the plot."
        raise ValueError(err_msg)
    
    # If maserlist / labels aren't a list (i.e. is single object), turns into a length-1 list
    if not isinstance( maserlist, list ):
        maserlist = [ maserlist ]
    if isinstance( labels, str ):
        labels = [ labels ]
    
    # Makes template of error message for attribute checks
    attr_missing_msg = func_name + ': Object attribute {0} does not exist for object index {1} in maserlist.'
    attr_shape_msg    = func_name + ': Shape of object attribute {0} for object index {1} in maserlist\n' + \
                    ' '*(12+len(func_name)+2) + 'is not consistent with its attributes betas, omegabar, and k.\n' + \
                    ' '*(12+len(func_name)+2) + 'Attribute {0} should be NumPy array of shape ( {2}, {3} ).'
    
    # Iterates through maser objects in maserlist to check that they have the cloud_end_stokes attributes
    for i,mobj in enumerate(maserlist):
        
        # Iterates through required attributes without shape requirements to make sure the attributes exist
        for req_attr in ['tau_idx', 'betas']:
            if req_attr not in mobj.__dict__.keys():
                raise AttributeError( attr_missing_msg.format(req_attr,i) )
        
        # Sets aside the number of betas and frequencies expected for this object
        Nbetas = mobj.betas.size
        Nfreq  = mobj.omegabar.size - 2 * mobj.k
        
        # Then checks that attributes with shape requirements exist and have the correct shape
        required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                'stacked_ml', 'stacked_evpa' ]
        for req_att in required_attributes:
            if req_att not in mobj.__dict__.keys():
                raise AttributeError( attr_missing_msg.format(req_att, i) )
        
            # If it does exist, makes sure that the array shape is correct
            elif mobj.__dict__[req_att].shape != ( Nbetas, Nfreq ):
                raise ValueError( attr_shape_msg.format(req_att, i, Nbetas, Nfreq) )
    
    # Checks that labels, if provided, is a list with the same length as maserlist
    if not isinstance( labels, list ):
        err_msg = "{0}: Keyword labels must be either a string or a list of strings.".format(func_name) 
        raise ValueError(err_msg)
    elif len( labels ) != len( maserlist ):
        err_msg = "{0}: List provided for labels must be the same length as maserlist.\n".format(func_name)  + \
                  " "*(12+len(func_name)+2) + \
                  "( Current length of labels = {0}, Current length of maserlist = {1} ).".format( len(labels), len(maserlist) )
        raise ValueError(err_msg)
    
    
    
    
    #### Determine the colors and markers ####
    if len(maserlist) <= 7:
        color_list  = color_sets[ 7][:len(maserlist)]
        marker_list = marker_sets[7][:len(maserlist)]
        fill_list = [ 'full', ] * len(maserlist)
    elif len(maserlist) in [8,9]:
        color_list  = color_sets[ len(maserlist)]
        marker_list = marker_sets[len(maserlist)]
        fill_list = [ 'full', ] * len(maserlist)
    else:
        color_list  = list( islice( cycle( color_sets[ 8] ), len(maserlist) ))
        marker_list = list( islice( cycle( marker_sets[9] ), len(maserlist) ))
        fill_template = [ 'full', ]*8
        fill_template.extend( ['none',]*8 )
        fill_list   = list( islice( cycle( fill_template ), len(maserlist) ))
        
        
        
        
    #### Calculates and plots peak mc values ####
    
    # Creates figure
    fig, ax = P.subplots(figsize = (4.5,3.5))
    fig.subplots_adjust( hspace=0, left=0.18,bottom=0.13,right=0.91,top=0.91 )
    
    # Begins iterating through the maser objects
    for i, mobj in enumerate(maserlist):
    
        # Remove edges of array to avoid edge effects; stacked_mc has shape (betas, angfreq)
        stacked_mc = mobj.stacked_mc[:, 2*mobj.k : - 2*mobj.k ]
        
        # Finds max values for each beta along axis 1
        mc_maxes = np.max( stacked_mc, axis=1 )
        
        # Plots curve
        ax.plot( mobj.betas, mc_maxes, marker = marker_list[i], color = color_list[i], fillstyle = fill_list[i], \
                 label = labels[i] )
    
    
    
    
    
    #### Figure axes and labels ####
    
    # Axis limits; makes sure enough vertical room for the legend
    ax.set_xlim(left=0)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim( ymin, ymax * 1.2 )
            
    # Axis labels
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'max $m_c$')
    
    # Make the legend
    ax.legend(loc='upper center', ncol=3, fontsize='x-small' )
    
    
    
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


def plot_all_v_gkk( mvtlist, curve_labels, beta, overplot_gkk = False, label = None, label_loc = 'left', \
    legend_loc = 3, legend_cols = 2, colors = None, markers = None, figname = None, show = True  ):
    """
    Plots m_l and EVPA, both vs. theta, in two windows at a single total optical depth (beta) for a selection
    of maser_v_theta class objects.
    
    Intended to be run *after* stokes at a given point in cloud have been calculated or read in for a range  
    of beta values with cloud_end_stokes or read_cloud_end_stokes method, respectively.
    
    Required Parameters:
        
        mvtlist         List
                            A list of prism.maser_v_theta class objects. Each object should have the 
                            associated stokes arrays calculated (with method cloud_end_stokes) or read in 
                            (with method read_cloud_end_stokes) to the appropriate object attributes (eg. 
                            stacked_stoki, etc.).
                    
        curve_labels    List of Strings
                            Ahould be a list with the same length as mvtlist, containing the legend label 
                            for each maser_v_theta class object in mvtlist.
        
        beta            Float
                            The value of total optical depth to be plotted. Should be in the betas attribute
                            for each maser_v_theta object in mvtlist.
            
    Optional Parameters:
        
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
                            
        legend_loc      Integer
                            [ Default = 3 ]
                            Location of the plot legend in the EVPA subfigure. Default (legend_loc=3) puts 
                            legend in lower left corner. Based on matplotlib legend location options. Other
                            common options are upper left (legend_loc=2), upper right (legend_loc=1), and
                            lower right (legend_loc=4).
                            
        legend_cols     Integer
                            [ Default = 2 ]
                            Number of columns in the legend.
        
        colors          String, List of Strings, or None
                            [ Default = None ]
                            Color or list of colors used to plot each curve. If single string is provided,
                            uses same color for each curve. If list of strings is provided, uses one
                            color provided per curve. If None, uses default list.
                            
        markers         String, List of Strings, or None
                            [ Default = None ]
                            Marker or list of markers used to plot each curve. If single string is provided,
                            uses same marker for each curve. If list of strings is provided, uses one
                            marker provided per curve. If None, uses default list.
                            
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
    
    #### First, checks values and formats some defaults ####
        
    func_name = 'PLOT_ALL_V_GKK'
        
    # Makes sure that, if show is False, a figname has been specified
    if show is False and figname is None:
        err_msg = "{0}: Setting show = False without specifying a file name for the plot will result in no\n".format(func_name) + \
                  " "*(12+len(func_name)+2) + \
                  "figure produced. Please either set show = True or provide a figname for the plot."
        raise ValueError(err_msg)
    
    # If mvtlist / labels aren't a list (i.e. is single object), turns into a length-1 list
    if not isinstance( mvtlist, list ):
        mvtlist = [ mvtlist ]
    if isinstance( curve_labels, str ):
        curve_labels = [ curve_labels ]
    
    # Makes template of error message for attribute checks
    attr_missing_msg = func_name + ': Object attribute {0} does not exist for object index {1} in mvtlist.'
    attr_shape_msg    = func_name + ': Shape of object attribute {0} for object index {1} in mvtlist\n' + \
                    ' '*(12+len(func_name)+2) + 'is not consistent with its attributes betas, omegabar, and k.\n' + \
                    ' '*(12+len(func_name)+2) + 'Attribute {0} should be NumPy array of shape ( {2}, {3} ).'
    
    # Iterates through maser objects in mvtlist to check that they have the cloud_end_stokes attributes
    for i,mobj in enumerate(mvtlist):
        
        # Iterates through required attributes without shape requirements to make sure the attributes exist
        for req_attr in ['tau_idx', 'betas']:
            if req_attr not in mobj.__dict__.keys():
                raise AttributeError( attr_missing_msg.format(req_attr,i) )
        
        # Sets aside the number of betas and frequencies expected for this object
        Nbetas = mobj.betas.size
        Nfreq  = mobj.omegabar.size - 2 * mobj.k
        
        # Then checks that attributes with shape requirements exist and have the correct shape
        for theta in mobj.thetas:
            required_attributes = [ 'stacked_stoki', 'stacked_stokq', 'stacked_stoku', 'stacked_stokv', 'stacked_mc', \
                                    'stacked_ml', 'stacked_evpa' ]
            for req_att in required_attributes:
                if req_att not in mobj.masers[theta].__dict__.keys():
                    raise AttributeError( attr_missing_msg.format(req_att, i) )
        
                # If it does exist, makes sure that the array shape is correct
                elif mobj.masers[theta].__dict__[req_att].shape != ( Nbetas, Nfreq ):
                    raise ValueError( attr_shape_msg.format(req_att, i, Nbetas, Nfreq) )
    
    # Makes sure that all maser_v_theta objects in mvtlist use the same units for thetas
    unitlist = np.unique(np.array([ mobj.units.lower() for mobj in mvtlist ]))
    if np.any([ x in ['degrees','deg','d'] for x in unitlist ]) and np.any([ x in ['radians','rad','r'] for x in unitlist ]):
        err_msg = '{0}: maser_v_theta objects provided in mvtlist do not have consistent units for theta'.format(func_name)
        raise ValueError(err_msg)
    elif np.all([ x in ['degrees','deg','d'] for x in unitlist ]):
        units = 'degrees'
    elif np.all([ x in ['radians','rad','r'] for x in unitlist ]):
        units = 'radians'
    else:
        err_msg = '{0}: Units of maser_v_theta objects not all recognized.\n'.format(func_name) + \
                  " "*(12+len(func_name)+2) + \
                  "Accepted values are 'degrees' (or 'deg' or 'd') or 'radians' (or 'rad' or 'r'). Not case sensitive."
        raise ValueError(err_msg)
        
        
    
    # Checks that curve_labels, if provided, is a list with the same length as mvtlist
    if not isinstance( curve_labels, list ):
        err_msg = "{0}: Keyword curve_labels must be either a string or a list of strings.".format(func_name) 
        raise ValueError(err_msg)
    elif len( curve_labels ) != len( mvtlist ):
        err_msg = "{0}: List provided for curve_labels must be the same length as mvtlist.\n".format(func_name)  + \
                  " "*(12+len(func_name)+2) + \
                  "( Current length of curve_labels = {0}, Current length of mvtlist = {1} ).".format( len(curve_labels), len(mvtlist) )
        raise ValueError(err_msg)
    
        
        
        
    #### Does some processing on requested beta value ####
    
    # Makes sure beta is a float
    beta = float(beta)
        
    # Determines indices for beta for each maser object in maserlist
    beta_idxs = []
    for i, mobj in enumerate(mvtlist):
        if beta in mobj.betas:
            beta_idxs.append( np.where( mobj.betas == beta )[0][0] )
        else:
            err_msg = '{0}: Requested beta value, {1}, not in betas object attribute of maser_v_theta object {2}.\n'.format(func_name, beta, i) + \
                      ' '*(12+len(func_name)+2) + \
                      'Please make sure that cloud_end_stokes attributes have been generated or read for\n' + \
                      ' '*(12+len(func_name)+2) + \
                      'the desired beta values before calling this method.'
            raise ValueError(err_msg)
    
    
    
    
    #### Determine the colors and markers ####
    
    # Processes markers and/or colors provided on call; they will be either a list or None
    if isinstance( markers, str ):
        markers = [ markers, ]*len(mvtlist)
    elif not isinstance( markers, list ):
        markers = None
    if isinstance( colors, str ):
        colors = [ colors, ]*len(mvtlist)
    elif not isinstance( colors, list ):
        colors = None
        
    # Determines color list to use of appropriate length
    # If color list provided, uses only first N values if too long, or iterates if too short
    if colors is not None:
        if len(colors) >= len(mvtlist):
            color_list  = colors[:len(mvtlist)]
        else:
            color_list = list( islice( cycle( colors ), len(mvtlist) ))
    
    # If colors are not provided, sets from defaults based on length
    else:
        if len(mvtlist) <= 7:
            color_list  = color_sets[ 7][:len(mvtlist)]
        elif len(mvtlist) in [8,9]:
            color_list  = color_sets[ len(mvtlist) ]
        else:
            color_list  = list( islice( cycle( color_sets[ 8] ), len(mvtlist) ))
    
    # Determines marker list to use of appropriate length
    # If provided, uses only first N values if too long, or iterates if too short
    if markers is not None:
        if len(markers) >= len(mvtlist):
            marker_list  = markers[:len(mvtlist)]
        else:
            marker_list = list( islice( cycle( markers ), len(mvtlist) ))
    
    # If markers are not provided, sets from defaults based on length
    else:
        if len(mvtlist) <= 7:
            marker_list  = marker_sets[ 7][:len(mvtlist)]
        elif len(mvtlist) in [8,9]:
            marker_list  = marker_sets[ len(mvtlist) ]
        else:
            marker_list  = list( islice( cycle( marker_sets[9] ), len(mvtlist) ))
    
    # Determines fill template
    if len(mvtlist) <= 9:
        fill_list = [ 'full', ] * len(mvtlist)
    else:
        fill_template = [ 'full', ]*8
        fill_template.extend( ['none',]*8 )
        fill_list   = list( islice( cycle( fill_template ), len(mvtlist) ))
        
        
    
        
        
        
        
    #### Calculates and plots peak mc values ####
    
    # Creates figure
    fig, ax = P.subplots(nrows=2, ncols=1, sharex=True, figsize = (5.5,4.5))
    fig.subplots_adjust( hspace=0, left=0.15,bottom=0.13,right=0.95,top=0.91 )
    
    # Begins iterating through the maser_v_theta objects
    for i, mobj in enumerate(mvtlist):
            
        # Gets the index that corresponds to this total optical depth in the stacked arrays
        beta_idx = beta_idxs[i]
        
        # Determines the index of the line center frequency
        jcenter = int((mobj.omegabar.size - 2*mobj.k) / 2)
        
        # Makes the lists of line center ml and evpa values to plot
        plot_ml   = [ mobj.masers[theta].stacked_ml[   beta_idx , jcenter ] for theta in mobj.thetas ]
        plot_evpa = [ mobj.masers[theta].stacked_evpa[ beta_idx , jcenter ] for theta in mobj.thetas ]
        
        # Actually plots with corresponding color/marker/fill
        ax[0].plot( mobj.thetas, plot_ml  , marker = marker_list[i], \
                                            color = color_list[i], fillstyle = fill_list[i] )
        ax[1].plot( mobj.thetas, plot_evpa, marker = marker_list[i], \
                                            color = color_list[i], fillstyle = fill_list[i], label=curve_labels[i] )
    
    # Calculates and plots GKK curve, if requested
    if overplot_gkk:
        theta_min = np.min(np.array([  mobj.thetas.min() for mobj in mvtlist  ]))
        theta_max = np.max(np.array([  mobj.thetas.max() for mobj in mvtlist  ]))
        gkk_thetas = np.linspace( theta_min, theta_max, 1001 )
        gkk_stokqi, gkk_evpa = gkk( gkk_thetas, units = units)
        ax[0].plot( gkk_thetas, np.abs( gkk_stokqi ), 'k:')
        ax[1].plot( gkk_thetas, gkk_evpa, 'k:', label='GKK' )
        
        
        
    
    
    #### Figure axes and labels ####
    
    # X-axis limits and label depend on the units used for theta
    if units in ['degrees','deg','d']:
        P.xlim( 0 , 90 )
        ax[1].set_xlabel(r'$\theta$ [$^{\circ}$]')
    else:
        P.xlim( 0 , 0.5*pi )
        ax[1].set_xlabel(r'$\theta$ [radians]')
        
    # Y-axis limits and label of ml plot
    ax[0].set_ylim(bottom=0)
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
    
    # Make the legend and plot title
    ax[1].legend(loc=legend_loc, fontsize='small', ncol=legend_cols)
    # ax[0].set_title(r'Linear Polarization at Line Center for $\tau = {0}$ vs. GKK'.format( beta ), size = 11)
    
    # Sets plot label, if requested.
    if label is not None:
        if label_loc in ['left','upperleft']:
            ax[0].text( 90.*0.02, ymax - (ymax -ymin )*0.05, label, ha='left', va='top', fontsize='large')
        elif label_loc in ['right','upperright']:
            ax[0].text( 90.*0.98, ymax - (ymax -ymin )*0.05, label, ha='right', va='top', fontsize='large')
        elif label_loc == 'lowerleft':
            ax[1].text( 90.*0.02, ymax1- (ymax1-ymin1)*0.05, label, ha='left', va='top', fontsize='large')
        elif label_loc == 'lowerright':
            ax[1].text( 90.*0.98, ymax1- (ymax1-ymin1)*0.05, label, ha='right', va='top', fontsize='large')
    
    
    
    
    
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
        








################################ Low-level functions for formatting plots ##################################

def _update_label_(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
        
    label = old_label.replace("[{}]".format(units), "")
    exponent_text = exponent_text.replace("\\times", "")
    new_label = "{} [{} {}]".format(label, exponent_text, units)
    return new_label
    
def format_label_string_with_exponent(fig, ax, axis='both'):  
    """
    For an axis given in scientific notation, moves the constant exponential scalar offset to the axis label
    rather than having it in each tick label.
    """
    
    # Retrieves original subplots spacing 
    left, bottom, right, top, wspace, hspace = fig.subplotpars.left, fig.subplotpars.bottom, fig.subplotpars.right, \
                                               fig.subplotpars.top , fig.subplotpars.wspace, fig.subplotpars.hspace    
    
    ax.ticklabel_format(axis=axis, style='sci', scilimits = (0,0))
    
    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
        
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        P.tight_layout() # Has to be run to get offset text to generate
        exponent_text = ax.get_offset_text().get_text()
        #print('           exp_text = {0}'.format(exponent_text))
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        newlabel = _update_label_(label, exponent_text)
        ax.set_label_text(newlabel)
    
    # Reset subplot spacing
    fig.subplots_adjust( left = left, bottom = bottom, right = right, top = top, wspace = wspace, hspace = hspace )