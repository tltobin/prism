
# Basic imports
import matplotlib.pyplot as P

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