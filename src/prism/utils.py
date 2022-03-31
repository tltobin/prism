# Created 3/30/2021
#     - Split txt_to_fits from primary prism.py file for organization

from glob import glob


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
