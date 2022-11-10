# PRISM
PRISM (Polarized Radiation Intensity from Saturated Masers) is a package for calculating and plotting solutions to the 1-dimensional maser polarization formalism presented in Tobin, Gray, and Kemball 2022.

## Citations
If you use this code, we request that you cite the corresponding article:

> Tobin, Gray, & Kemball 2022 (in review)

And cite this repository:

- v1.0.0 (version used in above paper): [![DOI](https://zenodo.org/badge/476032047.svg)](https://zenodo.org/badge/latestdoi/476032047)

## System Requirements

**Required:**
- Python 3
- NumPy &ge; 1.16
  - Required as code uses numpy's matmul function to exist and be capable of handling ufunc arguments.
- astropy
- scipy
- matplotlib

**Recommended:**
- tol_colors ([Available on GitHub](https://github.com/Descanonge/tol_colors))
  - Used by many of the contained plotting functions for selection of colorblind-friendly color maps.

## Installation

Package may be installed either by cloning the Git package, changing to the top-level prism directory in your download, (activating your desired anaconda environment, if necessary,) and entering in the command line:

```
python setup.py install
```

Alternately, if you have `git` set up from your command line, you can run instead:

```
pip install git+https://github.com/tltobin/prism.git
```

## Usage Guide

There are two primary classes that users will interact with in this package: the `maser` class, and the `maser_v_theta` class. Both of these are located in `prism.prism`. The `maser` class is the class used for computing solutions, while the `maser_v_theta` class will primarily be used for organizing and analyzing the solutions after computation.

As both of these classes require many parameters, the easiest way to interface with them will be through the parameter file.

Finally, there are two additional plotting functions in `prism.prism_plots`.

Typical use will look like the following:

- Customize the parameter file to contain common parameters describing all of the masers that you want to calculate solutions for.

- Write an (optionally multi-processing-enabled) script that:
    
    - initializes a `maser` class object for each parameter set desired; common parameters can be read from the file, while parameters that you are varying can be provided directly.
    
    - calculates their Faraday coefficient, if a non-zero value is desired,
    
    - calculates the inversion solution for each `maser` object at each total optical depth &tau;<sub>f</sub>. Recommend gradually stepping through increasing values of &tau;<sub>f</sub>, using the solution from each as the initial guess for the next.

- You may want to check the convergence of the calculated `LDI_terms` to ensure that the number of terms that you're using is high enough.

- Once solutions are calculated, the `maser` class and `maser_v_theta` class have several functions that can calculate other values for these solutions, such as the unitless stokes parameters or the stimulated emission rate, R. These values can take a long time to compute in some cases, so these functions are often equipped with options to save the results to a file or read results from an existing file, to prevent having to re-run calculations.

- Finally, the `maser` class has two plotting functions for individual solutions, but the `maser_v_theta` class has a larger range of plotting methods for analyzing the results as a function of &theta;. The `prism.prism_plot` module also includes two additional plotting functions that take, as arguments, lists of `maser` or `maser_v_theta` class objects, for more flexibility in plotting comparisons across parameter sets.
