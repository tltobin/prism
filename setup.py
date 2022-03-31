import os
import sys

from setuptools import find_packages, setup

# Simple function to read text from a file
#     Used by get_version
def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()



# Short function to read package version from a version file
#    version file contains line of format, eg.: __version__ = "0.9"
def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")



# Gets long package description from the README file
long_description = read("README.md")


# Actual package setup
setup(
    name="prism",
    version=get_version("VERSION"),
    description="1D Maser Polarized Radiative Transfer Code with Faraday Rotation from GTK.",
    long_description=long_description,
    url="https://github.com/tltobin/prism",
    
    # Project URLS - set these up when URLS are set up
    # project_urls={
    #     "Documentation": "https://pip.pypa.io",
    #     "Source": "https://github.com/pypa/pip",
    #     "Changelog": "https://pip.pypa.io/en/stable/news/",
    # },
    
    author="Taylor Tobin",
    # Author email - decide which email to distribute
    # author_email="distutils-sig@python.org",
    
    package_dir={'': 'src'},
    packages= [ 'prism' ],
    # find_packages( where="prism" ),
    
    # Use these lines to prevent installation without specific versions of python & required subpackages
    # python_requires=">=3.7, <4",
    # install_requires=["peppercorn"],
    
    # Can use this to install any additional data files that need to be installed with package
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
)