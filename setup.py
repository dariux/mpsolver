import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mpsolver",
    version = "1.0.0",
    author = "Darius Braziunas",
    description = "Mathematical programming (MP) interface to CPLEX and GLPK solvers",
    long_description=read('README.txt'),
    #keywords = "example documentation tutorial",
    url = "http://cs.toronto.edu/~darius",
    packages=['mpsolver'],
    platforms = ['any'],
    install_requires =['numpy']
)

