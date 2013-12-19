mpsolver
========

Python code for solving mathematical programming problems


Installation
------------

### GLPK
* Install GLPK version 4.48 (newer ones don't work with PyGLPK): ``wget http://ftp.gnu.org/gnu/glpk/glpk-4.48.tar.gz``
* Install a fork of PyGLPK from https://github.com/cgaray/pyglpk: ``pip install -U https://github.com/cgaray/pyglpk/tarball/master``


Testing
-------

Run ``python test.py`` to start. Solution should be ``x = 3, y = -1, z = 7``, with max objective value ``62.0``.
