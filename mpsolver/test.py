## Copyright (c) 2006-2011 Darius Braziunas

## Permission is hereby granted, free of charge, to any person obtaining 
## a copy of this software and associated documentation files (the "Software"), 
## to deal in the Software without restriction, including without limitation the 
## rights to use, copy, modify, merge, publish, distribute, sublicense, 
## and/or sell copies of the Software, and to permit persons to whom 
## the Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all 
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
## OTHER DEALINGS IN THE SOFTWARE.


#SOLVER = 'CPLEX'
SOLVER = 'GLPK'

import numpy as np

if SOLVER == 'CPLEX':
    from mpsolver.cplexsolver import CPLEXSolver as Solver
elif SOLVER == 'GLPK':
    from mpsolver.glpksolver import GLPKSolver as Solver
from mpsolver.mpprob import MPProb

import mpsolver.sparsematrix
mpsolver.sparsematrix.MATRIXFORMAT = 'numpy'


def test1():
    """Solve LP

    Minimize
    obj: x + 4 y + 9 z
    Subject To
    c1: y + x <= 5
    c2: z + x >= 10
    c3: z - y >= 8
    Bounds
    x <= 4
    -1 <= y <= 1

    Solution: [x = 3, y = -1, z = 7], 'objval': 62.0
    """

    # define MPProb (mathematical programming) object
    p = MPProb(0,3) # numrows, numcols

    p.maximize = False # default is True (maximize)
    p.obj = [1,4,9] # objective coefficients
    p.lb[0], p.ub[0] = -np.Inf, 4
    p.lb[1], p.ub[1] = -1, 1

    p.ctype[:] = 'C' # all variables are continuous (could be 'I' integer or 'B' binary also)
    p.probtype = 'LP' # (LP, QP, MILP, MIQP)

    # constraints: we will create two constraints first, and then add the third constraint later
    # we can use either full or sparse representation
    A = [[1,1,0], # constraint matrix (first two rows)
         [1,0,1]]
    # sparse representation: list of (i,j,v) where v is non-zero
    A2 = [(0,0,1), (0,1,1), (1,0,1), (1,2,1)]
    b = [5,10]
    sense = ['L','G'] # either 'L', 'G', or 'E'

    # add constraints to object p
    numrows = 2
    p.setA(A, format='matrix')
    # p.setA(A2, format='coord', params={'n':numrows, 'm':p.numcols}) # for sparse matrix A2
    p.setRHS(b)
    p.setSense(sense)

    p.validate() # make sure everything is OK

    #############
    # now we have MPProb object p, but only with two constraint rows
    # let's create a solver

    s = Solver(p, name='test1 solver')
    solution = s.solve()
    print "Partial solution", solution

    # Let's add the third constraint row directly to the solver (and update the MPProb object p as well)
    # constraint c is a dictionary with keys {'indices', 'coeffs', 'sense', 'rhs'}
    # third row constraint is [0x,-1y,1z]] >= 8
    c = {'indices': (1,2),
         'coeffs': (-1,1),
         'sense': 'G',
         'rhs': 8
         }
    s.addConstraint(c, update=True)
    solution = s.solve()
    print "Final solution", solution


if __name__ == "__main__":

    test1()

