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

import numpy as np

from sparsematrix import Matrix

class MPProb(object):
    """Mathematical Programming problem
    By default, maximize=True, and obj is all zeros
    """

    def __init__(self, numrows, numcols):
        """
        Objective: all zeros, maximize
        Variable bounds: [-Inf, Inf]
        Probtype: LP
        """

        assert numcols > 0
        assert numrows >= 0
        self.numcols = numcols
        self.numrows = numrows
        self.probtype = "LP" # (LP, QP, MILP, MIQP)
        self.maximize = True

        self.obj = np.zeros((numcols,)) # numvars
        self.ctype = np.empty((numcols,), '|S1')
        self.ctype[:] = 'C' # {C,B,I} C = continuous
        self.lb = -np.Inf * np.ones((numcols,))
        self.ub = np.Inf * np.ones((numcols,))

        # make it a bit more difficult to set A and Q directly
        self.__dict__['A'] = Matrix(numrows, numcols)
        self.__dict__['Q'] = Matrix() # for quadratic objective
        self.rhs = np.zeros((numrows,))
        # 'L' (<=), 'E' (=), 'G' (>=), 'R' (range)
        self.sense = np.empty((numrows,),'|S1')
        self.sense[:] = 'E' # set by default
        self.rngval = None # or np.zeros((numrows,))

    def __setattr__(self, name, value):
        if name == 'A':
            raise AttributeError("Use setA() to update the matrix A")
        else:
            self.__dict__[name] = value

    def setA(self, A, format='matrix', params=None):
        """Set constraints matrix A"""
        if format == 'matrix':
            A = np.asarray(A, dtype=float)
            i,j = np.nonzero(A)
            v = A[(i,j)]
            nnz = len(v)
            
            numrows, numcols = A.shape
            assert(self.numcols == numcols)
            self.numrows = numrows
            self.A.init(numrows, numcols, nnz)
            for k in range(nnz):
                self.A[int(i[k]),int(j[k])] = v[k]

        if format == 'coord':
            assert(self.numcols == params['m'])
            self.numrows = params['n']
            self.A.init(self.numrows, self.numcols, len(A))
            for i,j,v in A:
                self.A[int(i),int(j)] = v

    def setQ(self, Q):
        """Set quadratic objective matrix Q"""
        raise NotImplementedError, "not implemented yet"

    def setRHS(self,rhs):
        """Set rhs vector rhs"""
        self.rhs = np.asarray(rhs, dtype=float)

    def setSense(self,sense):
        """Set sense vector sense"""
        self.sense = np.asarray(sense, '|S1')

    def addConstraint(self, c):
        # c = {'indices', 'coeffs', 'sense', 'rhs'}
        assert(len(c['indices']) == len(c['coeffs']))
        Arow = np.zeros((1,self.numcols))
        for i,a in zip(c['indices'],c['coeffs']):
            Arow[0,i] = a
        self.addConstraintRows((Arow, [c['rhs']], [c['sense']]))
        
    def addComparisonConstraint(self, c):
        # c = {'index1', 'sense', 'index2'}
        if c['index1'] != c['index2']:
            Arow = np.zeros((1,self.numcols))
            Arow[0, c['index1']] = 1
            Arow[0, c['index2']] = -1
            self.addConstraintRows((Arow, [0], [c['sense']]))

    def addBoundConstraint(self, c):
        # c = {'index', 'sense', 'val'}
        Arow = np.zeros((1,self.numcols))
        Arow[0, c['index']] = 1
        self.addConstraintRows((Arow, [c['val']], [c['sense']]))

    def addConstraintRows(self, r):
        # Arows = r[0], rhs = r[1], sense = r[2]
        self.A.add_rows(r[0])
        self.rhs = np.concatenate([self.rhs, r[1]])
        self.sense = np.concatenate([self.sense, r[2]])
        self.numrows += len(r[1])
        
    def removeLastConstraint(self):
        self.removeLastConstraints(1)

    def removeLastConstraints(self, n):
        self.A.remove_last_rows(n)
        self.rhs = self.rhs[:-n]
        self.sense = self.sense[:-n]
        self.numrows -= n

    def validate(self):
        assert self.numrows >= 0
        assert self.numcols > 0
        assert self.numrows == self.A.shape[0]
        assert self.numcols == self.A.shape[1]

        assert self.maximize in (True, False)

        assert(self.obj is not None and len(self.obj) == self.numcols)
        assert len(self.lb) == self.numcols
        assert len(self.ub) == self.numcols

        assert(self.rhs is not None and len(self.rhs) == self.numrows)
        assert(self.sense is not None and len(self.sense) == self.numrows)
        assert(self.rngval is None or len(self.rngval) == self.numrows)
        
        self.obj = np.asarray(self.obj, dtype=float)
        self.lb = np.asarray(self.lb, dtype=float)
        self.ub = np.asarray(self.ub, dtype=float)
        
        self.rhs = np.asarray(self.rhs, dtype=float)
        self.sense = np.asarray(self.sense, '|S1')
        if self.rngval is not None:
            self.rngval = np.asarray(self.rngval, dtype=float)

        if self.probtype in ("MILP", "MIQP"):
            assert(self.ctype is not None and len(self.ctype) == self.numcols)
            self.ctype = np.asarray(self.ctype, '|S1')
            # types: 'C' (continuous), 'B' (binary), 'I' (integer)

        if self.probtype in ("LP") and self.ctype is not None:
            for c in self.ctype:
                assert c == 'C'
            

    def prettyprint(self):
        def s(x):
            s = {0:'.', 1:'+', -1:'-', 'E':'= ', 'L':'<=', 'G':'>='}
            if x in s:
                return s[x]
            else:
                # return str(x)
                if x > 0:
                    return '+'
                else:
                    return '-'

            
        n,m = self.A.shape
        #print ' '.join([f(x) for x in self.obj])
        #print
        for i in range(n):
            row = np.empty((m,))
            for j in range(m):
                row[j] = self.A[i,j]
            # or row = self.A[i,:] if A is not a sparse matrix
            print ' '.join([s(x) for x in row]), " ", s(self.sense[i]), " ", int(self.rhs[i])


