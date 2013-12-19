## Copyright (c) 2006-2009 Darius Braziunas

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

import numpy as N
try:
    import pysparse
    MATRIXFORMAT = 'pysparse'
except:
    MATRIXFORMAT = 'numpy'
#MATRIXFORMAT = 'numpy'

class Matrix(object):
    """A wrapper for different sparse matrix implementations"""
    def __init__(self, n=0, m=0, sizeHint=1000):
        self.init(n,m,sizeHint)

    def init(self, n, m, sizeHint=1000):
        if MATRIXFORMAT == 'pysparse':
            self.matrix = pysparse.spmatrix.ll_mat(n,m,sizeHint)
        elif MATRIXFORMAT == 'numpy':
            self.matrix = N.zeros((n,m))

    """Return a list of (i,j,v) triples"""
    def to_coordinate(self):
        if MATRIXFORMAT == 'pysparse':
            return [(i,j,v) for (i,j),v in self.matrix.items()]
        elif MATRIXFORMAT == 'numpy':
            def toint(x): return int(x)
            i,j = N.nonzero(self.matrix)
            v = self.matrix[(i,j)]
            return zip(map(toint,i),map(toint,j),v)

    def nnz(self):
        if MATRIXFORMAT == 'pysparse':
            return self.matrix.nnz
        else:
            return len(self.matrix.nonzero()[0])

    def add_row(self, row):
        if MATRIXFORMAT == 'pysparse':
            # would have to create a new matrix, and copy old+row...
            raise NotImplementedError, "Adding rows not implemented for pysparse matrices"
        elif MATRIXFORMAT == 'numpy':
            self.matrix = N.vstack([self.A, row])

    def add_rows(self, rows):
        if MATRIXFORMAT == 'pysparse':
            raise NotImplementedError, "Adding rows not implemented for pysparse matrices"
        elif MATRIXFORMAT == 'numpy':
            self.matrix = N.vstack([self.matrix, rows])

    def remove_last_rows(self, n):
        if MATRIXFORMAT == 'pysparse':
            raise NotImplementedError, "Removing rows not implemented for pysparse matrices"
        elif MATRIXFORMAT == 'numpy':
            self.matrix = self.matrix[:-n,:]

    def to_cplex(self):
        """Convert matrix A to CPLEX sparse representation
        Thanks to Stephen Hartke for the code
        """

        nnz = self.nnz()
        #print "Matrix format: %s, size: %s, num of entries: %d" % (MATRIXFORMAT,
        #                                                           self.matrix.shape,
        #                                                           nnz) 

        numrows, numcols = self.matrix.shape

        matval = N.empty((nnz,), dtype=float)
        matind = N.empty((nnz,), dtype=N.int32)
        matbeg = N.empty((numcols,), dtype=N.int32)
        matcnt = N.empty((numcols,), dtype=N.int32)
        i = 0
        for col in xrange(0, numcols):
            if col+1 % 100 == 0: 
                print col, " ", 
            matbeg[col] = i
            cur_row_count = 0

            if MATRIXFORMAT == 'pysparse':
                for (row,tmpcol),v in self.matrix[:,col].items():
                    matval[i] = v
                    matind[i] = row
                    i += 1
                    cur_row_count += 1
                
            elif MATRIXFORMAT == 'numpy':
                for row in xrange(0, numrows):
                    if self.matrix[row,col] != 0:
                        matval[i] = self.matrix[row,col]
                        matind[i] = row
                        i += 1
                        cur_row_count += 1

            matcnt[col] = cur_row_count
        assert i == self.nnz(), (i, self.nnz())

        return {'matval':matval, 'matind':matind,
                'matbeg':matbeg, 'matcnt':matcnt}

    def __getattr__(self, name):
        return getattr(self.matrix, name)

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __getitem__(self, key):
        return self.matrix[key]

    def TMP__repr__(self):
        return repr(self.matrix)

    def TMP__str__(self):
        return str(self.matrix)
        
