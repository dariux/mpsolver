import numpy as N

class Solver(object):
    """Generic mathematic programming problem solver"""
    env = None

    def __init__(self, p, name='some solver'):
        """p is a problem instance of type MPProb"""
        self.name = name
        self.p = p
        self.p.validate()
        self.nVars = p.numcols

        self.options = {}

    def __del__(self):
        print 'Deleting solver', self.name
  
    def close(self):
        self.env = None

    def addConstraints(self, constraints, update=True):
        for c in constraints:
            self.addConstraint(c,update)

    def addComparisonConstraint(self, c):
        """add a comparison constraint between two variables
        c is a dictionary with keys {'index1', 'sense', 'index2'}
        var[index1] {'G','L','E'} var[index2]
        E.g., var[3] >= var[5]
        """
        if c['index1'] != c['index2']:
            c['indices'] = [c['index1'], c['index2']]
            c['coeffs'] = [1.0, -1.0]
            c['rhs'] = 0.0
            self.addConstraint(c)

    def addBoundConstraint(self, c):
        """add a bound constraint on a single parameter
        c is a dictionary with keys {'index', 'sense', 'val'}
        var[index] {'G','L','E'} val
        E.g., var[3] >= 6
        """
        c['indices'] = [c['index']]
        c['coeffs'] = [1.0]
        c['rhs'] = c['val']
        self.addConstraint(c)
   
    def testConstraint(self, c, obj):
        """Add constraint c, solve, remove constraint"""
        self.addConstraint(c)
        s = self.solve(obj)
        self.removeLastConstraint()
        return s
    
    def testBoundConstraint(self, c, obj=None):
        """Add constraint c, solve, remove constraint"""
        self.addBoundConstraint(c)
        s = self.solve(obj)
        self.removeLastConstraint()
        return s
    
    def testComparisonConstraint(self, c, obj=None):
        """Add constraint c, solve, remove constraint"""
        self.addComparisonConstraint(c)
        s = self.solve(obj)
        self.removeLastConstraint()
        self.lp.cpx_basis()
        return s



# TODO: generalize for non-CPLEX solvers
class Sampler(Solver):
    """Used for sampling from a convex polytope p"""
    def __init__(self, p0):
        # build p with one extra column
        p = MPProb(0, p0.numcols+1)
        p.maximize = True
        p.obj[-1] = 1 # extra variable t
        p.lb[:-1] = p0.lb
        p.ub[:-1] = p0.ub
        # add identity matrix underneath
        A = N.vstack([p0.A, N.eye(p0.numcols)])
        rhs = N.concatenate([p0.rhs, [0] * p0.numcols])
        sense = N.concatenate([p0.sense, ['E'] * p0.numcols])
        # add a column of zeros to A
        A = N.hstack([A, N.zeros((p0.numrows+p0.numcols,1))])
        p.setA(A)
        p.setRHS(rhs)
        p.setSense(sense)
        
        Solver.__init__(self, p, 'sampler')
        self.p0 = p0
        
        # last nVars rows
        self.rowlist = N.arange(p.numrows-p0.numcols, p.numrows, dtype=N.int32)
        # last column
        self.collist = (p.numcols-1) * N.ones(p0.numcols, dtype=N.int32)   

        # a feasible point (not always)
        self.start = p0.lb
        #assert(self.isFeasible(self.start))

    def lineintersection(self,a,b):
        numcoefs = self.p0.numcols
        assert(numcoefs == len(a) == len(b))
        # line is a + tb
        a = N.asarray(a, dtype=float) # rhs
        b = N.asarray(b, dtype=float) # last column

        #changes a list of matrix coefficients 
        CPX.chgcoeflist(self.env, self.lp, numcoefs, self.rowlist, self.collist, b)
        CPX.chgrhs(self.env, self.lp, numcoefs, self.rowlist, a)
        return self.solve()['x'][:-1]

    def sample(self, nSamples=10):
        samples = [None] * nSamples
        for i in range(nSamples):
            samples[i] = self.walk()
        return samples
        
    def walk(self, steps=1000):
        points = [None] * steps
        a = self.start
        for i in range(steps):
            points[i] = a
            # direction
            b = self.sampleDirection()
            # intersection
            c = self.lineintersection(a,b)
            # next point along direction
            a = a + N.random.random_sample() * (c-a)
        return points[-1] # return last point
    
    def sampleDirection(self):
        d = N.random.standard_normal((self.p0.numcols,))
        d = d/N.dot(d,d)
        
        # same direction for equal vars
        try:
            for eqs in self.p0.u.equalparams:
                val = d[eqs[0]]
                for i2 in eqs[1:]:
                    d[i2] = val
        except:
            pass
        return d
