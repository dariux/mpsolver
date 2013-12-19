import numpy as np
import glpk

from solver import Solver

CTYPES = {'B':bool, 'C':float, 'I':int}
glpk.env.term_on = False

class GLPKSolver(Solver):
    """Generic GLPK problem solver"""
    def __init__(self, p, name='some solver'):
        """p is a problem instance of type MPProb"""
        Solver.__init__(self, p, name)
        self.lp = glpk.LPX()    # Construct an empty linear program.
        self.lp.name = self.name
        self.lp.obj.maximize = self.p.maximize

        # columns
        self.lp.cols.add(self.p.numcols)
        self.lp.obj[:] = list(self.p.obj)
        for c, lb, ub, in zip(self.lp.cols, self.p.lb, self.p.ub):
            if np.isinf(lb):
                lb = None
            if np.isinf(ub):
                ub = None
            c.bounds = lb, ub

        # set variable types
        self.changeVarType(self.p.ctype)

        # rows
        if self.p.numrows > 0:
            self.lp.rows.add(self.p.numrows)
        for i in range(self.p.numrows):
            row = self.lp.rows[i]
            if self.p.sense[i] == 'E':
                row.bounds = self.p.rhs[i]
            elif self.p.sense[i] == 'L':
                row.bounds = None, self.p.rhs[i]
            elif self.p.sense[i] == 'G':
                row.bounds = self.p.rhs[i], None
            else:
                assert False, "wrong sense %s" % (self.p.sense[i],)

        # matrix coefficients
        self.lp.matrix = self.p.A.to_coordinate()


    def __del__(self):
        #print 'Deleting problem', self.name
        try:
            del self.lp
        except:
            pass
        
    def NOTIMPLEMENTEDchangeObjective(self, obj):
        """Change objective function"""
        obj = np.asarray(obj, dtype=float)
        # change objective function
        CPX.chgobj(self.env, self.lp, self.nVars, self.indices, obj)

    def changeVarType(self, ctype):
        for c, ctype in zip(self.lp.cols, ctype):
            c.kind = CTYPES[ctype]
        
    def solve(self, obj=None):
        """Find max obj (obj is objective function)
        If obj is None, use existing obj function
        """
        if obj is not None:
            tmpobj = self.lp.obj[:]
            self.lp.obj[:] = list(obj)

        #self.lp.cpx_basis()
        if self.p.probtype == "LP":
            self.lp.simplex() # or self.lp.interior(), self.lp.exact
            #self.lp.exact() # or self.lp.interior(), self.lp.exact
        elif self.p.probtype == "MILP":
            self.lp.simplex()
            self.lp.integer() # or self.lp.intopt()
        else:
            raise Error("wrong problem type")
        s = self.solution()
        # change objective function back
        if obj is not None:
            self.lp.obj[:] = tmpobj
        return s

    def solution(self):
        """get LP solution"""
        s = {'x': np.array([c.value for c in self.lp.cols]),
             'x primal': [c.primal for c in self.lp.cols],
             #'x dual': [c.dual for c in self.lp.cols],
             'objval': self.lp.obj.value,
             'status': self.lp.status,
             'feasible': self.lp.status in ('feas', 'opt')}
        return s


    def NOTIMPLEMENTEDisFeasible(self, x):
        """Is point x feasible?"""
        x = np.asarray(x, dtype=float)
        assert(len(x) == self.nVars)
        # both lower and upper bound are set to x
        lu = np.array(['B']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, x)      
        feasible = self.solve()['feasible']
        # change bounds back
        lu = np.array(['L']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, self.p.lb)      
        lu = np.array(['U']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, self.p.ub)  
        return feasible

    def writeprob(self, fname=None):
        if fname is None:
            fname = self.name+'.GLPK'
        self.lp.write(cpxlp=fname)

    def addConstraint(self, c, update=True):
        """add general constraint c
        c is a dictionary with keys {'indices', 'coeffs', 'sense', 'rhs'}
        """
        if update:
            self.p.addConstraint(c)
        else:
            self.p.numrows += 1            

        self.lp.rows.add(1)
        row = self.lp.rows[-1]
        if c['sense'] == 'E':
            row.bounds = c['rhs']
        elif c['sense'] == 'L':
            row.bounds = None, c['rhs']
        elif c['sense'] == 'G':
            row.bounds = c['rhs'], None
        else:
            assert False, "wrong sense"
        # make sure indices are ints, not numpy.int64s
        indices = [int(i) for i in c['indices']]
        row.matrix = zip(indices, c['coeffs'])

        self.lp.cpx_basis()

    def removeLastConstraint(self):
        del self.lp.rows[-1:]
        self.p.removeLastConstraint()
        self.lp.cpx_basis()

    def removeLastConstraints(self, n):
        """remove n last constraints"""
        del self.lp.rows[-n:]
        self.p.removeLastConstraints(n)
        self.lp.cpx_basis()


