import numpy as np

from pycplex import CPX
from pycplex import cplexcodes as C
from mpprob import MPProb

from solver import Solver

OPTIMAL = [C.CPX_STAT_OPTIMAL, C.CPXMIP_OPTIMAL, C.CPXMIP_OPTIMAL_TOL]
UNBOUNDED = [C.CPX_STAT_UNBOUNDED, C.CPXMIP_UNBOUNDED]
INFEASIBLE = [C.CPX_STAT_INFEASIBLE, C.CPXMIP_INFEASIBLE, C.CPXMIP_INForUNBD]

#Inf = C.CPX_INFBOUND # 1e20
#LP = C.CPXPROB_LP
#QP = C.CPXPROB_QP
#MILP = C.CPXPROB_MILP
#MIQP = C.CPXPROB_MIQP

class CPLEXSolver(Solver):
    """Generic CPLEX problem solver"""
    def __init__(self, p, name='some solver'):
        """p is a problem instance of type MPProb"""
        # do this once ever
        if Solver.env is None:
            Solver.env = CPX.openCPLEX()

        Solver.__init__(self, p, name)

        self.indices = np.arange(p.numcols, dtype=np.int32)
        self.lp = CPX.createprob(self.env, self.name)
        if p.maximize is True:
            self.objsen = C.CPX_MAX # maximization problem
        else:
            self.objsen = C.CPX_MIN # minimization problem

        # initialize bounds
        self.lb = p.lb.copy()
        self.lb[np.isinf(self.lb)] = -C.CPX_INFBOUND
        self.ub = p.ub.copy()
        self.ub[np.isinf(self.ub)] = C.CPX_INFBOUND
        s = p.A.to_cplex()
        CPX.copylp(self.env, self.lp, p.numcols, p.numrows, self.objsen, 
                   p.obj, p.rhs, p.sense,
                   s['matbeg'], s['matcnt'], s['matind'], s['matval'], 
                   self.lb, self.ub)
        CPX.copyctype(self.env, self.lp, p.ctype)

    def __del__(self):
        #print 'Deleting problem', self.name
        try:
            CPX.freeprob(self.env, self.lp)
        except:
            pass

    def changeVarType(self, ctype):
        CPX.copyctype(self.env, self.lp, np.asarray(ctype))
        
    def close(self):
        CPX.closeCPLEX(self.env)
        self.env = None

    def changeObjective(self, obj):
        """Change objective function"""
        obj = np.asarray(obj, dtype=float)
        # change objective function
        CPX.chgobj(self.env, self.lp, self.nVars, self.indices, obj)
        
    def solve(self, obj=None):
        """Find max obj (obj is objective function)
        If obj is None, use existing obj function in CPLEX
        """
        if obj is not None:
            obj = np.asarray(obj, dtype=float)
            # change objective function
            CPX.chgobj(self.env, self.lp, self.nVars, self.indices, obj)
            
        CPX.lpopt(self.env, self.lp)
        solution = self.solution()
        
        if obj is not None:
            # change objective function back
            CPX.chgobj(self.env, self.lp, self.nVars, self.indices, self.p.obj)
        return solution

    def solution(self):
        """get LP solution"""
        #begin, end = 0, self.p.nVars-1 #inclusive index
        s = {'x':None, 'objval':None, 'status':0, 'feasible':False}
        lpstat = CPX.getstat(self.env, self.lp)
        s['x'] = CPX.getx(self.env, self.lp)
        s['objval'] = CPX.getobjval(self.env, self.lp)
        if lpstat in OPTIMAL:
            s['optimal'] = True
            s['feasible'] = True
        else:
            pass
            #print self.name, "Solution unbounded or infeasible. Code: ", lpstat
        s['status'] = lpstat
        return s
    
    def isFeasible(self, x):
        """Is point x feasible?"""
        x = np.asarray(x, dtype=float)
        assert(len(x) == self.nVars)
        # both lower and upper bound are set to x
        lu = np.array(['B']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, x)      
        feasible = self.solve()['feasible']
        # change bounds back
        lu = np.array(['L']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, self.lb)      
        lu = np.array(['U']*self.p.numcols)
        CPX.chgbds(self.env, self.lp, self.nVars, self.indices, lu, self.ub)  
        return feasible

    def writeprob(self, fname=None):
        if fname is None:
            fname = self.name+'.LP'
        CPX.writeprob(self.env, self.lp, fname)
        
    def addConstraint(self, c, update=True):
        """add general constraint c
        c is a dictionary with keys {'indices', 'coeffs', 'sense', 'rhs'}
        """
        if update:
            self.p.addConstraint(c)
        else:
            self.p.numrows += 1            

        rhs = np.asarray([c['rhs']],dtype=float)
        sense = np.asarray([c['sense']])
        #rcnt = 1
        CPX.newrows(self.env, self.lp, 1, rhs, sense)
        for i,a in zip(c['indices'],c['coeffs']):
            CPX.chgcoef(self.env, self.lp, self.p.numrows-1, i, a)
 
    def removeLastConstraint(self):
        CPX.delrows(self.env, self.lp, self.p.numrows-1, self.p.numrows-1)
        self.p.removeLastConstraint()
                   
if __name__ == "__main__":
    A = [[1,2]]
    b = [4]
    s = ['L']

    #A = [[1,0],[0,1],[1,0]]
    #b = [4,2,4]
    #s = ['L','L','L']
    
    R = MPProb(0,2)
    R.lb[:] = 0
    R.ub[:] = 4
    R.setA(A)
    R.setRHS(b)
    R.setSense(s)
    
    sampler = Sampler(R)
    points = sampler.sample(100)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    print x
    print y

