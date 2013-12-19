## Copyright (c) 2006-2008 Darius Braziunas

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
from pycplex import CPX
from pycplex import cplexcodes as C
from pycplex.mpprob import MPProb


Inf = C.CPX_INFBOUND
LP = C.CPXPROB_LP
QP = C.CPXPROB_QP
MILP = C.CPXPROB_MILP
MIQP = C.CPXPROB_MIQP

OPTIMAL = [C.CPX_STAT_OPTIMAL, C.CPXMIP_OPTIMAL, C.CPXMIP_OPTIMAL_TOL]
UNBOUNDED = [C.CPX_STAT_UNBOUNDED, C.CPXMIP_UNBOUNDED]
INFEASIBLE = [C.CPX_STAT_INFEASIBLE, C.CPXMIP_INFEASIBLE, C.CPXMIP_INForUNBD]


def test1():
    """Call CPLEX to solve LP

    Minimize
    obj: x + 4 y + 9 z
    Subject To
    c1: y + x <= 5
    c2: z + x >= 10
    c3: z - y = 7
    Bounds
    x <= 4
    -1 <= y <= 1
    """

    # define MPProb object
    p = MPProb(0,3) # numrows, numcols
    p.objsen = C.CPX_MIN
    p.obj = [1,4,9]
    p.ub[0] = 4
    p.lb[1], p.ub[1] = -1,1

    #constraints
    A = [[1,1,0], # constraint matrix
         [1,0,1],
         [0,-1,1]]
    rhs = [5,10,7]
    sense = ['L','G','E']
    
    p.setA(A) 
    p.setRHS(rhs)
    p.setSense(sense)
 
    #p.addComparisonConstraint({'index1':0, 'sense':'L', 'index2':2})
    #p.addBoundConstraint({'index':1, 'sense':'L', 'val':200})

    # verify all input parameters are set up
    p.prepare()
    
    # open CPLEX environment, set some parameters
    env = CPX.openCPLEX()
    CPX.setintparam(env, C.CPX_PARAM_SCRIND, C.CPX_ON) # print info to screen
    CPX.setintparam(env, C.CPX_PARAM_DATACHECK, C.CPX_ON)

    # create CPLEX problem, add objective and constraints to it
    lp = CPX.createprob(env, 'test1')
    CPX.copylp(env, lp, p.numcols, p.numrows, p.objsen, 
               p.obj, p.rhs, p.sense,
               p.matbeg, p.matcnt, p.matind, p.matval, 
               p.lb, p.ub) #rngval optional
    # change bounds of x1 and x3 (x1 <= 3, x3=5)
    #indices = N.array([0,2],dtype=N.int32)
    #lu = N.array(['L','B'])
    #bd = N.array([-3.0,5.0])
    #CPX.chgbds(env,lp,2,indices,lu,bd)

    status = CPX.writeprob(env, lp, "test1.lp") # write problem to file                

    # solve the problem
    CPX.lpopt(env,lp)

    # get the solution
    lpstat = CPX.getstat(env,lp)
    if lpstat in OPTIMAL:
        x, objval = CPX.getx(env,lp), CPX.getobjval(env,lp)
        print x, objval
    elif lpstat in UNBOUNDED: print "Solution unbounded"
    elif lpstat in INFEASIBLE: print "Solution infeasible"
    else: print "Solution code: ", lpstat
    print
    
    CPX.freeprob(env,lp) # free the problem
    CPX.closeCPLEX(env) # close CPLEX environment


def test2():
    """Call CPLEX to solve MILP

    Minimize
    obj: x + 4 y + 9 z
    Subject To
    c1: y + x <= 5
    c2: z + x >= 10
    c3: z - y = 7
    Bounds
    x <= 4
    -1 <= y <= 1
    Types
    x is continuous, y and z boolean
    """

    # define MPProb object
    p = MPProb(3,3) # numrows, numcols
    p.objsen = C.CPX_MIN
    p.obj = [1,4,9]
    A = [[1,1,0], # constraint matrix
         [1,0,1],
         [0,-1,1]]
    p.setA(A) 
    p.rhs = [5,10,7]
    p.sense = ['L','G','E']
    p.ub[0] = 4
    p.lb[1], p.ub[1] = -1,1

    # make the problem mixed integer problem
    p.probtype = MILP
    p.ctype = ['I','B','I']

    # verify all input parameters are set up
    p.prepare()
    
    # open CPLEX environment, set some parameters
    env = CPX.openCPLEX()
    CPX.setintparam(env, C.CPX_PARAM_SCRIND, C.CPX_ON) # print info to screen
    CPX.setintparam(env, C.CPX_PARAM_DATACHECK, C.CPX_ON)

    # create CPLEX problem, add objective and constraints to it
    lp = CPX.createprob(env, 'test2')
    
    CPX.copylp(env, lp, p.numcols, p.numrows, p.objsen, 
               p.obj, p.rhs, p.sense,
               p.matbeg, p.matcnt, p.matind, p.matval, 
               p.lb, p.ub) # rngval is optional
    CPX.copyctype(env, lp, p.ctype)
    status = CPX.writeprob(env, lp, "test2.lp") # write problem to file
                
    # solve the problem
    CPX.mipopt(env,lp) # can call lpopt, too

    # get the solution
    lpstat = CPX.getstat(env,lp)
    if lpstat in OPTIMAL:
        # can call getx, getobjval, too
        x, objval = CPX.getmipx(env,lp), CPX.getmipobjval(env,lp)
        print x, objval
    elif lpstat in UNBOUNDED: print "Solution unbounded"
    elif lpstat in INFEASIBLE: print "Solution infeasible"
    else: print "Solution code: ", lpstat
    print
    
    CPX.freeprob(env,lp) # free the problem
    CPX.closeCPLEX(env) # close CPLEX environment


# thanks to Christina Fabritius Eskebaek for adding this test case
def test3():
    """Call CPLEX to solve MIQP

    Maximize
    obj: x + 2y + 3z - 0.5(33*x*x + 22*y*y + 11*z*z - 12*x*y - 23*y*z)
    Subject To
    c1: -x + y + z <= 20
    c2: x - 3y + z <= 30
    Bounds
    0 <= x <= 40
    0 <= y
    0 <= z
    Types
    """

    # define MPProb object
    p = MPProb(2,3) # numrows, numcols
    p.objsen = C.CPX_MAX
    p.obj = [1,2,3]
    A = [[-1,1,1], # constraint matrix
         [1,-3,1]]
    p.setA(A) 
    p.rhs = [20,30]
    p.sense = ['L','L']
    p.lb[0], p.ub[0] = 0, 40
    p.lb[1] = 0
    p.lb[2], p.ub[2] = 0, 10

    p.probtype = QP

    # make the problem a quadratic problem
    Q = [[ -33,   6,     0   ],
	 [   6, -33,    11.5 ],
	 [   0,  11.5, -33   ]]
    p.setQ(Q)

    # verify all input parameters are set up
    p.prepare()

    # open CPLEX environment, set some parameters
    env = CPX.openCPLEX()
    CPX.setintparam(env, C.CPX_PARAM_SCRIND, C.CPX_ON) # print info to screen
    CPX.setintparam(env, C.CPX_PARAM_DATACHECK, C.CPX_ON)

    # create CPLEX problem, add objective and constraints to it
    lp = CPX.createprob(env, 'test3')
    
    CPX.copylp(env, lp, p.numcols, p.numrows, p.objsen, 
               p.obj, p.rhs, p.sense,
               p.matbeg, p.matcnt, p.matind, p.matval, 
               p.lb, p.ub) # rngval is optional
    CPX.copyquad(env, lp, p.qmatbeg, p.qmatcnt, p.qmatind, p.qmatval)
    # to make the problem mixed integer quadratic problem, do this:
    # p.probtype = MIQP
    # p.ctype = ['I', 'I', 'I']
    # CPX.copyctype(env, lp, p.ctype)

    status = CPX.writeprob(env, lp, "test3.lp") # write problem to file
                
    # solve the problem
    CPX.qpopt(env,lp) # can call lpopt, to


    # get the solution
    lpstat = CPX.getstat(env,lp)
    if lpstat in OPTIMAL:
        x, objval = CPX.getmipx(env,lp), CPX.getobjval(env,lp)
        print x, objval
    elif lpstat in UNBOUNDED: print "Solution unbounded"
    elif lpstat in INFEASIBLE: print "Solution infeasible"
    else: print "Solution code: ", lpstat
    print

    CPX.freeprob(env,lp) # free the problem
    CPX.closeCPLEX(env) # close CPLEX environment

if __name__ == "__main__":

    # run 3 tests
    test1()
    test2()
    test3()
