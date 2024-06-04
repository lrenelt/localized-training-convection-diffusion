# -*- tab-width: 4; indent-tabs-mode: nil  -*-

import numpy as np
import pymor

from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

from common.dune_istl_bindings import DuneIstlVectorSpace

def boundaryFunctionFromCoefficients(coeffs, config):
    c = coeffs.to_numpy()[0,:]
    n = c.shape[0] // 4

    s = config['grid']['oversampling_stride']
    lx = config['grid']['lx'] + 2*s
    ly = config['grid']['ly'] + 2*s

    k = config['oversamplingDegree']

    def funcP0(x,y):
        tol = 1e-10
        xtransformed = (x+s)/lx # normalize to [0,1]
        ytransformed = (y+s)/ly # normalize to [0,1]

        # go anti-clockwise
        if xtransformed < tol:
            idx = int(np.floor(n*(1-ytransformed)))
        elif ytransformed < tol:
            idx = n + int(np.floor(n*xtransformed))
        elif xtransformed > 1-tol:
            idx = 2*n + int(np.floor(n*ytransformed))
        elif ytransformed > 1-tol:
            idx = 3*n + int(np.floor(n*(1-xtransformed)))
        else:
            return 0

        return float(c[idx])

    def funcP1(x,y):
        tol = 1e-10
        xtransformed = (x+s)/lx # normalize to [0,1]
        ytransformed = (y+s)/ly # normalize to [0,1]

        # go anti-clockwise
        if xtransformed < tol:
            idx = int(np.floor(n*(1-ytransformed)))
            alpha = n*(1-ytransformed) - idx
        elif ytransformed < tol:
            idx = n + int(np.floor(n*xtransformed))
            alpha = n + n*xtransformed - idx
        elif xtransformed > 1-tol:
            idx = 2*n + int(np.floor(n*ytransformed))
            alpha = 2*n + n*ytransformed - idx
        elif ytransformed > 1-tol:
            idx = 3*n + int(np.floor(n*(1-xtransformed)))
            alpha = 3*n + n*(1-xtransformed) - idx
        else:
            return 0

        clow = float(c[idx])
        if idx == c.shape[0]-1:
            chigh = float(c[0])
        else:
            chigh = float(c[idx+1])
        return (1-alpha)*clow + alpha*chigh

    if k==0:
        return funcP0
    elif k==1:
        return funcP1
    else:
        raise NotImplementedError('Boundary function: Only P0 and P1 implemented')

class TransferOperator(Operator):
    """
    An operator mapping from a P0-coefficient vector on the boundary
    to the solution on the restricted domain.
    """

    def __init__(self, nBoundaryDofs, solver, config):
        self.nBoundaryDofs = nBoundaryDofs
        self.solver = solver
        self.config = config

        self.source = NumpyVectorSpace(nBoundaryDofs)
        self.range = BlockVectorSpace((DuneIstlVectorSpace(solver.dim_range[0]),
                                       DuneIstlVectorSpace(solver.dim_range[1])))
        self.linear = True

    def apply(self, U):
        #assert isinstance(mu, pymor.parameters.base.Mu):
        assert U in self.source

        def apply_once(u):
            g = boundaryFunctionFromCoefficients(u, self.config)
            sol = self.solver.solve(g)

            return [block for block in sol]

        sols = [apply_once(u) for u in U]

        return self.range.make_array([self.range.subspaces[b].make_array(
            [sol[b] for sol in sols])
                               for b in range(len(sols[0]))])

def applyTransferOperator(solver, filename, config):
    # TODO: depends on the degree of the boundary polynomial!
    nx = config['grid']['yasp_x']
    nBoundaryDofs = 4*nx
    nSamples = 1

    transferOp = TransferOperator(nBoundaryDofs, solver, config)

    randCoeffs = transferOp.source.random(nSamples, distribution='normal')
    solutions = transferOp.apply(randCoeffs)

    if nSamples > 1:
        for i in range(nSamples):
            solver.visualize([b.vectors[i]._impl for b in solutions.blocks],
                             filename+f'_{i}')
    else:
        solver.visualize([b.vectors[0]._impl for b in solutions.blocks],
                         filename)
    #solver.visualizeOversampling(filename)
