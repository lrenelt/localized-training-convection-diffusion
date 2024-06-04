# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from mpi4py import MPI

import ipyultraweak as uw
import numpy as np
import pymor

from pymor.algorithms.rand_la import adaptive_rrf
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.numpy import NumpyMatrixOperator

from common.operator import AssembledDuneOperator

import convectionDiffusion.configs as configs
from convectionDiffusion.transferOperator import TransferOperator

# TODO: move elsewhere?
# define boundary products
def BoundaryMatrixP0(nx, gridwidth):
    return gridwidth * np.eye(4*nx)

def BoundaryMatrixP1(nx, gridwidth):
    mat = 2./3 * np.eye(4*nx)
    mat += 1./6 * np.diag(np.ones(4*nx-1), k=1)
    mat += 1./6 * np.diag(np.ones(4*nx-1), k=-1)
    mat[-1,0] = 1./6
    mat[0,-1] = 1./6
    return gridwidth * mat

def rangefinder(config, solver_config, identifier=''):
    # assemble boundary product matrices
    nGridpoints = config['grid']['yasp_x']
    k = config['oversamplingDegree']
    nBoundaryDofs = 4*(nGridpoints-1)
    if k==0:
        boundaryProduct = NumpyMatrixOperator(BoundaryMatrixP0(nBoundaryDofs, 1./(nGridpoints-1)))
    elif k==1:
        boundaryProduct = NumpyMatrixOperator(BoundaryMatrixP1(nBoundaryDofs, 1./(nGridpoints-1)))
    else:
        raise NotImplementedError('Only boundary functions in P0 and P1 implemented!')

    solvers = []
    suffixes = []
    # choose testcase
    testcase = config['testcase']
    if testcase == 'basicDiffusion':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicBasicDiffusionProblem(config, solver_config)]
            suffixes += [f'classicBasicDiffusion']
        if config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointBasicDiffusionProblem(config, solver_config)]
            suffixes += [f'adjointBasicDiffusion']
    elif testcase == 'parallelChannels':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicParallelChannelProblem(config, solver_config)]
            suffixes += [f'classicParallelChannelProblem']
        elif config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointParallelChannelProblem(config, solver_config)]
            suffixes += [f'adjointParallelChannelProblem']
    elif testcase == 'latticeChannels':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicLatticeChannelProblem(config, solver_config)]
            suffixes += [f'classicLatticeChannelProblem']
        elif config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointLatticeChannelProblem(config, solver_config)]
            suffixes += [f'adjointLatticeChannelProblem']
    else:
        raise NotImplementedError(f'Unknown testcase {testcase}!')

    for solver, suffix in zip(solvers, suffixes):
        transferOp = TransferOperator(nBoundaryDofs, solver, config)

        product = config['rangefinder']['product']
        if product == 'l2':
            range_product = BlockDiagonalOperator(
                [AssembledDuneOperator(block) for block in solver.getL2MassMatrix()])
        elif product == 'weightedL2':
            range_product = BlockDiagonalOperator(
                [AssembledDuneOperator(block) for block in solver.getWeightedL2MassMatrix()])
        elif product == 'graph':
            range_product = BlockDiagonalOperator(
                [AssembledDuneOperator(block) for block in solver.getHdivH1MassMatrix()])
        else:
            raise NotImplementedError(f'Unknown product {product}!')
        suffix += f'_{product}'

        # perform adaptive rangefinder algorithm
        cfg = config['rangefinder']
        ntest = cfg['ntest']
        basis = adaptive_rrf(transferOp,
                             num_testvecs=ntest,
                             tol=cfg['tol'],
                             range_product=range_product,
                             source_product=boundaryProduct)
        print(f'\n Generated a range approximation of dimension {len(basis)}!')

        # compute additional evaluation set
        neval = cfg['neval']
        print(f'\n Computing {neval} test vectors for evaluation...\n')
        bcs = transferOp.source.random(neval, distribution='normal')
        testvecs = transferOp.apply(bcs)
        norms = testvecs.norm(range_product)

        # evaluate
        basisSize = len(basis)
        errors = np.zeros((neval,basisSize))
        errorsU = np.zeros((neval,basisSize))
        errorsFlux = np.zeros((neval,basisSize))
        relErrors = np.zeros((neval,basisSize))
        for b in range(basisSize):
            subbasis = basis[:b+1]
            diff = (testvecs - subbasis.lincomb(subbasis.inner(testvecs, range_product).T))
            errors[:,b] = diff.norm(range_product)
            errorsFlux[:,b] = diff.blocks[0].norm(range_product.blocks[0,0])
            errorsU[:,b] = diff.blocks[1].norm(range_product.blocks[1,1])
            # TODO vectorize? Did give strange results before...
            for i in range(neval):
                relErrors[i,b] = errors[i,b] / norms[i]

        # write output
        import csv
        header = ['vecNr']
        header += [f'error_dim_{n+1}' for n in range(basisSize)]
        header += [f'rel_error_dim_{n+1}' for n in range(basisSize)]
        header += [f'scalar_error_dim_{n+1}' for n in range(basisSize)]
        header += [f'flux_error_dim_{n+1}' for n in range(basisSize)]

        filename = f'range_evaluation_n_{nGridpoints}_ntest_{ntest}_neval_{neval}_{suffix}{identifier}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i in range(neval):
                row = [i+1]
                row += [errors[i,b] for b in range(basisSize)]
                row += [relErrors[i,b] for b in range(basisSize)]
                row += [errorsU[i,b] for b in range(basisSize)]
                row += [errorsFlux[i,b] for b in range(basisSize)]
                writer.writerow(tuple(row))

        # if enabled also write visualization output
        if config['rangefinder']['visualizeEigenvectors']:
            for i in range(config['rangefinder']['nEigenvectors']):
                solver.visualize([b.vectors[i]._impl for b in basis.blocks],
                                 f'n_{nGridpoints}_ntest_{ntest}_neval_{neval}_{suffix}{identifier}_eig{i}')

        if (config['visualization']['visualize']):
            filename = 'rangefinder_basis_function'
            for idx, u in enumerate(basis):
                for i in range(len(u.blocks[0])):
                    solver.visualize([b.vectors[i]._impl for b in u.blocks],
                                     filename+f'_{idx}')

if __name__ == '__main__':
    config = configs.oversampling_config
    solver_config = configs.default_solver_config
    rangefinder(config, solver_config)
