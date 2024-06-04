# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from mpi4py import MPI

import ipyultraweak as uw

import convectionDiffusion.configs as configs
from convectionDiffusion.transferOperator import applyTransferOperator

def chooseSolvers(config, solver_config):
    testcase = config['testcase']
    nx = config['grid']['yasp_x']

    solvers = []
    filenames = []
    if testcase == 'basicDiffusion':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicBasicDiffusionProblem(config, solver_config)]
            filenames += [f'basicDiffusion_classicSol_n_{nx}']
        if config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointBasicDiffusionProblem(config, solver_config)]
            filenames += [f'basicDiffusion_adjointSol_n_{nx}']
    elif testcase == 'parallelChannels':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicParallelChannelProblem(config, solver_config)]
            filenames += [f'parallelChannelProblem_classicSol_n_{nx}']
        if config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointParallelChannelProblem(config, solver_config)]
            filenames += [f'parallelChannelProblem_adjointSol_n_{nx}']
    elif testcase == 'latticeChannels':
        if config['solvePrimary']:
            solvers += [uw.TransferOperatorClassicLatticeChannelProblem(config, solver_config)]
            filenames += [f'latticeChannelProblem_classicSol_n_{nx}']
        if config['solveAdjoint']:
            solvers += [uw.TransferOperatorAdjointLatticeChannelProblem(config, solver_config)]
            filenames += [f'latticeChannelProblem_adjointSol_n_{nx}']
    else:
        raise NotImplementedError(f'Unknown testcase {testcase}!')

    return solvers, filenames


if __name__ == '__main__':
    config = configs.oversampling_config
    solver_config = configs.default_solver_config

    config['solvePrimary'] = True
    config['solveAdjoint'] = False

    config['testcase'] = 'parallelChannels'
    solvers, filenames = chooseSolvers(config, solver_config)
    applyTransferOperator(solvers[0], filenames[0], config)

    config['testcase'] = 'latticeChannels'
    solvers, filenames = chooseSolvers(config, solver_config)
    applyTransferOperator(solvers[0], filenames[0], config)
