# -*- tab-width: 4; indent-tabs-mode: nil  -*-

from mpi4py import MPI
from copy import deepcopy

import convectionDiffusion.configs as configs
from convectionDiffusion.rangefinder import rangefinder

if __name__ == '__main__':
    config = configs.oversampling_config
    solver_config = configs.default_solver_config

    config['testcase'] = 'basicDiffusion'

    if config['rangefinder']['testOSwidth']:
        # evaluate different oversampling sizes
        modConfig = deepcopy(config)
        config['grid']['yasp_x'] = 120
        config['grid']['yasp_y'] = 120
        for stride, s in zip([0.25, 0.5, 1.0], ['025', '05', '1']):
            modConfig['grid']['oversampling_stride'] = stride
            modConfig['grid']['yasp_x'] = int(config['grid']['yasp_x']//3 * (1+2*stride))
            modConfig['grid']['yasp_y'] = int(config['grid']['yasp_y']//3 * (1+2*stride))
            rangefinder(modConfig, solver_config, f'_os{s}')

    if config['rangefinder']['testNumTestvecs']:
        # evaluate different number of testvectors
        modConfig = deepcopy(config)
        for ntest in [5,10,20,40]:
            modConfig['rangefinder']['ntest'] = ntest
            rangefinder(modConfig, solver_config)

    if config['rangefinder']['testNorms']:
        # evaluate different products
        modConfig = deepcopy(config)
        for prod in ['l2', 'graph']:
            modConfig['rangefinder']['product'] = prod
            rangefinder(modConfig, solver_config)

    config['grid']['yasp_x'] = 50
    config['grid']['yasp_y'] = 50

    # solve problem with channels in x-direction
    config['testcase'] = 'parallelChannels'
    rangefinder(config, solver_config)

    # solve problem with channels in x- and y-direction
    config['testcase'] = 'latticeChannels'
    rangefinder(config, solver_config)
