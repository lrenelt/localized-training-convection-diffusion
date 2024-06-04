oversampling_config = {
    'rescalingBoundary': 1e3,
    'oversamplingDegree': 1,
    'solvePrimary': True,
    'solveAdjoint': False,
    'grid': {
        'dim': 2,
        'yasp_x':  50,
        'yasp_y':  50,
        'lx': 1.0,
        'ly': 1.0,
        'oversampling_stride': 1.0
    },
#    'testcase': 'basicDiffusion',
    'testcase': 'parallelChannels',
#    'testcase': 'latticeChannels',
    'problem': {
        'discontinuousInflow': False,
        'nInflowBumps':  1,
        'conv-diff': {
            'useNeumannBC': False,
            'useAdvection': False,
            'fixedDiffusion': 1.0,
            'fixedReaction': 0.0,
            'fixedSource': 0.0,
            'fixedDirichletScaling': 1.0,
            'fixedNeumannScaling': 0.0
        },
        'channels': {
            'nChannels': 2,
            'channelWidth': 0.05,
            'useNeumannBC': False,
            'useAdvection': True,
            'fixedDiffusionHigh': 1e2,
            'fixedDiffusionLow': 1,
            'fixedReaction': 1,
            'fixedSource': 0,
            'fixedDirichletScaling': 1,
            'fixedNeumannScaling': 0
        }
    },
    'rangefinder': {
        'ntest': 40,
        'tol': 1e-2,
        'neval': 20,
        'product': 'weightedL2',
        'testOSwidth': True,
        'testNumTestvecs': False,
        'testNorms': False,
        'visualizeEigenvectors': False,
        'nEigenvectors': 3,
    },
    'visualization': {
        'visualize': False,
        'subsampling': 8
    }
}

default_solver_config = {
    'type': 'cgsolver',
    'verbose': 1,
    'maxit': 5000,
    'reduction': 1e-15,
    'preconditioner': {
        'type': 'jac',
        'iterations': 3,
        'relaxation': 1,
        'maxLevel': 15,
        'coarsenTarget': 2000,
        'criterionSymmetric': True,
        'smoother': 'sor',
        'smootherIterations': 1,
        'smootherRelaxation': 1
    }
}
