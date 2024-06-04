from pymor.operators.interface import Operator

from common.dune_istl_bindings import DuneIstlVectorSpace

class DuneOperator(Operator):
    """
    A general linear operator which can be initialized with a DUNE LinearOperator.
    """

    # TODO: verify that duneObj is valid!
    def __init__(self, duneObj):
        self.duneObj = duneObj
        self.source = DuneIstlVectorSpace(duneObj.dim_source)
        self.range = DuneIstlVectorSpace(duneObj.dim_range)
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))

        def apply_once(u, v):
            self.duneObj.apply(u._impl, v._impl)
            return v

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])


from dune.istl import CGSolver, SeqILU

class AssembledDuneOperator(Operator):
    """
    An assembled linear operator which can be initialized with a DUNE BCRSMatrix.
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.source = DuneIstlVectorSpace(matrix.cols)
        self.range = DuneIstlVectorSpace(matrix.rows)
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))

        def apply_once(u, v):
            self.matrix.mv(u._impl, v._impl)
            return v

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])

    def apply_inverse(self, V, mu=None):
        assert V in self.range
        U = self.source.zeros(len(V))

        prec = SeqILU(self.matrix)
        reduction = 1e-15
        maxIt = 500
        verbosity = 1
        solver = CGSolver(self.matrix.asLinearOperator(), prec, reduction, maxIt, verbosity)

        def apply_once(u, v):
            solver(u._impl, v._impl)
            return self.range.make_vector(u._impl)

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])
