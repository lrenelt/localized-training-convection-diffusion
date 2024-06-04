from dune.istl._istl import BlockVector as IstlBlockVector
import ipyultraweak as uw

from pymor.vectorarrays.list import CopyOnWriteVector

class DuneIstlVector(CopyOnWriteVector):
    """
    Wrapper class for the DUNE-internal vector type
    """

    def __init__(self, vector):
        assert isinstance(vector, IstlBlockVector), f'{vector} is not a dune-istl BlockVector!'
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    def to_numpy(self, ensure_copy=False):
        raise NotImplementedError

    def _copy_data(self):
        pass

    def _scal(self, alpha):
        self._impl *= alpha

    def _axpy(self, alpha, x):
        assert isinstance(x, DuneIstlVector)
        self._impl.axpy(alpha, x._impl)

    def inner(self, other, product=None):
        assert isinstance(other, DuneIstlVector)
        return self._impl.dot(other._impl)

    # This is the Euclidian norm on the coefficients!
    def norm(self):
        import numpy as np
        return np.sqrt(self.inner(self))

    def norm2(self):
        return self.inner(self)

    def sup_norm(self):
        raise NotImplementedError

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._impl)


from pymor.vectorarrays.list import ListVectorSpace

class DuneIstlVectorSpace(ListVectorSpace):
    """
    Wraps the DuneIstlVector into a vector space
    """

    def __init__(self, dim):
        self.dim = dim

    def zero_vector(self):
        impl = IstlBlockVector(self.dim)
        impl *= 0.0
        return DuneIstlVector(impl)

    def make_vector(self, obj):
        return DuneIstlVector(obj)

    def __eq__(self, other):
        return type(other) is DuneIstlVectorSpace and self.dim == other.dim
