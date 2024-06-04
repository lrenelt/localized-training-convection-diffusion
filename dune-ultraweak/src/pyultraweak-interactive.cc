#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>
#include <dune/python/pybind11/numpy.h>

#include "dune/ultraweak/test/convection-diffusion/basicProblem.hh"
#include "dune/ultraweak/test/convection-diffusion/channels.hh"
#include "dune/ultraweak/test/convection-diffusion/transferOperatorProblem.hh"

#include "dune/ultraweak/py/convDiffSolver.hh"
#include "dune/ultraweak/py/transferOperator.hh"

namespace py = pybind11;

template<typename Problem, NormalEquationType type>
auto generateTypeName(std::string baseName) {
  if constexpr (type == NormalEquationType::classic)
    baseName += "Classic";
  else if constexpr (type == NormalEquationType::adjoint)
    baseName += "Adjoint";
  baseName += Problem::name;
  return baseName;
}

template<typename RF, int dim, NormalEquationType type>
void bindTransferOperators(py::module m) {
  using ctype = double;
  using Grid =  Dune::YaspGrid<dim,Dune::EquidistantOffsetCoordinates<ctype,dim>>;
  using GV = typename Grid::LeafGridView;

  {
    using BaseProblem = BasicDiffusionProblem<GV,RF>;
    using Problem = TransferOperatorProblem<GV,RF,BaseProblem>;
    using Parametrization = BasicDiffusionParametrization;

    using T = Dune::Ultraweak::Python::TransferOperator<
      1,RF,dim,Problem,Parametrization,type>;
    registerTransferOperator<T>(m, generateTypeName<Problem,type>("TransferOperator"));
  }
  {
    using BaseProblem = ParallelChannelProblem<GV,RF>;
    using Problem = TransferOperatorProblem<GV,RF,BaseProblem>;
    using Parametrization = ChannelParametrization;

    using T = Dune::Ultraweak::Python::TransferOperator<
      1,RF,dim,Problem,Parametrization,type>;
    registerTransferOperator<T>(m, generateTypeName<Problem,type>("TransferOperator"));
  }
  {
    using BaseProblem = LatticeChannelProblem<GV,RF>;
    using Problem = TransferOperatorProblem<GV,RF,BaseProblem>;
    using Parametrization = ChannelParametrization;

    using T = Dune::Ultraweak::Python::TransferOperator<
      1,RF,dim,Problem,Parametrization,type>;
    registerTransferOperator<T>(m, generateTypeName<Problem,type>("TransferOperator"));
  }
}

PYBIND11_MODULE(ipyultraweak, m)
{
  m.doc() = "pybind11 dune-ultraweak plugin";

  constexpr int dim = 2;
  using RF = double;

  bindTransferOperators<RF,dim,NormalEquationType::classic>(m);
  bindTransferOperators<RF,dim,NormalEquationType::adjoint>(m);
}
