#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRAITS_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRAITS_HH

#include <dune/pdelab.hh>

#include "dune/ultraweak/convection-diffusion/discreteGridFunctionVectorReconstruction.hh"
#include "dune/ultraweak/convection-diffusion/discreteGridFunctionScalarReconstruction.hh"

template<typename GV, std::size_t order>
struct ConvDiffNormalEqTraits {
  static const int dim = GV::dimension;
  using DF = typename GV::Grid::ctype;
  using RF = double; //Dune::Float128;

  using VBE = Dune::PDELab::ISTL::VectorBackend<>;

  // velocity space typedefs
  static const int vectorOrder = order-1;
  using VectorFEM = Dune::PDELab::RaviartThomasLocalFiniteElementMap<GV,DF,RF,vectorOrder,
    Dune::GeometryType::BasicType::cube>;
  using VectorCON = Dune::PDELab::NoConstraints;
  using VectorGFS = Dune::PDELab::GridFunctionSpace<GV,VectorFEM,VectorCON,VBE>;

  // pressure space typedefs
  static const int scalarOrder = order;
  using ScalarFEM = Dune::PDELab::QkLocalFiniteElementMap<GV,DF,RF,scalarOrder>;
  using ScalarCON = Dune::PDELab::NoConstraints;
  using ScalarGFS = Dune::PDELab::GridFunctionSpace<GV,ScalarFEM,ScalarCON,VBE>;

  //tensor space typedefs
  using TensorVBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::bcrs>;
  using TensorGFS = Dune::PDELab::CompositeGridFunctionSpace<TensorVBE,
    Dune::PDELab::LexicographicOrderingTag, VectorGFS, ScalarGFS>;
  using CoefficientVector = Dune::PDELab::Backend::Vector<TensorGFS,DF>;

  // visualization typedefs
  using VectorSubGFS = Dune::PDELab::GridFunctionSubSpace<TensorGFS,
                                                          Dune::TypeTree::StaticTreePath<0>>;
  using ScalarSubGFS = Dune::PDELab::GridFunctionSubSpace<TensorGFS,
                                                          Dune::TypeTree::StaticTreePath<1>>;

  using VectorDGF = Dune::PDELab::DiscreteGridFunctionPiola<VectorSubGFS,CoefficientVector>;
  using VectorVTKF = Dune::PDELab::VTKGridFunctionAdapter<VectorDGF>;

  using ScalarDGF = Dune::PDELab::DiscreteGridFunction<ScalarSubGFS,CoefficientVector>;
  using ScalarVTKF = Dune::PDELab::VTKGridFunctionAdapter<ScalarDGF>;

  template<typename Problem, typename Parametrization>
  using VectorRecDGF = DiscreteGridFunctionVectorReconstruction<
    TensorGFS,CoefficientVector,Problem,Parametrization>;
  template<typename Problem, typename Parametrization>
  using VectorRecVTKF = Dune::PDELab::VTKGridFunctionAdapter<VectorRecDGF<Problem,Parametrization>>;

  template<typename Problem, typename Parametrization>
  using ScalarRecDGF = DiscreteGridFunctionScalarReconstruction<
    TensorGFS,CoefficientVector,Problem,Parametrization>;
  template<typename Problem, typename Parametrization>
  using ScalarRecVTKF = Dune::PDELab::VTKGridFunctionAdapter<ScalarRecDGF<Problem,Parametrization>>;
};

template<typename GV, int order>
struct ConvDiffNormalEqTraitsEqualOrder {
  static const int dim = GV::dimension;
  using DF = typename GV::Grid::ctype;
  using RF = double; //Dune::Float128;

  using VBE = Dune::PDELab::ISTL::VectorBackend<>;

  // velocity space typedefs
  static const int componentOrder = 0;
  using ComponentFEM = Dune::PDELab::RaviartThomasLocalFiniteElementMap<GV,DF,RF,componentOrder>;
  using ComponentCON = Dune::PDELab::NoConstraints;
  using ComponentGFS = Dune::PDELab::GridFunctionSpace<GV,ComponentFEM,ComponentCON,VBE>;

  using VectorGFS = Dune::PDELab::PowerGridFunctionSpace<ComponentGFS,dim,VBE>;
  using VectorCoefficients = Dune::PDELab::Backend::Vector<VectorGFS, DF>;
  using VectorDGF = Dune::PDELab::DiscreteGridFunctionPiola<VectorGFS, VectorCoefficients>;

  // pressure space typedefs
  static const int scalarOrder = 1;
  using ScalarFEM = Dune::PDELab::QkLocalFiniteElementMap<GV,DF,RF,scalarOrder>;
  using ScalarCON = Dune::PDELab::NoConstraints;
  using ScalarGFS = Dune::PDELab::GridFunctionSpace<GV,ScalarFEM,ScalarCON,VBE>;
  using ScalarCoefficients = Dune::PDELab::Backend::Vector<ScalarGFS, RF>;
  using ScalarDGF = Dune::PDELab::DiscreteGridFunction<ScalarGFS, ScalarCoefficients>;

  //tensor space typedefs
  using TensorGFS = Dune::PDELab::CompositeGridFunctionSpace<VBE,
    Dune::PDELab::LexicographicOrderingTag, VectorGFS, ScalarGFS>;
  using CoefficientVector = Dune::PDELab::Backend::Vector<TensorGFS,DF>;
};


#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRAITS_HH
