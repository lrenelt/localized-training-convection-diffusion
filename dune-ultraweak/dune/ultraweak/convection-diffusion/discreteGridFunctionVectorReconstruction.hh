#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONVECTORRECONSTRUCTION_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONVECTORRECONSTRUCTION_HH

#include <vector>

#include <dune/pdelab.hh>

// TODO: remove
using namespace Dune;
using namespace PDELab;

template<typename GFS, typename X, typename Problem, typename Parametrization,
         typename SubGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<0>>>
class DiscreteGridFunctionVectorReconstruction : public GridFunctionBase<
  GridFunctionTraits<typename GFS::Traits::GridViewType,
                     typename SubGFS::Traits::FiniteElementType
                     ::Traits::LocalBasisType::Traits::RangeFieldType,
                     SubGFS::Traits::FiniteElementType
                     ::Traits::LocalBasisType::Traits::dimRange,
                     typename SubGFS::Traits::FiniteElementType
                     ::Traits::LocalBasisType::Traits::RangeType>,
  DiscreteGridFunctionVectorReconstruction<GFS,X,Problem,Parametrization>>
{
private:
  using FluxGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<0>>;
  using ScalarGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<1>>;

  using GV = typename FluxGFS::Traits::GridViewType;
  using LBTraits = typename FluxGFS::Traits::FiniteElementType::Traits::LocalBasisType::Traits;
  using DF = typename GV::Grid::ctype;
  using RF = typename LBTraits::RangeFieldType;
  using RangeType = typename LBTraits::RangeType;

  using FluxCoeffs = Backend::Vector<FluxGFS,DF>;
  using ScalarCoeffs = Backend::Vector<ScalarGFS,DF>;

public:
  using Traits = GridFunctionTraits<GV,RF,LBTraits::dimRange,RangeType>;

private:
  using BaseT = GridFunctionBase<Traits,
                                 DiscreteGridFunctionVectorReconstruction<
                                   GFS,X,Problem,Parametrization>>;

  // using DGFVector = DiscreteGridFunctionPiola<FluxGFS,X>;
  using DGFVector = DiscreteGridFunction<FluxGFS,X>;
  using DGFGradient = DiscreteGridFunctionGradient<ScalarGFS,X>;

public:
  DiscreteGridFunctionVectorReconstruction(const GFS& gfs, const X& x_,
                                           const Problem& problem,
                                           const Parametrization& param) :
    fluxGfs_(gfs), scalarGfs_(gfs), problem_(problem), param_(param),
    dgfVector_(fluxGfs_, x_), dgfGradient_(scalarGfs_, x_) {}

  inline void evaluate(const typename Traits::ElementType& e, const typename Traits::DomainType& x,
                typename Traits::RangeType& y) const
  {
    //evaluate data functions
    const auto xcenter = e.geometry().local(e.geometry().center());
    const auto invDiffusivity = problem_.Ainv(e, xcenter);

    // first (vector-valued) component
    dgfVector_.evaluate(e,x,tau);
    dgfGradient_.evaluate(e,x,gradphi);

    constexpr std::size_t nBlocks = invDiffusivity.size();
    std::array<RangeType,nBlocks> Ainvtau;
    for (std::size_t b=0; b<nBlocks; ++b)
      invDiffusivity[b].mv(tau, Ainvtau[b]);

    y = param_.bilinear().left().makeParametricLincomb(gradphi, Ainvtau, RangeType(0));
  }

  //! get a reference to the GridView
  const typename Traits::GridViewType& getGridView() const
  { return dgfVector_.getGridView(); }


private:
  const FluxGFS fluxGfs_;
  const ScalarGFS scalarGfs_;

  const Problem& problem_;
  const Parametrization& param_;

  mutable typename DGFVector::Traits::RangeType tau;
  mutable typename DGFGradient::Traits::RangeType gradphi;
  DGFVector dgfVector_;
  DGFGradient dgfGradient_;
};

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONVECTORRECONSTRUCTION_HH
