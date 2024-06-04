#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONCONVDIFFDIFFOP_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONCONVDIFFDIFFOP_HH

#include <vector>

#include <dune/pdelab.hh>

#include "../discreteGridFunctionDivergence.hh"

// TODO: remove
using namespace Dune;
using namespace PDELab;

template<typename GFS, typename X, typename Problem, typename Parametrization,
         int subGfsIdx=1, //TODO: remove
         typename SubGFS = GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<subGfsIdx>>>
class DiscreteGridFunctionScalarReconstruction : public GridFunctionBase<
  GridFunctionTraits<typename GFS::Traits::GridViewType,
                     typename SubGFS::Traits::FiniteElementType::Traits::LocalBasisType
                       ::Traits::RangeFieldType,
                     SubGFS::Traits::FiniteElementType::Traits::LocalBasisType
                       ::Traits::dimRange,
                     typename SubGFS::Traits::FiniteElementType::Traits::LocalBasisType
                     ::Traits::RangeType>,
  DiscreteGridFunctionScalarReconstruction<GFS,X,Problem,Parametrization,subGfsIdx>>
{
private:
  using GV = typename GFS::Traits::GridViewType;
  using LBTraits = typename SubGFS::Traits::FiniteElementType::Traits::LocalBasisType::Traits;
  using DF = typename GV::Grid::ctype;
  using RF = typename LBTraits::RangeFieldType;
  using RangeType = typename LBTraits::RangeType;
;
public:
  using Traits = GridFunctionTraits<GV,RF,LBTraits::dimRange,RangeType>;

private:
  using BaseT = GridFunctionBase<Traits,
                                 DiscreteGridFunctionScalarReconstruction<
                                   GFS,X,Problem,Parametrization,subGfsIdx>>;

  using FluxGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<0>>;
  using ScalarGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<1>>;
  using FluxCoeffs = Backend::Vector<FluxGFS,DF>;
  using ScalarCoeffs = Backend::Vector<ScalarGFS,DF>;

  using DGFVector = DiscreteGridFunction<FluxGFS,X>;
  using DGFDivergence = DiscreteGridFunctionDivergence<FluxGFS,X>;
  using DGFScalar = DiscreteGridFunction<ScalarGFS,X>;
  using DGFGradient = DiscreteGridFunctionGradient<ScalarGFS,X>;

  using VectorRangeType = typename FluxGFS::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeType;
public:
  DiscreteGridFunctionScalarReconstruction(const GFS& gfs, const X& x_,
                                           const Problem& problem,
                                           const Parametrization& param) :
    fluxGfs_(gfs), scalarGfs_(gfs), problem_(problem), param_(param),
    dgfVector_(fluxGfs_, x_), dgfDivergence_(fluxGfs_, x_),
    dgfScalar_(scalarGfs_, x_), dgfGradient_(scalarGfs_, x_) {}

  inline void evaluate(const typename Traits::ElementType& e,
                       const typename Traits::DomainType& x,
                       typename Traits::RangeType& y) const
  {
    //evaluate data functions
    const auto velocity = problem_.b(e, x);
    const auto reaction = problem_.c(e, x);
    auto invDiffusivity = problem_.Ainv(e, x);

    // second (scalar-valued) component
    dgfVector_.evaluate(e,x,tau);
    dgfDivergence_.evaluate(e,x,divtau);
    dgfScalar_.evaluate(e,x,phi);

    VectorRangeType Ainvb;
    constexpr std::size_t nA = invDiffusivity.size();
    std::array<RangeType,nA> Ainvbtau;
    for (std::size_t b=0; b<nA; ++b) {
      Ainvb = 0.0;
      invDiffusivity[b].mv(velocity, Ainvb);
      Ainvbtau[b] = Ainvb.dot(tau);
    }

    constexpr std::size_t nC = reaction.size();
    std::array<RangeType,nC> cphi;
    for (std::size_t c=0; c<nC; ++c)
      cphi[c] = reaction[c]*phi;

    y = param_.bilinear().left().makeParametricLincomb(divtau, Ainvbtau, cphi);
  }

  //! get a reference to the GridView
  const typename Traits::GridViewType& getGridView() const
  { return dgfScalar_.getGridView(); }


private:
  const FluxGFS fluxGfs_;
  const ScalarGFS scalarGfs_;

  const Problem& problem_;
  const Parametrization& param_;

  mutable typename DGFVector::Traits::RangeType tau;
  mutable typename DGFScalar::Traits::RangeType divtau;
  mutable typename DGFScalar::Traits::RangeType phi;
  mutable typename DGFGradient::Traits::RangeType gradphi;
  DGFVector dgfVector_;
  DGFDivergence dgfDivergence_;
  DGFScalar dgfScalar_;
  DGFGradient dgfGradient_;
};

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_DISCRETEGRIDFUNCTIONCONVDIFFDIFFOP_HH
