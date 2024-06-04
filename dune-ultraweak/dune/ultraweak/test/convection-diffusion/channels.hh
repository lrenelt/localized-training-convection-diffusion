#ifndef DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_CHANNELS_HH
#define DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_CHANNELS_HH

#include <cmath>

#include <dune/pdelab.hh>

#include "dune/ultraweak/parametricCoefficientsWrapper.hh"
#include "dune/ultraweak/test/problemFunctions.hh"
#include "dune/ultraweak/test/convection-diffusion/parametrizationDecorator.hh"

/**
   Base class for convection-diffusion problems with
   two separate diffusivity regions of different magnitude.
*/
template <typename GV, typename RF_>
class ChannelProblemBase
  : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF_>
{
public:
  using RF = RF_;
  using Base = Dune::PDELab::ConvectionDiffusionModelProblem<GV,RF>;
  using Traits = typename Base::Traits;

  static constexpr double tol = 1e-10;
  using MatrixType = typename Traits::PermTensorType;
  using XGlobal = Dune::FieldVector<RF,GV::dimension>;

  //TODO: move to interface
  static constexpr std::size_t nDiffusionComponents = 2;
  static constexpr std::size_t nReactionComponents = 1;

  ChannelProblemBase() = delete;

  ChannelProblemBase(Dune::ParameterTree pTree) :
    Base(), pTree_(pTree),
    channelWidth_(pTree_.get<double>("problem.channels.channelWidth")),
    useAdvection_(pTree.get<bool>("problem.channels.useAdvection")),
    identityMat_(0.0)
  {
    for (std::size_t i=0; i<Traits::dimDomain; i++)
      identityMat_[i][i] = 1.0;
  }

  template<typename Element, typename X>
  auto A (const Element& el, const X& x) const = delete;

protected:
  virtual bool isHighConductivity(const XGlobal& xglobal) const = 0;

public:
  // Inverse permeability tensor (isotropic)
  template<typename Element, typename X>
  std::array<MatrixType,nDiffusionComponents> Ainv (const Element& el, const X& x) const
  {
    const auto& global = el.geometry().center();
    if (isHighConductivity(global))
      return { identityMat_, MatrixType(0.0) };
    else
      return { MatrixType(0.0), identityMat_ };
  }

  // velocity field
  template<typename Element, typename X>
  auto b (const Element& el, const X& x) const
  {
    if (useAdvection_)
      return typename Traits::RangeType({1.0, 1.0});
    else
      return typename Traits::RangeType({0, 0});
  }

  // reaction coefficient
  template<typename Element, typename X>
  std::array<RF,nReactionComponents> c (const Element& el, const X& x) const
  {
    return {1.0 * !isHighConductivity(el.geometry().global(x))};
  }

  // source term
  template<typename Element, typename X>
  RF f(const Element& el, const X& x) const
  {
    return 1.0 * !isHighConductivity(el.geometry().global(x));
  }

protected:
  Dune::ParameterTree pTree_;
  const double channelWidth_;
  const bool useAdvection_;
private:
  typename Traits::PermTensorType identityMat_;
};

/**
   High-conductivity channels in x-direction
*/
template <typename GV, typename RF_>
class ParallelChannelProblem : public ChannelProblemBase<GV,RF_>
{
public:
  static inline std::string name = "ParallelChannelProblem";
  using RF = RF_;
  using Base = ChannelProblemBase<GV,RF>;
  using Traits = typename Base::Traits;

  ParallelChannelProblem(Dune::ParameterTree pTree) :
    Base(pTree),
    nChannels_(pTree.get<double>("problem.channels.nChannels")),
    nInflowBumps_(pTree.get<int>("problem.nInflowBumps")),
    useDiscInflow_(pTree.get<bool>("problem.discontinuousInflow"))
  {}

protected:
  bool isHighConductivity(const typename Base::XGlobal& xglobal) const override {
    using std::abs;
    for (std::size_t i=0; i<nChannels_; ++i)
      if (abs(xglobal[1] - floor(xglobal[1]) - (i+1.)/(nChannels_+1)) < channelWidth_)
          return true;
    return false;
  }

public:
  // Boundary condition type
  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }

  // Dirichlet condition
  template<typename Element, typename X>
  RF g (const Element& el, const X& x) const
  {
    const auto& global = el.geometry().global(x);
    if (useDiscInflow_)
      return BoundaryProfiles::L2bump(global[0],nInflowBumps_)
             + BoundaryProfiles::L2bump(global[1],nInflowBumps_);
    else
      return BoundaryProfiles::sineBump(global[0],nInflowBumps_)
             + BoundaryProfiles::sineBump(global[1],nInflowBumps_);
  }

protected:
  using Base::channelWidth_;
  const std::size_t nChannels_;
  const std::size_t nInflowBumps_;
  const bool useDiscInflow_;
};

// TODO: merge with above?
/**
   High-conductivity channels in x- and y-direction.
 */
template <typename GV, typename RF_>
class LatticeChannelProblem : public ParallelChannelProblem<GV,RF_>
{
public:
  static inline std::string name = "LatticeChannelProblem";
  using Base = ParallelChannelProblem<GV,RF_>;
  using Traits = typename Base::Traits;

  LatticeChannelProblem(Dune::ParameterTree pTree) : Base(pTree) {}

protected:
  bool isHighConductivity(const typename Base::XGlobal& xglobal) const override {
    using std::abs;
    for (std::size_t i=0; i<Base::nChannels_; ++i) {
      // vertical channels
      if (abs(xglobal[0] - floor(xglobal[0]) - (i+1.)/(Base::nChannels_+1)) < Base::channelWidth_)
          return true;
      // horizontal channels
      if (abs(xglobal[1] - floor(xglobal[1]) - (i+1.)/(Base::nChannels_+1)) < Base::channelWidth_)
          return true;
    }
    return false;
  }
};

/**
   Problem similar to the one used in the dissertation of Andreas Buhr

template <typename GV, typename RF_>
class HighConductivityChannelProblem : public ChannelProblemBase<GV, RF_>
{
public:
  using RF = RF_;
  using Base = ChannelProblemBase<GV,RF_>;
  using Traits = typename Base::Traits;

  HighConductivityChannelProblem(Dune::ParameterTree pTree) :
    Base(pTree),
    nChannels_(pTree.get<double>("problem.channels.nChannels")),
    useNeumann_(pTree.get<bool>("problem.channels.useNeumannBC"))
  {}

protected:
  bool isHighConductivity(const typename Base::XGlobal& xglobal) const override {
    if (xglobal[0] < 0.1 or xglobal[0] > 0.9)
      return true;
    using std::abs;
    for (std::size_t i=0; i<nChannels_; ++i)
      if (abs(xglobal[1] - (i+1)/(nChannels_+1)) < channelWidth_ and xglobal[0] < 0.8)
        return true;
    return false;
  }

public:
  // Boundary condition type
  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    const auto& global = el.geometry().center();
    if (global[1] < tol or global[1] > 1.0-tol)
        return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;

    if ( and global[0] < tol)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
    else
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }

  // Dirichlet condition
  template<typename Element, typename X>
  RF g (const Element& el, const X& x) const
  {
    const auto& global = el.geometry().global(x);
    return (global[0] < tol) ? 1.0 : 0.0;
  }

  // Neumann condition
  template<typename Intersection, typename X>
  RF j (const Intersection& is, const X& x) const
  {
    const auto& global = is.geometry().global(x);
    if (global[1] < tol or global[1] > 1.0-tol)
      return 0.0;
    return 1.0;
  }

protected:
  using Base::channelWidth_;
  const double nChannels_;
};
*/

/**
   Parametrization for a ChannelProblem.
 */
class ChannelParametrization {
public:
  static constexpr std::size_t nParams = 6;
  using ParameterValueType = double;
  using ParameterType = std::array<ParameterValueType, nParams>;

  static constexpr std::size_t Qa = 4;
  using ParametrizationOnesided = ParametricCoefficientsWrapper<ParameterType, Qa>;
  using ParametrizationBilinear = NormalEqParametricCoefficientWrapper<ParametrizationOnesided>;

  static constexpr std::size_t Qf = 4;
  using ParametrizationRhs = ParametricCoefficientsWrapper<ParameterType, Qf>;

  using ParameterFunctionalType =
    typename ParametrizationOnesided::ParameterFunctionalType;

  ChannelParametrization() {
    std::array<ParameterFunctionalType,Qa> thetas = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return 1./mu[0]; },
      [](const ParameterType& mu){ return 1./mu[1]; },
      [](const ParameterType& mu){ return mu[2]; }};

    ParametrizationOnesided paramLeft(thetas);
    ParametrizationOnesided paramRight(thetas);

    parametrizationBilinear_ = std::make_shared<ParametrizationBilinear>(std::move(paramLeft),
                                                                         std::move(paramRight));

    std::array<ParameterFunctionalType,Qf> thetasRhs = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[nParams-3]; },
      [](const ParameterType& mu){ return mu[nParams-2]; },
      [](const ParameterType& mu){ return mu[nParams-1]; }
    };

    parametrizationRhs_ = std::make_shared<ParametrizationRhs>(std::move(thetasRhs));
  }

  ChannelParametrization(ChannelParametrization& other) = delete;
  ChannelParametrization(const ChannelParametrization& other) = delete;

  const ParametrizationBilinear& bilinear() const {
    return *parametrizationBilinear_;
  }

  std::shared_ptr<ParametrizationBilinear> bilinearPtr() const {
    return parametrizationBilinear_;
  }

  const ParametrizationRhs& rhs() const {
    return *parametrizationRhs_;
  }

  std::shared_ptr<ParametrizationRhs> rhsPtr() const {
    return parametrizationRhs_;
  }

  // convenience function
  void setParameter(const ParameterType& mu) const {
    parametrizationBilinear_->setParameter(mu);
    parametrizationRhs_->setParameter(mu);
  }

  ParameterType getParameterFromConfig(const Dune::ParameterTree& pTree) const {
    const ParameterValueType mu1 =
      pTree.template get<ParameterValueType>("problem.channels.fixedDiffusionHigh");
    const ParameterValueType mu2 =
      pTree.template get<ParameterValueType>("problem.channels.fixedDiffusionLow");
    const ParameterValueType mu3 =
      pTree.template get<ParameterValueType>("problem.channels.fixedReaction");

    const ParameterValueType mu4 =
      pTree.template get<ParameterValueType>("problem.channels.fixedSource");
    const ParameterValueType mu5 =
      pTree.template get<ParameterValueType>("problem.channels.fixedDirichletScaling");
    const ParameterValueType mu6 =
      pTree.template get<ParameterValueType>("problem.channels.fixedNeumannScaling");
    return { mu1 ,mu2, mu3, mu4, mu5, mu6 };
  }

  void initializeFromConfig(const Dune::ParameterTree& pTree) const {
    this->setParameter(getParameterFromConfig(pTree));
  }

  // TODO: ensure Problem is derived from ChannelProblemBase<GV,RF>
  template<typename Problem>
  auto bindToProblem(Problem& problem) const {
    return ConvDiffParametrizationDecorator(problem, *this);
  }

  template<typename Problem>
  using ParametrizedProblemType = ConvDiffParametrizationDecorator<
    Problem,ChannelParametrization>;

private:
  std::shared_ptr<ParametrizationBilinear> parametrizationBilinear_;
  std::shared_ptr<ParametrizationRhs> parametrizationRhs_;
};

#endif  // DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_CHANNELS_HH
