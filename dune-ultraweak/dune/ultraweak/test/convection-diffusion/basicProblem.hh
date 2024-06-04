// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_BASIC_PROBLEM_HH
#define DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_BASIC_PROBLEM_HH

#include <dune/pdelab.hh>

#include "dune/ultraweak/parametricCoefficientsWrapper.hh"
#include "dune/ultraweak/test/problemFunctions.hh"
#include "dune/ultraweak/test/convection-diffusion/parametrizationDecorator.hh"

/**
   Basic problem with homogeneous constant diffusion
 */
template <typename GV, typename RF_>
class  BasicDiffusionProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF_>
{
public:
  static inline std::string name = "BasicDiffusionProblem";

  using RF = RF_;
  using Base = Dune::PDELab::ConvectionDiffusionModelProblem<GV,RF>;
  using Traits = typename Base::Traits;

protected:
  static constexpr double tol = 1e-10;

public:
  using MatrixType = typename Traits::PermTensorType;

  static constexpr std::size_t nDiffusionComponents = 1;
  static constexpr std::size_t nReactionComponents = 1;

  BasicDiffusionProblem(Dune::ParameterTree pTree) :
    Base(), pTree_(pTree), identityMat_(0.0),
    useNeumannBC_(pTree.get<bool>("problem.conv-diff.useNeumannBC")),
    useAdvection_(pTree.get<bool>("problem.conv-diff.useAdvection"))
  {
    for (std::size_t i=0; i<Traits::dimDomain; i++)
      identityMat_[i][i] = 1.0;
  }

  template<typename Element, typename X>
  auto A (const Element& el, const X& x) const = delete;

  // Inverse permeability tensor (isotropic)
  template<typename Element, typename X>
  std::array<MatrixType,nDiffusionComponents>
  Ainv (const Element& el, const X& x) const
  {
    return {identityMat_};
  }

  // Boundary condition type
  template<typename Element, typename X>
  auto bctype(const Element& el, const X& x) const
  {
    if (!useNeumannBC_)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;

    const auto& global = el.geometry().global(x);
    if (global[0] < tol or global[1] < tol)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    else
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  // velocity field
  template<typename Element, typename X>
  auto b (const Element& el, const X& x) const
  {
    if (useAdvection_)
      return typename Traits::RangeType({1.0, 1.0});
    return typename Traits::RangeType({0.0, 0.0});
  }

  // reaction coefficient
  template<typename Element, typename X>
  std::array<RF,nReactionComponents> c (const Element& el, const X& x) const
  {
    return {1.0};
  }

  // Dirichlet condition
  template<typename Element, typename X>
  RF g (const Element& el, const X& x) const
  {
    const auto& global = el.geometry().global(x);
    const int nBumps = pTree_.get<int>("problem.nInflowBumps");
    if (pTree_.get<bool>("problem.discontinuousInflow"))
      return BoundaryProfiles::L2bump(global[0],nBumps)
             + BoundaryProfiles::L2bump(global[1],nBumps);
    else
      return BoundaryProfiles::sineBump(global[0],nBumps)
        + BoundaryProfiles::sineBump(global[1],nBumps);
  }

  template<typename Element, typename X>
  RF gAdjoint (const Element& el, const X& x) const
  {
    return 0.0;
  }

  // source term
  template<typename Element, typename X>
  RF f(const Element& el, const X& x) const
  {
    return 1.0;
  }

  // Neumann condition
  template<typename Intersection, typename X>
  RF j (const Intersection& is, const X& x) const
  {
    return 1.0;
  }

protected:
  Dune::ParameterTree pTree_;
private:
  typename Traits::PermTensorType identityMat_;
  const bool useNeumannBC_;
  const bool useAdvection_;
};

class BasicDiffusionParametrization {
public:
  static constexpr std::size_t nParams = 5;
  using ParameterValueType = double;
  using ParameterType = std::array<ParameterValueType, nParams>;

  static constexpr std::size_t Qa = 3;
  using ParametrizationOnesided = ParametricCoefficientsWrapper<ParameterType, Qa>;
  using ParametrizationBilinear = NormalEqParametricCoefficientWrapper<ParametrizationOnesided>;

  static constexpr std::size_t Qf = 4;
  using ParametrizationRhs = ParametricCoefficientsWrapper<ParameterType, Qf>;

  using ParameterFunctionalType =
    typename ParametrizationOnesided::ParameterFunctionalType;

  BasicDiffusionParametrization() {
    std::array<ParameterFunctionalType,Qa> thetas = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return 1./mu[0]; },
      [](const ParameterType& mu){ return mu[1]; }
    };
    ParametrizationOnesided paramLeft(thetas);
    ParametrizationOnesided paramRight(thetas);

    parametrizationBilinear_ = std::make_shared<ParametrizationBilinear>(std::move(paramLeft),
                                                                         std::move(paramRight));

    // currently no parametrization
    // TODO: make this a default?
    std::array<ParameterFunctionalType,Qf> thetasRhs = {
      [](const ParameterType& mu){ return 1.0; },
      [](const ParameterType& mu){ return mu[2]; },
      [](const ParameterType& mu){ return mu[3]; },
      [](const ParameterType& mu){ return mu[4]; }
    };

    parametrizationRhs_ = std::make_shared<ParametrizationRhs>(std::move(thetasRhs));
  }

  BasicDiffusionParametrization(BasicDiffusionParametrization& other) = delete;
  BasicDiffusionParametrization(const BasicDiffusionParametrization& other) = delete;

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
    const double mu0 = pTree.template get<double>("problem.conv-diff.fixedDiffusion");
    const double mu1 = pTree.template get<double>("problem.conv-diff.fixedReaction");
    const double mu2 = pTree.template get<double>("problem.conv-diff.fixedSource");
    const double mu3 = pTree.template get<double>("problem.conv-diff.fixedDirichletScaling");
    const double mu4 = pTree.template get<double>("problem.conv-diff.fixedNeumannScaling");
    return ParameterType({mu0, mu1, mu2, mu3, mu4});
  }

  void initializeFromConfig(const Dune::ParameterTree& pTree) const {
    this->setParameter(getParameterFromConfig(pTree));
  }

  template<typename GV, typename RF>
  auto bindToProblem(BasicDiffusionProblem<GV,RF>& problem) const {
    return ConvDiffParametrizationDecorator(problem, *this);
  }

private:
  std::shared_ptr<ParametrizationBilinear> parametrizationBilinear_;
  std::shared_ptr<ParametrizationRhs> parametrizationRhs_;
};

#endif // DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_BASIC_PROBLEM_HH
