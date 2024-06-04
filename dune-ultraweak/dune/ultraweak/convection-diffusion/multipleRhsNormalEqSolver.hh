// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ULTRAWEAK_TEST_CONV_DIFF_MULTI_RHS_NORMAL_EQ_SOLVER_HH
#define DUNE_ULTRAWEAK_TEST_CONV_DIFF_MULTI_RHS_NORMAL_EQ_SOLVER_HH

#include <dune/pdelab.hh>

#include "dune/ultraweak/convection-diffusion/normalEqSolver.hh"
#include "dune/ultraweak/convection-diffusion/preconditioner.hh"

/**
   Derived class that preassembles the system matrix to solver for multiple
   right-hand sides.
*/
template<std::size_t order, typename GV, typename Problem, typename Parametrization,
         NormalEquationType type>
class MultipleRhsConvDiffNormalEqSolver
  : public ConvDiffNormalEqSolver<order,GV,Problem,Parametrization,type> {
private:
  using BaseT = ConvDiffNormalEqSolver<order,GV,Problem,Parametrization,type>;

  using typename BaseT::GridOperator;

  using typename BaseT::FullMatrixType;
  using typename BaseT::Y;
  using typename BaseT::MatrixOperatorType;

protected:
  using Preconditioner = BlockJacobiAMG<FullMatrixType,Y,Y>;
  using Solver = Dune::CGSolver<Y>;

public:
  MultipleRhsConvDiffNormalEqSolver(const GV& gv_, const Problem& problem_,
                                    Dune::ParameterTree& pTree_,
                                    const Dune::ParameterTree& solverConfig) :
    BaseT(gv_, problem_, pTree_), solverConfig_(solverConfig) {
    // preassemble the system matrix
    std::cout << "Assembling the system matrix..." << std::endl;
    typename GridOperator::Jacobian wrappedFullMat(*gridOperator_, 0.0);
    const typename GridOperator::Domain tempEvalPoint(gfs, 0.0);
    gridOperator_->jacobian(tempEvalPoint, wrappedFullMat);

    // wrap the matrix into a linear operator
    systemMatrix_ = Dune::PDELab::Backend::native(wrappedFullMat);
    matrixOp_ = std::make_shared<MatrixOperatorType>(systemMatrix_);

    // make preconditioner
    preconditioner_ = std::make_shared<Preconditioner>(
      systemMatrix_, solverConfig_.sub("preconditioner"));

    // make solver
    solver_ = std::make_shared<Solver>(matrixOp_, preconditioner_, solverConfig_);
  }

  /**
     Solves the mixed formulation of convection-diffusion using the 'normal equations'
  */
  void solve() {
    std::cout << "Solving the "
              << ((type == NormalEquationType::classic) ? "classic" : "adjoint")
              << " normal equation..." << std::endl;

    // assemble the rhs
    const typename GridOperator::Domain tempEvalPoint(gfs, 0.0);
    typename GridOperator::Range rhs(gfs, 0.0);
    gridOperator_->residual(tempEvalPoint, rhs);
    rhs *= -1;
    auto natRhs = Dune::PDELab::Backend::native(rhs);

    /**
    // set (strong) boundary values
    typename GridOperator::Domain x0(gfs, 0.0);
    if constexpr (type == NormalEquationType::classic) {
      auto glambda = [&](const auto& el, const auto& x){
        return BaseT::problem.g(el,x);
      };
      auto gDirich = Dune::PDELab::makeGridFunctionFromCallable(BaseT::gv, glambda);
      Dune::PDELab::interpolate(gDirich, gfs, x0);
    }
    else if constexpr (type == NormalEquationType::adjoint) {
      auto glambda = [&](const auto& el, const auto& x){
        return BaseT::problem.gAdjoint(el,x);
      };
      auto gDirich = Dune::PDELab::makeGridFunctionFromCallable(BaseT::gv, glambda);
      Dune::PDELab::interpolate(gDirich, gfs, x0);
    }
    Dune::PDELab::set_nonconstrained_dofs(BaseT::cc, 0.0, x0);

    // solve
    auto solution = std::make_shared<Y>(Dune::PDELab::Backend::native(x0));
    */
    auto solution = std::make_shared<Y>(Dune::PDELab::Backend::native(
      typename GridOperator::Domain(gfs, 0.0)));
    solver_->apply(*solution, natRhs, solvingStats);
    this->coeffs.attach(solution); // attach the underlying container

    if (solvingStats.condition_estimate > 0) {
      std::cout << "Convergence rate: " << solvingStats.conv_rate;
      std::cout << ", condition (estimate): " << solvingStats.condition_estimate << std::endl;
    }
  }

  // TODO: this feels unintended...
  auto gfsSize() const {
    std::array<std::size_t,2> sz{0, 0};
    for (std::size_t i=0; i<systemMatrix_.N(); ++i)
      sz[i] = systemMatrix_[i][i].N();
    return sz;
  }

  const auto& getParametrization() {
    return parametrization_;
  }


protected:
  using BaseT::gridOperator_;
  using BaseT::gfs;
  using BaseT::solvingStats;
  using BaseT::parametrization_;

  typename BaseT::FullMatrixType systemMatrix_;
  std::shared_ptr<typename BaseT::MatrixOperatorType> matrixOp_;
  std::shared_ptr<Preconditioner> preconditioner_;
  std::shared_ptr<Solver> solver_;

  Dune::ParameterTree solverConfig_;
};

#endif  // DUNE_ULTRAWEAK_TEST_CONV_DIFF_MULTI_RHS_NORMAL_EQ_SOLVER_HH
