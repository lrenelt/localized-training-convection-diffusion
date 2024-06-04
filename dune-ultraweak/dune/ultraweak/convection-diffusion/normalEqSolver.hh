// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ULTRAWEAK_TEST_CONV_DIFF_NORMAL_EQ_SOLVER_HH
#define DUNE_ULTRAWEAK_TEST_CONV_DIFF_NORMAL_EQ_SOLVER_HH

#include <dune/pdelab.hh>

#include "dune/ultraweak/convection-diffusion/discreteGridFunctionScalarReconstruction.hh"
#include "dune/ultraweak/convection-diffusion/discreteGridFunctionVectorReconstruction.hh"
#include "dune/ultraweak/convection-diffusion/normalEqLocalOperator.hh"
#include "dune/ultraweak/convection-diffusion/preconditioner.hh"
#include "dune/ultraweak/convection-diffusion/traits.hh"
#include "dune/ultraweak/convection-diffusion/visualization.hh"

// Wrapper class for all solving aspects
// TODO: inherit from SolvingManager (current problem: tensor space structure)
template<std::size_t order, typename GV, typename Problem, typename Parametrization,
         NormalEquationType type>
class ConvDiffNormalEqSolver {
public:
  using Traits = ConvDiffNormalEqTraits<GV,order>;
private:
  static constexpr int dim = Traits::dim;
  using DF = typename Traits::DF;
  using RF = typename Traits::RF;

  // typedefs analogous to SolvingManager
  using GFS = typename Traits::TensorGFS;
  using Coeffs = typename Traits::CoefficientVector;
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;

  using VectorGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<0>>;
  using ScalarGFS = Dune::PDELab::GridFunctionSubSpace<GFS, Dune::TypeTree::StaticTreePath<1>>;

  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using LocalOperator = SymmetricConvDiffMixedOperator<
    Problem, Parametrization, typename Traits::ScalarFEM,type>;

protected:
  using GridOperator = Dune::PDELab::GridOperator<GFS, GFS, LocalOperator,
                                                  MBE, DF, RF, RF, CC, CC>;

  using FullMatrixType = Dune::PDELab::Backend::Native<typename GridOperator::Jacobian>;
  using Y = Dune::PDELab::Backend::Native<typename GridOperator::Domain>;
  using MatrixOperatorType = Dune::MatrixAdapter<FullMatrixType,Y,Y>;

public:
  ConvDiffNormalEqSolver(const GV& gv_, const Problem& problem_, Dune::ParameterTree& pTree_) :
    problem(problem_), gv(gv_),
    vectorFem(gv), vectorGfs(gv, vectorFem),
    scalarFem(gv), scalarGfs(gv, scalarFem),
    gfs(vectorGfs, scalarGfs),
    coeffs(gfs), pTree(pTree_)
  {
    vectorGfs.name("Vector-valued testsolution component");
    scalarGfs.name("Scalar-valued testsolution component");
    gfs.name("Tensorspace");

    parametrization_.initializeFromConfig(pTree);

    const int intorder = Traits::scalarOrder + 1;
    double rescaling = 1;
    if constexpr (type == NormalEquationType::classic)
      rescaling = pTree.template get<double>("rescalingBoundary");
    localOperator_ = std::make_shared<LocalOperator>(problem, parametrization_,
                                                     intorder, rescaling);

    cc.clear();
    Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem> bctype(gv,problem);
    Dune::PDELab::constraints(bctype,gfs,cc);
    gfs.update();

    MBE mbe(1<<(dim+1));
    gridOperator_ = std::make_shared<GridOperator>(gfs, cc, gfs, cc, *localOperator_, mbe);
  }

  Dune::InverseOperatorResult solvingStats;

  /**
     Solves the mixed formulation of convection-diffusion using the 'normal equations'
  */
  void solve(const Dune::ParameterTree& solverConfig) {
    std::cout << "Solving the "
              << ((type == NormalEquationType::classic) ? "classic" : "adjoint")
              << " normal equation..." << std::endl;
    // assemble the system matrix of the normal equations
    typename GridOperator::Jacobian wrappedFullMat(*gridOperator_, 0.0);
    typename GridOperator::Domain tempEvalPoint(gfs, 0.0);
    gridOperator_->jacobian(tempEvalPoint, wrappedFullMat);

    // assemble the rhs
    typename GridOperator::Range rhs(gfs, 0.0);
    gridOperator_->residual(tempEvalPoint, rhs);
    rhs *= -1;
    auto natRhs = Dune::PDELab::Backend::native(rhs);

    // wrap the matrix into a linear operator
    const FullMatrixType fullMat = Dune::PDELab::Backend::native(wrappedFullMat);
    auto op = std::make_shared<MatrixOperatorType>(fullMat);

    // create solver
    using Preconditioner = BlockJacobiAMG<FullMatrixType,Y,Y>;
    auto preconditioner = std::make_shared<Preconditioner>(
      fullMat, solverConfig.sub("preconditioner"));

    Dune::CGSolver<Y> solver(op, preconditioner, solverConfig);

    // solve
    auto solution = std::make_shared<Y>(
      Dune::PDELab::Backend::native(typename GridOperator::Domain(gfs, 0.0)));
    solver.apply(*solution, natRhs, solvingStats);
    coeffs.attach(solution); // attach the underlying container

    if (solvingStats.condition_estimate > 0) {
      std::cout << "Convergence rate: " << solvingStats.conv_rate;
      std::cout << ", condition (estimate): " << solvingStats.condition_estimate << std::endl;
    }
  }

  void writeVTK(std::string filename = "convDiffNormalEqSol") const {
    if constexpr (type == NormalEquationType::classic)
      filename += "Classic";
    else if constexpr (type == NormalEquationType::adjoint)
      filename += "Adjoint";

    const int subsampling = pTree.template get<int>("visualization.subsampling");
    visualizeConvDiffNormalEqSolution(gv, gfs, coeffs,
                                      problem, parametrization_,
                                      filename, subsampling);
  }

  // template<typename = std::enable_if_t<type==NormalEquationType::adjoint>>
  auto getDiscreteGridFunctionReconstruction() {
    return typename Traits::ScalarRecDGF<Problem,Parametrization>
      (gfs, coeffs, problem, parametrization_);
  }

  Coeffs& getCoefficientVector() {
    return coeffs;
  }

  const GFS& getGfs() const {
    return gfs;
  }

  const Parametrization& getParametrization() const {
    return parametrization_;
  }

public:
  const Problem& problem;
protected:
  const GV& gv;

  const typename Traits::VectorFEM vectorFem;
  typename Traits::VectorGFS vectorGfs;
  const typename Traits::ScalarFEM scalarFem;
  typename Traits::ScalarGFS scalarGfs;
  GFS gfs;
  CC cc;

  Coeffs coeffs;
  Dune::ParameterTree& pTree;

  Parametrization parametrization_;
  std::shared_ptr<LocalOperator> localOperator_;
  std::shared_ptr<GridOperator> gridOperator_;
};

#endif  // DUNE_ULTRAWEAK_TEST_CONV_DIFF_NORMAL_EQ_SOLVER_HH
