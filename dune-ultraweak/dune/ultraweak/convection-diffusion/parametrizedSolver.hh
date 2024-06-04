// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PARAMETRIZED_SOLVER_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PARAMETRIZED_SOLVER_HH

#include "dune/ultraweak/parametricCoefficientsWrapper.hh"
#include <dune/pdelab.hh>

#include "dune/ultraweak/convection-diffusion/discreteGridFunctionScalarReconstruction.hh"
#include "dune/ultraweak/convection-diffusion/norm.hh"
#include "dune/ultraweak/convection-diffusion/normalEqLocalOperator.hh"
#include "dune/ultraweak/convection-diffusion/preconditioner.hh"
#include "dune/ultraweak/convection-diffusion/traits.hh"
#include "dune/ultraweak/convection-diffusion/visualization.hh"

// Wrapper class for all solving aspects
// TODO: inherit from ParametricSolvingManager (current problem: tensor space structure)
// TODO: rename / use namespaces
template<std::size_t order, typename GV, typename Problem, typename Parametrization,
         NormalEquationType type>
class ConvDiffParametrizedSolver {
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

public:
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

  using LocalOperator = SymmetricConvDiffMixedOperator<
    Problem, Parametrization, typename Traits::ScalarFEM,type>;
  using GridOperator = Dune::PDELab::GridOperator<GFS, GFS, LocalOperator,
                                                  MBE, DF, RF, RF, CC, CC>;

  // parametric typedefs
  using ParameterType = typename Parametrization::ParameterType;
  static constexpr std::size_t Qa = Parametrization::Qa;
  static constexpr std::size_t Qasquared = Qa * Qa;
  static constexpr std::size_t Qf = Parametrization::Qf;

  // container types
  using MatrixType = Dune::PDELab::Backend::Native<typename GridOperator::Jacobian>;
  using RHSVectorType = Dune::PDELab::Backend::Native<typename GridOperator::Range>;
  using Y = Dune::PDELab::Backend::Native<typename GridOperator::Domain>;

private:
  using L2MassOperatorType = Dune::PDELab::L2;
  using L2MassGO = Dune::PDELab::GridOperator<
    GFS, GFS, L2MassOperatorType, MBE, RF, RF, RF, CC, CC>;

  using H1MassOperatorType = HdivH1norm<typename Traits::ScalarFEM>;
  using H1MassGO = Dune::PDELab::GridOperator<
    GFS, GFS, H1MassOperatorType, MBE, RF, RF, RF, CC, CC>;

public:
  using L2MatrixType = Dune::PDELab::Backend::Native<typename L2MassGO::Jacobian>;
  using H1MatrixType = Dune::PDELab::Backend::Native<typename H1MassGO::Jacobian>;

  using OperatorType = Dune::MatrixAdapter<MatrixType,Y,Y>;

  // TODO: do we need to instantiate the parametrization externally?
  ConvDiffParametrizedSolver(const GV& gv, Problem& problem,
                     Parametrization& parametrization,
                     Dune::ParameterTree& pTree) :
    problem_(problem), gv_(gv),
    vectorFem_(gv_), vectorGfs_(gv_, vectorFem_),
    scalarFem_(gv_), scalarGfs_(gv_, scalarFem_),
    gfs_(vectorGfs_, scalarGfs_),
    coeffs_(gfs_), pTree_(pTree),
    parametrization_(parametrization)
  {
    vectorGfs_.name("Vector-valued testsolution component");
    scalarGfs_.name("Scalar-valued testsolution component");
    gfs_.name("Tensorspace");

    // make parameter-independent operators
    MBE mbe(1<<(dim+1)); // guess nonzeros per row
    const int intorder = Traits::scalarOrder + 1;
    const bool rescaling = pTree_.template get<RF>("rescalingBoundary");

    typename GridOperator::Domain tempEvalPoint(gfs_, 0.0);

    lop_ = std::make_shared<LocalOperator>(problem_, parametrization_, intorder, rescaling);
    go_ = std::make_shared<GridOperator>(gfs_, cc_, gfs_, cc_, *lop_, mbe);

    // assembly
    assembleAllParameterIndependentParts(
      *go_, parametrization_, matrices_, rhsVectors_);

    // TODO: move this into the python wrapper, this isnt needed here
    // wrap into an operator
    for (const auto& mat : matrices_)
      ops_.emplace_back(mat);

    // assemble L2-product on the test space
    L2MassOperatorType l2massop;
    L2MassGO l2massgo(gfs_, cc_, gfs_, cc_, l2massop, mbe);
    typename L2MassGO::Jacobian l2mat(l2massgo, 0.0);
    l2massgo.jacobian(tempEvalPoint, l2mat);
    l2matrix_ = Dune::PDELab::Backend::native(l2mat);

    // assemble Hdiv/H1-product on the test space
    H1MassOperatorType h1massop;
    H1MassGO h1massgo(gfs_, cc_, gfs_, cc_, h1massop, mbe);
    typename H1MassGO::Jacobian h1mat(h1massgo, 0.0);
    h1massgo.jacobian(tempEvalPoint, h1mat);
    h1matrix_ = Dune::PDELab::Backend::native(h1mat);
  }

  // TODO: this feels unintended...
  auto gfsSize() const {
    std::array<std::size_t,2> sz{0, 0};
    for (std::size_t i=0; i<l2matrix_.N(); ++i)
      sz[i] = l2matrix_[i][i].N();
    return sz;
  }

  Dune::InverseOperatorResult solvingStats;

  /**
     Solves linear transport using the 'normal equations'
  */
  Y solve(const ParameterType mu, Dune::ParameterTree& solverConfig) {
    parametrization_.setParameter(mu);
    // assemble parameter-dependent matrix
    MatrixType mat = matrices_[0]; // TODO: different initialization?
    mat *= 0.0;
    for (std::size_t i=0; i<Qasquared; i++)
      mat.axpy(parametrization_.bilinear().theta(i), matrices_[i]);

    /**
    // assemble parameter-dependent RHS
    RHSVectorType rhsVector = rhsVectors_[0]; // TODO: different initialization?
    rhsVector *= 0.0;
    for (std::size_t i=0; i<Qf; i++)
      rhsVector.axpy(parametrization_.rhs().theta(i), rhsVectors_[i]);
    */

    // TODO: temporary, reassemble to account for changed boundary function!
    typename GridOperator::Domain tempEvalPoint(gfs_, 0.0);
    parametrization_.bilinearPtr()->deactivate();
    typename GridOperator::Range rhs(gfs_, 0.0);
    go_->residual(tempEvalPoint, rhs);
    rhs *= -1;
    auto rhsVector = Dune::PDELab::Backend::native(rhs);

    const auto linOp = std::make_shared<OperatorType>(mat);

    // create solver
    using Preconditioner = Dune::BlockJacobiAMG<MatrixType,Y,Y>;
    auto preconditioner = std::make_shared<Preconditioner>(
                            mat, solverConfig.sub("preconditioner"));

    Dune::CGSolver<Y> solver(linOp, preconditioner, solverConfig);

    auto solution = Dune::PDELab::Backend::native(Coeffs(gfs_,0.0));
    solver.apply(solution, rhsVector, solvingStats);

    coeffs_ = Coeffs(gfs_, solution); // attach the underlying container

    if (solvingStats.condition_estimate > 0)
      std::cout << "Convergence rate: " << solvingStats.conv_rate << ", condition (estimate): " << solvingStats.condition_estimate << std::endl;

    return solution;
  }

  void visualize(Y rawCoeffs, const ParameterType mu, std::string filename = "normal_eq_sol") {
    coeffs_ = Coeffs(gfs_, rawCoeffs);
    parametrization_.setParameter(mu);
    writeVTK(filename);
  }

  void writeVTK(std::string filename) const {
    if (Traits::scalarOrder == 2)
      filename += "_so";

    const int subsampling = pTree_.get<int>("visualization.subsampling");
    visualizeConvDiffNormalEqSolution(gv_, gfs_, coeffs_,
                                      problem_, parametrization_,
                                      filename, subsampling);
  }

  const auto& getMatrices() const {
    return matrices_;
  }

  const L2MatrixType& getL2MassMatrix() const {
    return l2matrix_;
  }

  const H1MatrixType& getHdivH1MassMatrix() const {
    return h1matrix_;
  }

  const auto& getOperators() const {
    return ops_;
  }

  const auto& getRhsVectors() const {
    return rhsVectors_;
  }

  // SolvingManager compatibility functions
  Coeffs& getCoefficientVector() {
    return coeffs_;
  }

  const Coeffs& getCoefficientVector() const {
    return coeffs_;
  }

  const GV& getGridView() const {
    return gv_;
  }

  const GFS& getGfs() const {
    return gfs_;
  }

  const Problem& getProblem() const {
    return problem_;
  }

protected:
  const Problem& problem_;
  const GV& gv_;
  CC cc_;

  const typename Traits::VectorFEM vectorFem_;
  typename Traits::VectorGFS vectorGfs_;
  const typename Traits::ScalarFEM scalarFem_;
  typename Traits::ScalarGFS scalarGfs_;
  GFS gfs_;

  Coeffs coeffs_;
  Dune::ParameterTree& pTree_;
  Parametrization& parametrization_;

  L2MatrixType l2matrix_;
  H1MatrixType h1matrix_;
  std::array<MatrixType,Qasquared> matrices_;
  std::vector<OperatorType> ops_;
  std::array<RHSVectorType,Qf> rhsVectors_;
  std::shared_ptr<LocalOperator> lop_;
  std::shared_ptr<GridOperator> go_;
};

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PARAMETRIZED_SOLVER_HH
