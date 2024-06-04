// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_DIFFUSIONDGSOLVER_HH
#define DUNE_ULTRAWEAK_DIFFUSIONDGSOLVER_HH

#include "dune/pdelab.hh"

#include "solvingManager.hh"
#include "transport/traits.hh"

// Solves stationary convection-diffusion using upwind flux for the advection
template<std::size_t order, typename GV, typename Problem, typename Traits = DGSolverTraits<GV,order>>
class DiffusionDGSolver : public SolvingManager<GV, Problem, Traits> {
private:
  using Base = SolvingManager<GV, Problem, Traits>;

public:
  DiffusionDGSolver(const GV& gv_, Problem& problem_, Dune::ParameterTree& pTree_) : Base(gv_, typename Traits::FEM(), problem_, pTree_) {
    gfs.name("dg_sol");
  }

  Dune::InverseOperatorResult solvingStats;

  void solve() {
    static const int dim = Traits::dim;
    using DF = typename Traits::DF;
    using RF = typename Traits::RF;

    // Local operator
     using LocalOperator = Dune::PDELab::ConvectionDiffusionDG<Problem, typename Traits::FEM>;
     LocalOperator localOperator(problem, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG,
                                 Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, 1.0);

    // constraints
    using GridFunctionSpace = typename Traits::GridFunctionSpace;
    using ConstraintsContainer = typename GridFunctionSpace::template ConstraintsContainer<RF>::Type;
    ConstraintsContainer constraintsContainer;
    constraintsContainer.clear();

    // Grid operator
    gfs.update();
    using std::pow;
    const int dofestimate = pow(2, dim) * gfs.maxLocalSize();
    using MatrixBackend = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    MatrixBackend matrixBackend(dofestimate);
    using GridOperator = Dune::PDELab::GridOperator<GridFunctionSpace,
                                                    GridFunctionSpace,
                                                    LocalOperator,
                                                    MatrixBackend,
                                                    DF,
                                                    RF,
                                                    RF,
                                                    ConstraintsContainer,
                                                    ConstraintsContainer>;
    GridOperator gridOperator(gfs, constraintsContainer,
                              gfs, constraintsContainer,
                              localOperator, matrixBackend);

    std::cout << "gfs with " << gfs.size() << " dofs generated  "<< std::endl;
    std::cout << "cc with " << constraintsContainer.size() << " dofs generated  "<< std::endl;

    // Solution vector
    coeffs = 0.0;

    // Solve matrix-based with PDELab preset
    const int verbosity = 2;
    const int maxiter = 5000;

    std::cout << "\nUsing sequential BCGS-solver with ILU0 preconditioner... \n" << std::endl;
    using LinearSolver = Dune::PDELab::ISTLBackend_SEQ_BCGS_ILU0;

    LinearSolver linearSolver(maxiter, verbosity);

    using CoefficientVector = typename Traits::CoefficientVector;
    using Solver = Dune::PDELab::StationaryLinearProblemSolver<GridOperator, LinearSolver, CoefficientVector>;

    const double reduction = pTree.template get<double>("reduction");
    Solver solver(gridOperator, linearSolver, coeffs, reduction);
    solver.apply();

    const auto res = linearSolver.result();
    solvingStats.converged = res.converged;
    solvingStats.iterations = res.iterations;
    solvingStats.elapsed = res.elapsed;
    solvingStats.reduction = res.reduction;
    solvingStats.conv_rate = res.conv_rate;
  }

  // Visualization
  void writeVTK(std::string filename = "diffusionSolutionDG") const {
    using VTKWriter = Dune::SubsamplingVTKWriter<GV>;
    Dune::RefinementIntervals subsampling(pTree.template get<double>("visualization.subsampling_dg"));
    VTKWriter vtkwriter(gv, subsampling);
    std::string vtkfile(filename);
    using DGF = typename Traits::DiscreteGridFunction;
    using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;
    auto dgf = this->getDiscreteGridFunction();
    vtkwriter.addVertexData(std::make_shared<VTKF>(dgf, "dg_solution"));
    vtkwriter.write(vtkfile, Dune::VTK::appendedraw);
  }

protected:
  using Base::problem;
  using Base::gv;
  using Base::gfs;
  using Base::coeffs;
  using Base::pTree;
};

#endif // DUNE_ULTRAWEAK_DIFFUSIONDGSOLVER_HH
