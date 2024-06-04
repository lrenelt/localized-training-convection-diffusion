// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ULTRAWEAK_TESTUTIL_HH
#define DUNE_ULTRAWEAK_TESTUTIL_HH

#include <iostream>
#include <fstream>

#include <dune/common/parametertreeparser.hh>

#include <dune/pdelab/test/l2norm.hh>

#include "upwindDGsolver.hh"
#include "darcy-velocity/catalysatorProblems.hh"
#include "darcy-velocity/darcyMixedSolver.hh"
#include "darcy-velocity/darcyProblem.hh"
#include "darcy-velocity/darcySIPGSolver.hh"
#include "darcy-velocity/darcyVelocityAdapter.hh"
#include "darcy-velocity/utility.hh"

#include "refinementadapter.hh"
#include "test/problems.hh"
#include "transport/defaultParametrization.hh"
#include "transport/discreteGridFunctionReconstruction.hh"
#include "transport/normalEqSolver.hh" // TODO: remove

/**
   Write computed data to a csv file
*/
// TODO: pass gridwidths (how to calculate?)
void writeToCSV(std::string filename, const std::vector<double>& data, const std::vector<typename Dune::InverseOperatorResult> res, Dune::ParameterTree& pTree, double h0, double refMin) {
  // open file
  std::ofstream file;
  filename += ".csv";
  file.open(filename);

  // write header
  file << "gridwidth, l2error, condition_estimate, iterations, time \n";

  // write data
  for (size_t i=0; i<data.size(); i++) {
    file << std::to_string(pow(0.5, refMin+i) * h0) << ", " << std::to_string(data[i]) << ", ";
    file << res[i].condition_estimate << ", " << res[i].iterations << ", " << res[i].elapsed <<  "\n";
  }

  // close the file
  file.close();
}


template<typename Grid>
void doTest(std::shared_ptr<Grid> gridp, Dune::ParameterTree pTree) {
  std::size_t ref_min, ref_max;
  ref_min = pTree.get<int>("grid.refMin");
  ref_max = pTree.get<int>("grid.refMax");

  gridp->globalRefine(ref_max);

  for (std::size_t ref = ref_min; ref<=ref_max; ++ref)
    solve_normal_eq(gridp->levelGridView(ref), pTree);
}

// TODO: pass reference function and numerical solve function
template<std::size_t order, typename Grid, typename Problem>
void doTestWithRefSol(std::shared_ptr<Grid> gridp, Dune::ParameterTree pTree,
                      Dune::ParameterTree pTreeConvTest,
                      Problem& problem) {
  using GV = typename Grid::LevelGridView;

  // read convergence test specific parameters
  std::size_t refMin = pTreeConvTest.get<int>("grid.refMin");
  std::size_t refMax = pTreeConvTest.get<int>("grid.refMax");
  std::string filename = pTreeConvTest.get<std::string>("filename");
  const bool writeVTKoutput = pTreeConvTest.get<bool>("visualization.writeVTKoutput");

  const std::size_t dgAddRef = pTreeConvTest.get<int>("evaluation.dg_add_ref");
  gridp->globalRefine(refMax+dgAddRef);

  // get finest refinement level grid view
  auto gv = gridp->levelGridView(refMax+dgAddRef);

  ExtendedTransportParametrization parametrization;
  parametrization.initializeFromConfig(pTree);
  ParametrizationDecorator paramProblem(problem, parametrization);

  // calculate DG-reference solution
  static const std::size_t orderdg = order+1;
  std::cout << "\n\nSolving for DG-reference solution (order=" << orderdg << ")..." << std::endl;
  using ReferenceSolver = UpwindDGSolver<orderdg,GV,decltype(paramProblem)>;
  ReferenceSolver dgsolver(gv, paramProblem, pTree);
  dgsolver.solve();
  auto refSol = dgsolver.getDiscreteGridFunction();
  std::cout << "Solving for DG-reference solution...done!" << std::endl;

  if(writeVTKoutput) {
    std::cout << "Writing DG solution to vtk-file..." << std::endl;
    dgsolver.writeVTK();
    std::cout << "Done!" << std::endl;
  }

  // initialize QoI vectors
  std::vector<double> l2errors;
  std::vector<typename Dune::InverseOperatorResult> solvingStats;

  Dune::ParameterTree solverpTree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("solver_config.ini", solverpTree);

  const int nx = pTreeConvTest.get<int>("grid.yasp_x");

  // loop over all refined grids
  using Solver = NormalEqSolver<order, GV, Problem>;
  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(1));
  for (std::size_t ref = refMin; ref<=refMax; ++ref) {
    std::cout << "\n\nRefinement stage " << ref << "/" << refMax << " ..." << std::endl;
    auto gvRef = gridp->levelGridView(ref);

    // solve the normal equations
    Solver normaleqsolver(gvRef, problem, pTree);
    normaleqsolver.solve(solverpTree);
    normaleqsolver.writeVTK("normalEqSol_ref" + std::to_string(ref));
    solvingStats.push_back(normaleqsolver.solvingStats);

    // adapt solution to finest grid level and compute the error
    const auto numTestSol = normaleqsolver.getDiscreteGridFunction();
    const auto numTestSolRef = RefinementAdapter(numTestSol, gv, (refMax+dgAddRef)-ref);
    const auto numTestSolGradient = Dune::PDELab::DiscreteGridFunctionGradient(
      normaleqsolver.getGfs(), normaleqsolver.getCoefficientVector());
    const auto numTestSolGradientRef = RefinementAdapter(numTestSolGradient, gv,
                                                         (refMax+dgAddRef)-ref);
    const auto numSolReconstruction = DiscreteGridFunctionReconstruction(
      numTestSolRef, numTestSolGradientRef, problem, parametrization);
    const auto diff = Dune::PDELab::DifferenceAdapter(refSol, numSolReconstruction);
    const std::size_t intorder = order+1;
    l2errors.push_back(l2norm(diff,intorder));

    using DGFError = decltype(diff);
    using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFError>;
    vtkwriter.addVertexData(std::make_shared<VTKF>(diff, "Difference"));
    std::string filename = "difference_n_" + std::to_string(nx) + "_ref_" + std::to_string(ref);
    if (order == 2)
      filename += "_so";
    vtkwriter.write(filename, Dune::VTK::appendedraw);
  }

  // write output data
  if(!std::is_same_v<Grid, Dune::YaspGrid<2>>)
    DUNE_THROW(Dune::NotImplemented, "Could not determine gridwidth, grid needs to be a YaspGrid<2>");
  const double h0 = 1. / nx;
  if (order == 1)
    filename += "_fo";
  else if (order == 2)
    filename += "_so";
  filename += "_ref" + std::to_string(refMax);
  writeToCSV(filename, l2errors, solvingStats, pTree, h0, refMin);
}

template<std::size_t order, typename Grid, typename Problem>
void doTestWithRefSolDarcy(std::shared_ptr<Grid> gridp, Dune::ParameterTree pTree,
                      Dune::ParameterTree pTreeConvTest,
                      Problem& baseProblem) {
  using GV = typename Grid::LevelGridView;
  using RF = double;

  // read convergence test specific parameters
  std::size_t refMin = pTreeConvTest.get<int>("grid.refMin");
  std::size_t refMax = pTreeConvTest.get<int>("grid.refMax");
  std::string filename = pTreeConvTest.get<std::string>("filename");
  const bool writeVTKoutput = pTreeConvTest.get<bool>("visualization.writeVTKoutput");

  const std::size_t dgAddRef = pTreeConvTest.get<int>("evaluation.dg_add_ref");
  gridp->globalRefine(refMax+dgAddRef);

  // get finest refinement level grid view
  auto gv = gridp->levelGridView(refMax+dgAddRef);

  /** calculation of the fine Darcy-velocity field */
  std::cout << "\n\nSolving for fine Darcy field..." << std::endl;

  // define the darcy problem
  using DarcyProblem = CatalysatorProblem<GV, RF, BoundaryConditionType::pressure>;
  DarcyProblem darcyProblem(pTree);

  // set up the solver
  static const std::size_t orderDarcy = 3;
  using DarcySolver = DarcySIPGSolver<orderDarcy,GV,DarcyProblem>;
  auto darcysolver = std::make_unique<DarcySolver>(gv, darcyProblem, pTree);

  darcysolver->solve();
  auto darcyVelocity = darcysolver->getDiscreteGridFunctionFluxPtr();
  using DGFType = typename DarcySolver::DGFFlux;

  std::cout << "Solving for fine Darcy field...done." << std::endl;

  // maybe write output
  if(writeVTKoutput) {
    std::cout << "Writing velocity field to vtk-file..." << std::endl;
    writeVelocityFieldVTK(*darcyVelocity);
    std::cout << "Done!" << std::endl;
  }

  // set up the problem with the computed velocity field
  using ProblemType = DarcyVelocityAdapter<Problem,DGFType>;
  auto problem = std::make_unique<ProblemType>(baseProblem, *darcyVelocity);
  const bool useFineVelocity = pTreeConvTest.get<bool>("evaluation.use_fine_velocity");
  problem->setRefinement(useFineVelocity);
  ExtendedTransportParametrization parametrization;
  parametrization.initializeFromConfig(pTree);
  ParametrizationDecorator paramProblem(*problem, parametrization);

  /** calculation of the fine DG-reference solution */
  static const std::size_t orderdg = order+1;
  std::cout << "\n\nSolving for DG-reference solution (order=" << orderdg << ")..." << std::endl;

  // define the solver
  using ReferenceSolver = UpwindDGSolver<orderdg,GV,decltype(paramProblem)>;
  ReferenceSolver dgsolver(gv, paramProblem, pTree);

  dgsolver.solve();

  auto refSol = dgsolver.getDiscreteGridFunction();
  std::cout << "Solving for DG-reference solution...done!" << std::endl;

  // maybe write output
  if(writeVTKoutput) {
    std::cout << "Writing DG solution to vtk-file..." << std::endl;
    dgsolver.writeVTK();
    std::cout << "Done!" << std::endl;
  }

  /** solve the transport problem */
  // initialize QoI vectors
  std::vector<double> l2errors;
  std::vector<typename Dune::InverseOperatorResult> solvingStats;

  // read solver configuration file
  Dune::ParameterTree solverpTree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("solver_config.ini", solverpTree);

  const std::size_t intorder = orderdg+1;
  std::string postfix = "";

  // pre-initialize VTK writer
  const std::size_t subsampling = pTreeConvTest.get<int>("visualization.subsampling_error");
  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(subsampling));

  // option 1: always recompute the Darcy field on the current refinement level
  // TODO: check whether we can pre-initialize more stuff. otherwise remove some pointer stuff
  // TODO: move some of the typedefs into the solver class?
  if (!useFineVelocity) {
    using CoarseDarcySolver = DarcySIPGSolver<order,GV,DarcyProblem>;
    std::unique_ptr<CoarseDarcySolver> coarseDarcySolver;

    // loop over all refined grids
    for (std::size_t ref = refMin; ref<=refMax; ++ref) {
      std::cout << "\n\nRefinement stage " << ref << "/" << refMax << " ..." << std::endl;
      auto gvRef = gridp->levelGridView(ref);
      postfix = "_ref" + std::to_string(ref) + "_coarse";

      std::cout << "\n\nRecomputing Darcy field..." << std::endl;

      // define Darcy-solver and solve
      coarseDarcySolver = std::make_unique<CoarseDarcySolver>(gvRef, darcyProblem, pTree);
      coarseDarcySolver->solve();

      // get the grid function for the flux
      auto coarseDarcyVelocity = coarseDarcySolver->getDiscreteGridFunctionFluxPtr();
      using CoarseDGFType = typename CoarseDarcySolver::DGFFlux;
      std::cout << "Solving for coarse Darcy field...done." << std::endl;

      using CoarseProblemType = DarcyVelocityAdapter<Problem,CoarseDGFType>;
      auto coarseProblem = std::make_unique<CoarseProblemType>(baseProblem, *coarseDarcyVelocity);

      // define the solver for the transport equation
      using Solver = NormalEqSolver<order,GV, CoarseProblemType>;
      auto normaleqsolver = std::make_unique<Solver>(gvRef, *coarseProblem, pTree);

      // From here on the code coincides with the else-path...
      // solve the normal equations
      normaleqsolver->solve(solverpTree);
      solvingStats.push_back(normaleqsolver->solvingStats);

      // maybe write output
      if(writeVTKoutput) {
        std::cout << "Writing solution to vtk-file..." << std::endl;
        normaleqsolver->writeVTK("normalEqSol" + postfix);
        std::cout << "Done!" << std::endl;
      }

      // make grid functions of the solution components
      const auto numTestSol = normaleqsolver->getDiscreteGridFunction();
      const auto numTestSolGradient = Dune::PDELab::DiscreteGridFunctionGradient(
        normaleqsolver->getGfs(), normaleqsolver->getCoefficientVector());

      // perform coarse reconstruction
      const auto numSolReconstruction = DiscreteGridFunctionReconstruction(
        numTestSol, numTestSolGradient, *coarseProblem, parametrization);

      // adapt to fine grid
      const auto numSolReconstructionRef = RefinementAdapter(numSolReconstruction, gv,
                                                             (refMax+dgAddRef)-ref);

      // compute the L2-error
      const auto diff = Dune::PDELab::DifferenceAdapter(refSol, numSolReconstructionRef);
      l2errors.push_back(l2norm(diff,intorder));

      // maybe write output
      if(writeVTKoutput) {
        std::cout << "Writing difference to vtk-file..." << std::endl;
        using DGFError = decltype(diff);
        using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFError>;
        vtkwriter.addCellData(std::make_shared<VTKF>(diff, "Difference"));
        vtkwriter.write("difference" + postfix, Dune::VTK::appendedraw);
        std::cout << "Done!" << std::endl;
      }
    }
  }
  else { // option 2: always use the fine velocity field
    using Solver = NormalEqSolver<order,GV, ProblemType>;
    std::unique_ptr<Solver> normaleqsolver;

    // loop over all refined grids
    for (std::size_t ref = refMin; ref<=refMax; ++ref) {
      std::cout << "\n\nRefinement stage " << ref << "/" << refMax << " ..." << std::endl;
      auto gvRef = gridp->levelGridView(ref);
      postfix = "_ref" + std::to_string(ref);

      normaleqsolver.reset(new Solver(gvRef, *problem, pTree));

      // solve the normal equations
      normaleqsolver->solve(solverpTree);
      solvingStats.push_back(normaleqsolver->solvingStats);

      // maybe write output
      if(writeVTKoutput) {
        std::cout << "Writing solution to vtk-file..." << std::endl;
        normaleqsolver->writeVTK("normalEqSol" + postfix);
        std::cout << "Done!" << std::endl;
      }

      // make grid functions of the solution components
      // TODO: can we pre-define those?
      const auto numTestSol = normaleqsolver->getDiscreteGridFunction();
      const auto numTestSolGradient = Dune::PDELab::DiscreteGridFunctionGradient(
        normaleqsolver->getGfs(), normaleqsolver->getCoefficientVector());

      // adapt solution to finest grid level and reconstruct using the fine velocity
      const auto numTestSolRef = RefinementAdapter(numTestSol, gv, (refMax+dgAddRef)-ref);
      const auto numTestSolGradientRef = RefinementAdapter(numTestSolGradient, gv,
                                                           (refMax+dgAddRef)-ref);
      const auto numSolReconstruction = DiscreteGridFunctionReconstruction(
        numTestSolRef, numTestSolGradientRef, *problem, parametrization);

      // compute the L2-error
      const auto diff = Dune::PDELab::DifferenceAdapter(refSol, numSolReconstruction);
      l2errors.push_back(l2norm(diff,intorder));

      // maybe write output
      if(writeVTKoutput) {
        std::cout << "Writing difference to vtk-file..." << std::endl;
        using DGFError = decltype(diff);
        using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFError>;
        vtkwriter.addCellData(std::make_shared<VTKF>(diff, "Difference"));
        vtkwriter.write("difference" + postfix, Dune::VTK::appendedraw);
        std::cout << "Done!" << std::endl;
      }
    }
  }

  // write output data
  if(!std::is_same_v<Grid, Dune::YaspGrid<2>>)
    DUNE_THROW(Dune::NotImplemented,
               "Could not determine gridwidth, grid needs to be a YaspGrid<2>");
  else {
    const double h0 = 1. / pTreeConvTest.get<int>("grid.yasp_x");;
    if (order == 1)
      filename += "_fo";
    else if (order == 2)
      filename += "_so";
    filename += postfix;
    writeToCSV(filename, l2errors, solvingStats, pTree, h0, refMin);
  }
}

template<std::size_t order, typename Grid, typename Problem>
void testDgSolutionDarcy(std::shared_ptr<Grid> gridp, Dune::ParameterTree pTree,
                      Dune::ParameterTree pTreeConvTest,
                      Problem& baseProblem) {
  using GV = typename Grid::LevelGridView;
  using RF = double;

  // read convergence test specific parameters
  const std::size_t refMin = pTreeConvTest.get<int>("grid.refMin");
  const std::size_t refMax = pTreeConvTest.get<int>("grid.refMax");
  std::string filename = "dg_convergence";
  const bool writeVTKoutput = pTreeConvTest.get<bool>("visualization.writeVTKoutput");

  const std::size_t dgAddRef = pTreeConvTest.get<int>("evaluation.dg_add_ref");
  gridp->globalRefine(refMax+dgAddRef);

  // get finest refinement level grid view
  auto gv = gridp->levelGridView(refMax+dgAddRef);

  // calculate Darcy-velocity field
  std::cout << "\n\nSolving for fine Darcy field..." << std::endl;

  using DarcyProblem = CatalysatorProblem<GV, RF>;
  DarcyProblem darcyProblem(pTree);

  //using DarcySolver = DarcyMixedSolver<GV,DarcyProblem>;
  static const std::size_t orderDarcy = 2;
  using DarcySolver = DarcySIPGSolver<orderDarcy,GV,DarcyProblem>;

  auto darcysolver = std::make_unique<DarcySolver>(gv, darcyProblem, pTree);
  darcysolver->solve();

  // TODO: use template magic to make this nice, depends only on the solver object/type
  ////////////////////
  const auto gradientDgf = Dune::PDELab::DiscreteGridFunctionGradient(
    darcysolver->getGfs(), darcysolver->getCoefficientVector());
  const auto& darcyProblem2 = darcysolver->getProblem();
  // TODO: allow matrix-valued grid function!
  const auto permLambda = [darcyProblem2](const auto& el, const auto& x){
    return -darcyProblem2.A(el,x)[0][0]; };
  const auto permDgf = Dune::PDELab::makeGridFunctionFromCallable(darcysolver->getGridView(), permLambda);
  using DGFType = Dune::PDELab::ProductGridFunctionAdapter<decltype(permDgf),decltype(gradientDgf)>;
  //permDgf and gradientDgf are passed as reference!
  auto darcyVelocity =  std::make_unique<DGFType>(permDgf, gradientDgf);
  ////////////////////

  std::cout << "Solving for fine Darcy field...done." << std::endl;

  if(writeVTKoutput) {
    std::cout << "Writing velocity field to vtk-file..." << std::endl;
    writeVelocityFieldVTK(*darcyVelocity);
    std::cout << "Done!" << std::endl;
  }

  using ProblemType = DarcyVelocityAdapter<Problem,DGFType>;
  ProblemType problem(baseProblem, *darcyVelocity);
  const bool useFineVelocity = pTreeConvTest.get<bool>("evaluation.use_fine_velocity");
  if (!useFineVelocity)
    DUNE_THROW(Dune::NotImplemented, "Recomputing velocities currently not implemented");
  problem.setRefinement(useFineVelocity);
  ExtendedTransportParametrization parametrization;
  parametrization.initializeFromConfig(pTree);
  using ParamProblemType = ParametrizationDecorator<ProblemType,ExtendedTransportParametrization>;
  auto paramProblem = std::make_unique<ParamProblemType>(problem, parametrization);

  // calculate DG-reference solution
  std::cout << "\n\nSolving for DG-reference solution (order=" << order << ")..." << std::endl;
  using DGSolver = UpwindDGSolver<order,GV,ParamProblemType>;
  auto referencesolver = std::make_unique<DGSolver>(gv, *paramProblem, pTree);
  referencesolver->solve();
  auto refSol = referencesolver->getDiscreteGridFunction();
  std::cout << "Solving for DG-reference solution...done!" << std::endl;

  if(writeVTKoutput) {
    std::cout << "Writing fine solution to vtk-file..." << std::endl;
    referencesolver->writeVTK();
    std::cout << "Done!" << std::endl;
  }

  // initialize QoI vectors
  std::vector<double> l2errors;
  std::vector<typename Dune::InverseOperatorResult> solvingStats;

  Dune::ParameterTree solverpTree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("solver_config.ini", solverpTree);

  const int nx = pTreeConvTest.get<int>("grid.yasp_x");

  std::unique_ptr<DGSolver> dgsolver;

  const std::size_t subsampling = pTreeConvTest.get<int>("visualization.subsampling_error");
  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(subsampling));

  // loop over all refined grids
  for (std::size_t ref = refMin; ref<=refMax; ++ref) {
    std::cout << "\n\nRefinement stage " << ref << "/" << refMax << " ..." << std::endl;
    auto gvRef = gridp->levelGridView(ref);

    dgsolver.reset(new DGSolver(gvRef, *paramProblem, pTree));

    // solve the normal equations
    dgsolver->solve();
    solvingStats.push_back(dgsolver->solvingStats);

    const auto coarseSol = dgsolver->getDiscreteGridFunction();
    const std::size_t intorder = order+1;

    // adapt solution to finest grid level and reconstruct using the fine velocity
    const auto coarseSolRef = RefinementAdapter(coarseSol, gv, (refMax+dgAddRef)-ref);
    const auto diff = Dune::PDELab::DifferenceAdapter(refSol, coarseSolRef);
    l2errors.push_back(l2norm(diff,intorder));

    if(writeVTKoutput) {
      std::cout << "Writing difference to vtk-file..." << std::endl;
      using DGFError = decltype(diff);
      using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFError>;
      vtkwriter.addCellData(std::make_shared<VTKF>(diff, "Difference"));
      vtkwriter.write("difference_n_" + std::to_string(nx) + "_ref_"
                      + std::to_string(ref), Dune::VTK::appendedraw);
      std::cout << "Done!" << std::endl;
    }
  }

  // write output data
  if(!std::is_same_v<Grid, Dune::YaspGrid<2>>)
    DUNE_THROW(Dune::NotImplemented, "Could not determine gridwidth, grid needs to be a YaspGrid<2>");
  const double h0 = 1. / nx;
  if (order == 1)
    filename += "_fo";
  else if (order == 2)
    filename += "_so";
  else
    filename += "_order" + std::to_string(order);
  filename += "_ref" + std::to_string(refMax);
  writeToCSV(filename, l2errors, solvingStats, pTree, h0, refMin);
}

#endif  // DUNE_ULTRAWEAK_TESTUTIL_HH
