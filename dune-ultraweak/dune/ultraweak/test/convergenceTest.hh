#ifndef DUNE_ULTRAWEAK_TEST_CONVERGENCETEST_HH
#define DUNE_ULTRAWEAK_TEST_CONVERGENCETEST_HH

#include <iostream>
#include <fstream>

#include <dune/common/parametertreeparser.hh>

#include <dune/pdelab/test/l2norm.hh>

#include "dune/ultraweak/refinementadapter.hh"

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

// TODO: do we need to pass the problem?
template<typename Traits, typename Grid>
void doTestWithRefSol(std::shared_ptr<Grid> gridp, Dune::ParameterTree pTree,
                      Dune::ParameterTree pTreeConvTest,
                      typename Traits::Problem& problem) {
  using GV = typename Grid::LevelGridView;

  static constexpr std::size_t order = Traits::order;
  static constexpr std::size_t orderdg = order+1;

  // read convergence test specific parameters
  const std::size_t refMin = pTreeConvTest.get<int>("grid.refMin");
  const std::size_t refMax = pTreeConvTest.get<int>("grid.refMax");
  std::string filename = pTreeConvTest.get<std::string>("filename");
  const bool writeVTKoutput = pTreeConvTest.get<bool>("visualization.writeVTKoutput");

  const std::size_t dgAddRef = pTreeConvTest.get<int>("evaluation.dg_add_ref");
  gridp->globalRefine(refMax+dgAddRef);

  // get finest refinement level grid view
  auto gv = gridp->levelGridView(refMax+dgAddRef);

  typename Traits::Parametrization parametrization;
  parametrization.initializeFromConfig(pTree);
  auto paramProblem = parametrization.bindToProblem(problem);

  using ReferenceSolver = typename Traits::ReferenceSolver<orderdg>;
  ReferenceSolver dgsolver(gv, paramProblem, pTree);

  // calculate DG-reference solution
  std::cout << "\n\nSolving for DG-reference solution (order=" << orderdg << ")..." << std::endl;
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
  std::string postfix;

  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(1));
  // loop over all refined grids
  for (std::size_t ref = refMin; ref<=refMax; ++ref) {
    std::cout << "\n\nRefinement stage " << ref << "/" << refMax << " ..." << std::endl;
    auto gvRef = gridp->levelGridView(ref);
    postfix = "_ref" + std::to_string(ref);

    // solve the normal equations
    typename Traits::NormalEqSolver normaleqsolver(gvRef, problem, pTree);
    normaleqsolver.solve(solverpTree);
    solvingStats.push_back(normaleqsolver.solvingStats);

    // maybe write output
    if(writeVTKoutput) {
        std::cout << "Writing solution to vtk-file..." << std::endl;
        normaleqsolver.writeVTK("normalEqSol" + postfix);
        std::cout << "Done!" << std::endl;
    }

    // adapt solution to finest grid level and compute the error
    const auto numSolReconstruction = normaleqsolver.getDiscreteGridFunctionReconstruction();
    const auto numSolRecRef = RefinementAdapter(numSolReconstruction, gv, (refMax+dgAddRef)-ref);

    const auto diff = Dune::PDELab::DifferenceAdapter(refSol, numSolRecRef);
    const std::size_t intorder = order+1;
    l2errors.push_back(l2norm(diff,intorder));

    // maybe write output
    if(writeVTKoutput) {
      std::cout << "Writing difference to vtk-file..." << std::endl;
      using DGFError = decltype(diff);
      using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGFError>;
      vtkwriter.addCellData(std::make_shared<VTKF>(diff, "Difference"));
      std::string filename = "difference_n_" + std::to_string(nx) + postfix;
      if (order == 2)
        filename += "_so";
      vtkwriter.write(filename, Dune::VTK::appendedraw);
      std::cout << "Done!" << std::endl;
    }

  }

  // write output data
  if(!std::is_same_v<Grid, Dune::YaspGrid<2>>)
    DUNE_THROW(Dune::NotImplemented, "Can not determine gridwidth, grid needs to be a YaspGrid<2>");
  const double h0 = 1. / nx;
  if (order == 1)
    filename += "_fo";
  else if (order == 2)
    filename += "_so";
  filename += "_ref" + std::to_string(refMax);
  writeToCSV(filename, l2errors, solvingStats, pTree, h0, refMin);
}

#endif  // DUNE_ULTRAWEAK_TEST_CONVERGENCETEST_HH
