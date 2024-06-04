// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_SOLVING_MANAGER_HH
#define DUNE_ULTRAWEAK_SOLVING_MANAGER_HH

template<typename GV, typename Problem, typename Traits>
class SolvingManager {
private:
  using FEM = typename Traits::FEM;
  using GFS = typename Traits::GridFunctionSpace;
  using Coeffs = typename Traits::CoefficientVector;

public:
  using CC = typename GFS::template ConstraintsContainer<typename Traits::RF>::Type;
  using DiscreteGridFunction = typename Traits::DiscreteGridFunction;

  SolvingManager(const GV gv_, FEM fem_, Problem& problem_, Dune::ParameterTree& pTree_) : problem(problem_), gv(gv_), fem(fem_), gfs(gv, fem), coeffs(gfs), pTree(pTree_) {}

  // needs to implemented by derived classes
  template<typename... Args>
  auto solve(Args... args) {
    DUNE_THROW(Dune::NotImplemented, "SolvingManager-interface: solve() is not implemented");
  };

  // TODO: is a default implementation useful?
  virtual void writeVTK(std::string filename) const = 0;

  void setParameterTree(Dune::ParameterTree pTree_) {
    pTree = pTree_;
  }

  std::unique_ptr<DiscreteGridFunction> getDiscreteGridFunctionPtr() const {
    return std::make_unique<DiscreteGridFunction>(gfs, coeffs);
  }

  DiscreteGridFunction getDiscreteGridFunction() const {
    return DiscreteGridFunction(gfs, coeffs);
  }

  Coeffs& getCoefficientVector() {
    return coeffs;
  }

  const Coeffs& getCoefficientVector() const {
    return coeffs;
  }

  const GV& getGridView() const {
    return gv;
  }

  const GFS& getGfs() const {
    return gfs;
  }

  const Problem& getProblem() const {
    return problem;
  }

protected:
  Problem& problem;
protected:
  const GV gv;
  const FEM fem;
  GFS gfs;
  CC cc;
  Coeffs coeffs;
  Dune::ParameterTree& pTree;
};

#endif // DUNE_ULTRAWEAK_SOLVING_MANAGER_HH
