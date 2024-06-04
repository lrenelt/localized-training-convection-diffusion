#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRANSFEROPERATORPROBLEM_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRANSFEROPERATORPROBLEM_HH

/**
   Adds an arbitrary function as callback for the boundary function
   of a given problem class.
 */
template<typename GV, typename RF, typename BaseProblem>
class TransferOperatorProblem : public BaseProblem {

private:
  using BaseT = BaseProblem;

protected:
  static constexpr double tol = 1e-10;
  using BaseT::pTree_;

public:
  static inline std::string name = BaseProblem::name;
  using Traits = typename BaseT::Traits;

  using FunctionType = std::function<RF(const RF, const RF)>;

  TransferOperatorProblem(Dune::ParameterTree pTree) :
    BaseT(pTree)  {
    func_ = [](const RF& x, const RF& y){ return RF(0.0); };
    adjFunc_ = [](const RF& x, const RF& y){ return RF(0.0); };
  }

  //! Boundary value of the primary variable
  template<typename Element, typename X>
  RF g(const Element& el, const X& x) const
  {
    if (!func_) {
      std::cout << "Warning: boundary function is invalid!" << std::endl;
      return 0.0;
    }
    const auto global = el.geometry().global(x);
    return func_(global[0],global[1]);
  }

  //! Boundary value of the adjoint variable
  template<typename Element, typename X>
  RF gAdjoint(const Element& el, const X& x) const
  {
    if (!adjFunc_) {
      std::cout << "Warning: boundary function is invalid!" << std::endl;
      return 0.0;
    }
    const auto global = el.geometry().global(x);
    return adjFunc_(global[0],global[1]);
  }

  void setCallback(FunctionType func) {
    std::cout << "TransferOperatorProblem: Updated the boundary function!" << std::endl;
    func_ = func;
  }

  void setAdjointCallback(FunctionType adjFunc) {
    std::cout << "TransferOperatorProblem: Updated the adjoint boundary function!" << std::endl;
    adjFunc_ = adjFunc;
  }

protected:
  FunctionType func_;
  FunctionType adjFunc_;
};

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_TRANSFEROPERATORPROBLEM_HH
