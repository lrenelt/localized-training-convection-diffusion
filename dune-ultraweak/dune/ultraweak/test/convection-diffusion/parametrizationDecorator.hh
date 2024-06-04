#ifndef DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_PARAMETRIZATION_DECORATOR_HH
#define DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_PARAMETRIZATION_DECORATOR_HH

/**
   Decorator to attach a given parametrization to a base problem. Allows usage in standard
   PDELab-operators for a fixed parameter.
 */
template<typename BaseProblem, typename Parametrization>
class ConvDiffParametrizationDecorator : public BaseProblem
{
public:
  using Traits = typename BaseProblem::Traits;
  using RF = typename BaseProblem::RF;

  using ParameterType = typename Parametrization::ParameterType;

  ConvDiffParametrizationDecorator(BaseProblem& baseProblem,
                                   const Parametrization& parametrization)
    : BaseProblem(baseProblem), baseProblem_(baseProblem),
      parametrization_(parametrization) {}

  static constexpr std::size_t nDiffusionComponents = BaseProblem::nDiffusionComponents;
  static constexpr std::size_t nReactionComponents = BaseProblem::nReactionComponents;
  //static constexpr std::size_t nSourceComponents = BaseProblem::nSourceComponents;
  //static constexpr std::size_t nDirichletComponents = BaseProblem::nDirichletComponents;
  //static constexpr std::size_t nNeumannComponents = BaseProblem::nNeumannComponents;

  template<typename Element, typename X>
  auto Ainv (const Element& el, const X& x) const
  {
    const auto Ainv = baseProblem_.Ainv(el,x);
    if (Ainv.size() != nDiffusionComponents)
      DUNE_THROW(Dune::InvalidStateException,
                 "Problem class returned wrong number of matrices ("
                 << std::to_string(Ainv.size()) << ", should be "
                 << std::to_string(nDiffusionComponents) << ")!");

    auto ret = parametrization_.bilinear().left().theta(1) * Ainv[0];
    for (std::size_t i=1; i<nDiffusionComponents; ++i)
      ret += parametrization_.bilinear().left().theta(1+i) * Ainv[i];
    return ret;
  }

  template<typename Element, typename X>
  auto A (const Element& el, const X& x) const
  {
    auto ret = Ainv(el,x);
    ret.invert();
    return ret;
  }

  template<typename Element, typename X>
  RF c (const Element& el, const X& x) const
  {
    const auto reaction = baseProblem_.c(el,x);
    RF ret = 0.0;
    for (std::size_t i=0; i<nReactionComponents; ++i)
      ret += parametrization_.bilinear().left().theta(1+nDiffusionComponents+i) * reaction[i];
    return ret;
  }

  template<typename Element, typename X>
  RF f (const Element& el, const X& x) const
  {
    return parametrization_.rhs().theta(1) * baseProblem_.f(el,x);
  }

  template<typename Element, typename X>
  RF g (const Element& el, const X& x) const
  {
    return parametrization_.rhs().theta(2) * baseProblem_.g(el,x);
  }

  template<typename Intersection, typename X>
  RF j (const Intersection& is, const X& x) const
  {
    return parametrization_.rhs().theta(3) * baseProblem_.j(is,x);
  }


private:
  const BaseProblem baseProblem_;
  const Parametrization& parametrization_;
};

#endif  // DUNE_ULTRAWEAK_TEST_CONVECTION_DIFFUSION_PARAMETRIZATION_DECORATOR_HH
