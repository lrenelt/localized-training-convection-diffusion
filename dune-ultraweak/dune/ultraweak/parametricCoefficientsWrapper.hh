#ifndef DUNE_ULTRAWEAK_PARAMETRIC_COEFFICIENTS_WRAPPER_HH
#define DUNE_ULTRAWEAK_PARAMETRIC_COEFFICIENTS_WRAPPER_HH

#include <dune/pdelab.hh>

enum ParametricCoefficientMode {
  ParametricMode = 0,
  AssemblyMode = 1,
  Inactive = 2
};

struct zeros {
  constexpr zeros(const std::size_t size) : size_(size) {}

  const std::size_t size() const { return size_; }

private:
  const std::size_t size_;
};

static constexpr zeros zero(1);

template<typename ParameterType_, std::size_t Q_, typename RF=double, typename SeparabilityIndex_=std::size_t>
class ParametricCoefficientsWrapper
{
public:
  // typedefs
  using ParameterType = ParameterType_;
  static constexpr std::size_t Q = Q_;
  using SeparabilityIndex = SeparabilityIndex_;

  using ParameterFunctionalType =
    std::function<const RF(const ParameterType&)>;

public:
  ParametricCoefficientsWrapper() {};

  ParametricCoefficientsWrapper(const std::array<ParameterFunctionalType,Q> thetas) : thetas_(thetas) {}

  /** set a parameter for evaluation */
  void setParameter(const ParameterType newParameter) {
    fixedParameter_ = newParameter;
    mode = ParametricCoefficientMode::ParametricMode;
  }

  /** sets the coefficient of the specified component to one  */
  void activateComponent(const SeparabilityIndex componentIndex) {
    activatedIdx_ = componentIndex;
    mode = ParametricCoefficientMode::AssemblyMode;
  }

  /** deactivate all contributions */
  void deactivate() {
    mode = ParametricCoefficientMode::Inactive;
  }

  /** evaluate the coefficient of the specified index at x */
  RF theta(const SeparabilityIndex& componentIndex) const {
    switch(mode) {
    case ParametricCoefficientMode::ParametricMode:
      return thetas_[componentIndex](fixedParameter_);
    case ParametricCoefficientMode::AssemblyMode:
      return RF(activatedIdx_ == componentIndex);
    case ParametricCoefficientMode::Inactive:
      return 0.;
    default:
      DUNE_THROW(Dune::InvalidStateException,
                 "ParametricCoefficientWrapper has invalid internal mode");
      return -1;
    }
  }

  /** make a linear combination with the current coefficients */
  template<typename XLocal>
  auto makeParametricLincomb(std::array<XLocal,Q> localValuesU) const {
    XLocal ret(0.0);
    for (std::size_t i=0; i<Q; i++)
      ret += theta(i) * localValuesU[i];
    return ret;
  }

  template<std::size_t I=0, typename ResultType = void, typename... Types>
  auto makeParametricLincomb(std::tuple<Types...> t, std::size_t thetaIdx=0) const {
    using T = typename std::tuple<Types...>;

    if constexpr (std::is_same_v<std::tuple_element_t<I,T>,zeros>) {
      // current element is 'zeros' -> skip
      if constexpr (I == std::tuple_size_v<T>-1) {
        static_assert(!std::is_same_v<ResultType, void>,
                      "Could not deduce return type. Maybe your input contained only zeros?");
        return ResultType(0.0);
      }
      else {
        thetaIdx += std::get<I>(t).size();
        return makeParametricLincomb<I+1, ResultType, Types...>(t, thetaIdx);
      }
    }
    else {
      //compute current summand
      const auto s = computeSummand(std::get<I>(t), thetaIdx);
      if constexpr (I == std::tuple_size_v<T>-1)
        return s;
      else
        return s + makeParametricLincomb<I+1, decltype(s), Types...>(t, thetaIdx);
    }
  }

  template<typename... Args>
  auto makeParametricLincomb(Args... args) const {
    return makeParametricLincomb(std::make_tuple(args...));
  }

private:
  template<typename XLocal>
  auto computeSummand(const XLocal& x, std::size_t& idx) const {
    XLocal ret = theta(idx) * x;
    ++idx;
    return ret;
  }

  // TODO: need to excude the case where the basic type is an std::array
  template<typename XLocal, std::size_t N>
  auto computeSummand(const std::array<XLocal,N>& xvec, std::size_t& idx) const {
    XLocal ret(0.0);
    for (std::size_t k=0; k<N; ++k)
      ret += computeSummand(xvec[k],idx);
    return ret;
  }

protected:
  ParameterType fixedParameter_;
  std::array<ParameterFunctionalType,Q> thetas_;
  ParametricCoefficientMode mode = ParametricCoefficientMode::ParametricMode;
  SeparabilityIndex activatedIdx_;
};

template<typename Factor>
class NormalEqParametricCoefficientWrapper {
public:
  // use an unordered set here?
  using SeparabilityIndex = std::array<typename Factor::SeparabilityIndex, 2>;

  using Left = Factor;
  using Right = Factor;
  using ParameterType = typename Factor::ParameterType;

  // number of parts
  static constexpr std::size_t Qfactor = Factor::Q;
  static constexpr std::size_t Q = Qfactor * Qfactor; // temp

private:
  SeparabilityIndex convertIdx(std::size_t linIdx) const {
    std::size_t i = linIdx % Qfactor;
    std::size_t j = (linIdx - i) / Qfactor;
    return {i,j};
  }

public:
  NormalEqParametricCoefficientWrapper(Factor left, Factor right)
    : left_(left), right_(right) {};

  void setParameter(const ParameterType& newParameter) {
    left_.setParameter(newParameter);
    right_.setParameter(newParameter);
    fixedParameter_ = newParameter;
  }

  void activateComponent(const std::size_t componentIndex) {
    auto idx = convertIdx(componentIndex);
    left_.activateComponent(idx[0]);
    right_.activateComponent(idx[1]);
  }

  void deactivate() {
    left_.deactivate();
    right_.deactivate();
  }

  auto theta(const std::size_t componentIndex) const {
    auto idx = convertIdx(componentIndex);
    return left_.theta(idx[0]) * right_.theta(idx[1]);
  }

  template<typename XLocal>
  auto makeParametricLincomb(std::array<XLocal, Qfactor> localValuesU,
                             std::array<XLocal, Qfactor> localValuesV) const {
    return left_.makeParametricLincomb(localValuesU) * right_.makeParametricLincomb(localValuesV);
  }

  template<typename XLocal>
  auto makeParametricLincombVec(std::array<XLocal, Qfactor> localValuesU,
                                std::array<XLocal, Qfactor> localValuesV) const {
    return (left_.makeParametricLincomb(localValuesU)).dot(right_.makeParametricLincomb(localValuesV));
  }

  template<typename... Types1, typename... Types2>
  auto makeParametricLincomb(std::tuple<Types1...> t1,
                             std::tuple<Types2...> t2) const {
    return left_.makeParametricLincomb(t1) * right_.makeParametricLincomb(t2);
  }

  template<typename... Types1, typename... Types2>
  auto makeParametricLincombVec(std::tuple<Types1...> t1,
                                std::tuple<Types2...> t2) const {
    return (left_.makeParametricLincomb(t1)).dot(right_.makeParametricLincomb(t2));
  }

  const Factor& left() const { return left_; }
  const Factor& right() const { return right_; }

protected:
  Factor left_;
  Factor right_;

  ParameterType fixedParameter_;

  //SeparabilityIndex activatedComponent_;
};

template<typename GO, typename Parametrization, typename MatrixType, std::size_t Qa,
         typename VectorType, std::size_t Qf>
void assembleAllParameterIndependentParts(const GO& go,
                                const Parametrization& parametrization,
                                std::array<MatrixType,Qa>& matrices,
                                std::array<VectorType,Qf>& vectors) {
  const auto paramBilinear = parametrization.bilinearPtr();
  const auto paramRhs = parametrization.rhsPtr();

  typename GO::Domain tempEvalPoint(go.trialGridFunctionSpace(), 0.0);

  paramRhs->deactivate();
  std::cout << "Assembling parameter-independent system matrices..." << std::endl;
  for (std::size_t idx=0; idx<Qa; idx++) {
    std::cout << idx+1 << " of " << Qa << "..." << std::endl;;
    paramBilinear->activateComponent(idx);
    typename GO::Jacobian mat(go, 0.0);
    go.jacobian(tempEvalPoint, mat);
    matrices[idx] = Dune::PDELab::Backend::native(mat);
  }

  paramBilinear->deactivate();
  std::cout << "Assembling parameter-independent rhs vectors..." << std::endl;
  for (std::size_t idx=0; idx<Qf; idx++) {
    std::cout << idx+1 << " of " << Qf << "..." << std::endl;;
    paramRhs->activateComponent(idx);
    typename GO::Range vec(go.testGridFunctionSpace(), 0.0);
    go.residual(tempEvalPoint, vec);
    vectors[idx] = Dune::PDELab::Backend::native(vec);
    vectors[idx] *= -1; // needed due to residual formulation
  }
}

#endif  // DUNE_ULTRAWEAK_PARAMETRIC_COEFFICIENTS_WRAPPER_HH
