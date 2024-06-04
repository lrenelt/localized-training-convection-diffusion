#ifndef DUNE_ULTRAWEAK_PY_OPERATOR_HH
#define DUNE_ULTRAWEAK_PY_OPERATOR_HH

namespace py = pybind11;

namespace Dune {
  namespace Ultraweak {
    namespace Python {
      template<typename Impl>
      class Operator {
      public:
        using DomainType = typename Impl::domain_type;
        using RangeType = typename Impl::range_type;

        Operator(const Impl& op) : op_(op),
          dimDomain(op.getmat().N()), dimRange(op.getmat().M())
        {}

        void apply(const DomainType& x, RangeType& y) const {
          op_.apply(x,y);
        }

      protected:
        const Impl& op_;
      public:
        std::size_t dimDomain;
        std::size_t dimRange;
      };
    }
  }
}

template<typename T>
void registerOperator(py::module m, const std::string clsName) {
  auto ret = Dune::Python::insertClass<T>(m, clsName,
    Dune::Python::GenerateTypeName(clsName),
    Dune::Python::IncludeFiles{"dune/ultraweak/py/operator.hh"});

  if (ret.second) {
    auto cls = ret.first;
    cls.def("apply", &T::apply, py::arg("x"), py::arg("y"));
    cls.def_readonly("dim_source", &T::dimDomain);
    cls.def_readonly("dim_range", &T::dimRange);
  }
}

#endif  // DUNE_ULTRAWEAK_PY_OPERATOR_HH
