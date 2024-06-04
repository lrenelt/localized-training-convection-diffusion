#ifndef DUNE_ULTRAWEAK_PY_CONV_DIFF_SOLVER_HH
#define DUNE_ULTRAWEAK_PY_CONV_DIFF_SOLVER_HH

#include <dune/python/common/typeregistry.hh>
#include <dune/python/pybind11/operators.h>
#include <dune/python/pybind11/pybind11.h>

#include "dune/ultraweak/convection-diffusion/normalEqLocalOperator.hh"
#include "dune/ultraweak/convection-diffusion/parametrizedSolver.hh"

#include "dune/ultraweak/py/operator.hh"
#include "dune/ultraweak/py/parameter_tree.hh"

namespace py = pybind11;

namespace Dune {
  namespace Ultraweak {
    namespace Python {

      template<typename BlockType>
      py::list bvectorToList(Dune::BlockVector<BlockType> vec) {
        py::list listOfBlocks;
        for (auto& block : vec)
          listOfBlocks.append(block);
        return listOfBlocks;
      }

      template<typename BlockType>
      Dune::BlockVector<BlockType> listToBvector(py::list l) {
        Dune::BlockVector<BlockType> bvector(l.size());
        for (std::size_t i=0; i<l.size(); ++i)
          bvector[i] = l[i].cast<BlockType>();
        return bvector;
      }

      template<std::size_t order, typename RF, int dim,
               typename Problem, typename Parametrization,
               NormalEquationType type>
      class ConvDiffSolver {

      private:
        using Grid = Dune::YaspGrid<dim>;
        using GV = typename Grid::LeafGridView;
        using Base = ConvDiffParametrizedSolver<order,GV,Problem,Parametrization,type>;
        using WrappedVectorType = typename Base::Traits::CoefficientVector;

      public:
        using VectorType = typename Dune::PDELab::Backend::Native<WrappedVectorType>;
        using VectorBlockType = typename VectorType::block_type;
        using ParameterType = typename Parametrization::ParameterType;
        using OperatorType =  typename Dune::Ultraweak::Python::Operator<
          typename Base::OperatorType>;

        ConvDiffSolver(py::dict config) :
          pTree_(toParameterTree(config))
        {
          // grid setup
          assert(dim == 2);
          Dune::FieldVector<double, dim> domain({1.0, 1.0});
          std::array<int, dim> domainDims = {
            domainDims[0] = pTree_.get<int>("grid.yasp_x"),
            domainDims[1] = pTree_.get<int>("grid.yasp_y")};

          grid_ = std::make_unique<Grid>(domain, domainDims);
          gv_ = std::make_unique<GV>(grid_->leafGridView());

          // define the problem
          problem_ = std::make_unique<Problem>(pTree_);
          parametrization_ = std::make_unique<Parametrization>();

          // make solver object
          solver_ = std::make_unique<Base>(*gv_, *problem_, *parametrization_, pTree_);

          // set public members
          dimSource = solver_->getGfs().globalSize();
          dimRange = dimSource;

          // define the separated operators
          const auto& matrixOperators = solver_->getOperators();
          for(const auto& op : matrixOperators)
            operators_.emplace_back(op);
        }

        ConvDiffSolver(ConvDiffSolver&& other) = delete;

        auto solve(const ParameterType mu, py::dict solverConfig) {
          auto pTree = toParameterTree(solverConfig);
          return bvectorToList(Dune::PDELab::Backend::native(solver_->solve(mu, pTree)));
        }

        void visualize(py::list vec, const ParameterType mu,
                       const std::string filename="default_filename") {
          std::cout << "Writing output to " << filename << ".vtk ..." << std::endl;
          solver_->visualize(listToBvector<VectorBlockType>(vec), mu, filename);
        }

        // perform conversion to py::list of the system blocks
        const auto getMatrices() const {
          py::list matrices;
          for(const auto& mat : solver_->getMatrices()) {
            py::list listOfRows;
            for (std::size_t i=0; i<mat.N(); ++i) {
              py::list row;
              for (std::size_t j=0; j<mat.M(); ++j)
                row.append(mat[i][j]);
              listOfRows.append(row);
            }
            matrices.append(listOfRows);
          }
          return matrices;
        }

        const auto getL2MassMatrix() const {
          const auto& blockMat = solver_->getL2MassMatrix();
          py::list diagBlocks;
          for (std::size_t i=0; i<blockMat.N(); ++i)
            diagBlocks.append(blockMat[i][i]);
          return diagBlocks;
        }

        const auto getHdivH1MassMatrix() const {
          const auto& blockMat = solver_->getHdivH1MassMatrix();
          py::list diagBlocks;
          for (std::size_t i=0; i<blockMat.N(); ++i)
            diagBlocks.append(blockMat[i][i]);
          return diagBlocks;
        }

        const auto& getOperators() const {
          return operators_;
        }

        const auto getRhsVectors() const {
          py::list rhsVectors;
          for (auto& vec : solver_->getRhsVectors())
            rhsVectors.append(bvectorToList(vec));
          return rhsVectors;
        }

      protected:
        Dune::ParameterTree pTree_;
        std::unique_ptr<Problem> problem_;
        std::unique_ptr<Parametrization> parametrization_;
        std::unique_ptr<Base> solver_;
        std::unique_ptr<Grid> grid_;
        std::unique_ptr<GV> gv_;

        std::vector<OperatorType> operators_;

      public:
        std::size_t dimSource;
        std::size_t dimRange;
      };
    }
  }
}

template<typename T>
void registerConvDiffSolver(py::module m, const std::string clsName) {
  auto ret = Dune::Python::insertClass<T>(m, clsName,
    Dune::Python::GenerateTypeName(clsName),
    Dune::Python::IncludeFiles{"dune/ultraweak/py/convDiffSolver.hh"});

  if (ret.second) {
    auto cls = ret.first;

    cls.def(py::init<py::dict>(), py::arg("config"));
    cls.def_readonly("dim_source", &T::dimSource);
    cls.def_readonly("dim_range", &T::dimRange);

    cls.def("visualize", &T::visualize, "write graphic output for a given vector");
    cls.def("getMatrices", &T::getMatrices, "get the raw matrices");
    cls.def("getL2MassMatrix", &T::getL2MassMatrix, "get the l2 mass matrix");
    cls.def("getHdivH1MassMatrix", &T::getHdivH1MassMatrix, "get the reference mass matrix");
    cls.def("getRhsVectors", &T::getRhsVectors, "get the raw rhs vector");

    cls.def("solve", &T::solve, "solve for given parameter",
            py::arg("mu"), py::arg("solverConfig"));

    registerOperator<typename T::OperatorType>(m, clsName+"Operator");

    cls.def("getOperators", &T::getOperators, "get the matrix operators");
  }
}

#endif  // DUNE_ULTRAWEAK_PY_CONV_DIFF_SOLVER_HH
