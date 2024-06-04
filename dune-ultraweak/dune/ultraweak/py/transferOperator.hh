#ifndef DUNE_ULTRAWEAK_PY_TRANSFER_OPERATOR_HH
#define DUNE_ULTRAWEAK_PY_TRANSFER_OPERATOR_HH

#include <dune/python/common/typeregistry.hh>
#include <dune/python/pybind11/functional.h>
#include <dune/python/pybind11/operators.h>
#include <dune/python/pybind11/pybind11.h>

#include <dune/subgrid/subgrid.hh>

#include "dune/ultraweak/innerProductAssembler.hh"
#include "dune/ultraweak/convection-diffusion/multipleRhsNormalEqSolver.hh"
#include "dune/ultraweak/convection-diffusion/norm.hh"
#include "dune/ultraweak/convection-diffusion/restriction.hh"
#include "dune/ultraweak/convection-diffusion/weightedL2norm.hh"

#include "dune/ultraweak/py/convDiffSolver.hh"
#include "dune/ultraweak/py/parameter_tree.hh"

namespace py = pybind11;

namespace Dune {
  namespace Ultraweak {
    namespace Python {
      /**
         Models a parameter-independent transfer operator
         for a ConvectionDiffusion problem based on a normal equation.
       */
      template<std::size_t order, typename RF, int dim,
               typename Problem, typename Parametrization,
               NormalEquationType type>
      class TransferOperator {

      protected:
        using ctype = double;
        using Grid =  Dune::YaspGrid<dim,Dune::EquidistantOffsetCoordinates<ctype,dim>>;
        using GV = typename Grid::LeafGridView;
        using Base = MultipleRhsConvDiffNormalEqSolver<order,GV,Problem,Parametrization,type>;

        using Subgrid = Dune::SubGrid<GV::dimension,Grid>;
        using RestrictionOperator = Restriction<Subgrid,order>;

        using Traits = typename RestrictionOperator::Traits;

        using VectorType = Dune::PDELab::Backend::Native<typename Traits::CoefficientVector>;
        using VectorBlockType = typename VectorType::block_type;


        template<typename LOP>
        using Assembler = InnerProductAssembler<LOP,typename Traits::TensorGFS>;

        using DomainType = Dune::FieldVector<RF,2>;
        using Element = typename Subgrid::LeafGridView::Codim<0>::Entity;
        using WeightingFunction = std::function<
          typename Problem::MatrixType(const Element&, const DomainType&)>;
        using WeightedL2LOP = WeightedL2Norm<typename Traits::ScalarFEM, WeightingFunction>;
        using WeightedL2Assembler = Assembler<WeightedL2LOP>;

        using L2LOP = Dune::PDELab::L2;
        using L2Assembler = Assembler<L2LOP>;

        using HdivH1LOP = HdivH1norm<typename Traits::ScalarFEM>;
        using H1Assembler = Assembler<HdivH1LOP>;

      public:
        TransferOperator(py::dict config, py::dict solverConfig)
          : pTree_(toParameterTree(config)),
            solverConfig_(toParameterTree(solverConfig)),
            dimSource({0,0}), dimRange({0,0})
        {
          const auto lx = pTree_.get<double>("grid.lx");
          const auto ly = pTree_.get<double>("grid.ly");

          // TODO: currently an interior domain of [0,1]^2 is assumed
          const auto criterion = [lx,ly](const auto& e){
            const auto center = e.geometry().center();
            return (center[0] > 0.0 and center[1] > 0.0
                    and center[0] < lx and center[1] < ly);
          };

          // grid setup
          assert(dim == 2);

          const auto stride = pTree_.get<ctype>("grid.oversampling_stride");
          Dune::FieldVector<ctype,dim> lowerleft({-stride,-stride});
          Dune::FieldVector<ctype,dim> upperright(1.0);
          std::array<int, dim> domainDims;

          upperright[0] = pTree_.get<ctype>("grid.lx") + stride;
          upperright[1] = pTree_.get<ctype>("grid.ly") + stride;
          domainDims[0] = pTree_.get<int>("grid.yasp_x");
          domainDims[1] = pTree_.get<int>("grid.yasp_y");

          grid_ = std::make_unique<Grid>(lowerleft, upperright, domainDims);
          gv_ = std::make_unique<GV>(grid_->leafGridView());

          // define the problem
          problem_ = std::make_unique<Problem>(pTree_);

          // make solver object
          solver_ = std::make_unique<Base>(*gv_, *problem_, pTree_, solverConfig_);

          // set public members
          dimSource = solver_->gfsSize();

          std::cout << "Creating subgrid..." << std::endl;
          // make subgrid
          subgrid_ = std::make_unique<Subgrid>(*grid_);
          subgrid_->createBegin();
          for (const auto& e : elements(*gv_))
            if (criterion(e))
              subgrid_->insert(e);
          subgrid_->createEnd();

          restriction_ = std::make_unique<RestrictionOperator>(*subgrid_);
          dimRange = restriction_->gfsSize();

          // assembler for inner product on the subgrid
          L2LOP l2op;
          l2Assembler_ = std::make_shared<L2Assembler>(l2op, restriction_->getGfs());
          l2Assembler_->assemble();

          const auto nonparametricProblem = solver_->getParametrization().bindToProblem(*problem_);
          WeightingFunction weightingFunc
            = [nonparametricProblem](const auto& el, const auto& x){
              return nonparametricProblem.Ainv(el,x);
            };
          WeightedL2LOP weightedl2op(weightingFunc);
          weightedl2Assembler_ = std::make_shared<WeightedL2Assembler>(weightedl2op,
                                                               restriction_->getGfs());
          weightedl2Assembler_->assemble();

          HdivH1LOP h1op;
          h1Assembler_ = std::make_shared<H1Assembler>(h1op, restriction_->getGfs());
          h1Assembler_->assemble();
        }


        //! solve for given boundary data which is passed as a lambda
        auto solve(typename Problem::FunctionType boundaryFunction) {
          problem_->setCallback(boundaryFunction);
          solver_->solve();
          const auto solution = Dune::PDELab::Backend::native(
            restriction_->restrict(solver_->getCoefficientVector()));
          return bvectorToList(solution);
        }

        void visualize(py::list vec,
                       const std::string filename="default_filename") {
          restriction_->visualize(listToBvector<VectorBlockType>(vec),
                                  *problem_, solver_->getParametrization(),
                                  filename,
                                  pTree_.get<int>("visualization.subsampling"));
        }

        void visualizeOS(const std::string filename="default_filename") {
          const std::string filename_os = filename + "_osDomain";
          std::cout << "Writing output for the oversampling domain." << std::endl;
          std::cout << "Warning: This uses the last computed coefficients!" << std::endl;
          solver_->writeVTK(filename_os);
        }

        const auto getL2MassMatrix() const {
          const auto& blockMat = l2Assembler_->getAssembledMatrix();
          py::list diagBlocks;
          for (std::size_t i=0; i<blockMat.N(); ++i)
            diagBlocks.append(blockMat[i][i]);
          return diagBlocks;
        }

        const auto getWeightedL2MassMatrix() const {
          const auto& blockMat = weightedl2Assembler_->getAssembledMatrix();
          py::list diagBlocks;
          for (std::size_t i=0; i<blockMat.N(); ++i)
            diagBlocks.append(blockMat[i][i]);
          return diagBlocks;
        }

        const auto getHdivH1MassMatrix() const {
          const auto& blockMat = h1Assembler_->getAssembledMatrix();
          py::list diagBlocks;
          for (std::size_t i=0; i<blockMat.N(); ++i)
            diagBlocks.append(blockMat[i][i]);
          return diagBlocks;
        }

      private:
        std::shared_ptr<L2Assembler> l2Assembler_;
        std::shared_ptr<WeightedL2Assembler> weightedl2Assembler_;
        std::shared_ptr<H1Assembler> h1Assembler_;

      protected:
        Dune::ParameterTree pTree_;
        Dune::ParameterTree solverConfig_;
        std::unique_ptr<Problem> problem_;
        std::unique_ptr<Base> solver_;
        std::unique_ptr<Grid> grid_;
        std::unique_ptr<GV> gv_;

        std::unique_ptr<Subgrid> subgrid_;
        std::unique_ptr<RestrictionOperator> restriction_;

      public:
        std::array<std::size_t,2> dimSource;
        std::array<std::size_t,2> dimRange;
      };
    }
  }
}

template<typename T>
void registerTransferOperator(py::module m, const std::string clsName) {
  auto ret = Dune::Python::insertClass<T>(m, clsName,
    Dune::Python::GenerateTypeName(clsName),
    Dune::Python::IncludeFiles{"dune/ultraweak/py/transferOperator.hh"});

  if (ret.second) {
    auto cls = ret.first;

    cls.def(py::init<py::dict, py::dict>(), py::arg("config"), py::arg("solverConfig"));
    cls.def_readonly("dim_source", &T::dimSource);
    cls.def_readonly("dim_range", &T::dimRange);

    cls.def("visualize", &T::visualize, "write graphic output for a given vector");
    cls.def("visualizeOversampling", &T::visualizeOS, "plot the last computed full solution");

    cls.def("getL2MassMatrix", &T::getL2MassMatrix, "get the l2 mass matrix");
    cls.def("getWeightedL2MassMatrix", &T::getWeightedL2MassMatrix, "get the weighted l2 mass matrix");
    cls.def("getHdivH1MassMatrix", &T::getHdivH1MassMatrix, "get the reference mass matrix");

    cls.def("solve", &T::solve, py::arg("func"));

  }
}

#endif  // DUNE_ULTRAWEAK_PY_TRANSFER_OPERATOR_HH
