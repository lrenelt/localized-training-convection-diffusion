#ifndef DUNE_ULTRAWEAK_CONV_DIFF_VISUALIZATION_HH
#define DUNE_ULTRAWEAK_CONV_DIFF_VISUALIZATION_HH

#include <iostream>

#include <dune/pdelab.hh>
#include "dune/ultraweak/convection-diffusion/traits.hh"

template<typename GV, typename Problem, typename Parametrization,
         typename Traits = ConvDiffNormalEqTraits<GV,1>>
void visualizeConvDiffNormalEqSolution(const GV& gv,
                                       const typename Traits::TensorGFS& gfs,
                                       const typename Traits::CoefficientVector coeffs,
                                       const Problem& problem,
                                       const Parametrization& parametrization,
                                       std::string filename,
                                       const int subsampling) {

  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::RefinementIntervals(subsampling), false);

  typename Traits::VectorSubGFS vectorSubGfs(gfs);
  const typename Traits::VectorDGF vectordgf(vectorSubGfs, coeffs);
  const auto vectorvtkf =
    std::make_shared<typename Traits::VectorVTKF>(vectordgf, "vector-valued-solution");
  vtkwriter.addVertexData(vectorvtkf);

  typename Traits::ScalarSubGFS scalarSubGfs(gfs);
  const typename Traits::ScalarDGF scalardgf(scalarSubGfs, coeffs);
  const auto scalarvtkf =
    std::make_shared<typename Traits::ScalarVTKF>(scalardgf, "scalar-valued-solution");
  vtkwriter.addVertexData(scalarvtkf);

  const typename Traits::VectorRecDGF<Problem,Parametrization>
    vectorrecdgf(gfs, coeffs, problem, parametrization);
  const auto vectorrecvtkf =
    std::make_shared<typename Traits::VectorRecVTKF<Problem,Parametrization>>
    (vectorrecdgf, "flux_reconstruction");
  vtkwriter.addCellData(vectorrecvtkf);

  const typename Traits::ScalarRecDGF<Problem,Parametrization>
    scalarrecdgf(gfs, coeffs, problem, parametrization);
  const auto scalarrecvtkf =
    std::make_shared<typename Traits::ScalarRecVTKF<Problem,Parametrization>>
    (scalarrecdgf, "scalar_reconstruction");
  vtkwriter.addCellData(scalarrecvtkf);

  std::cout << "Writing output to " << filename << ".vtu ..." << std::endl;
  vtkwriter.write(filename, Dune::VTK::ascii);
}

#endif  // DUNE_ULTRAWEAK_CONV_DIFF_VISUALIZATION_HH
