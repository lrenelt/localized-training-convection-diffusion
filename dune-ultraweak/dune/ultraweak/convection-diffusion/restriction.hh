#ifndef DUNE_ULTRAWEAK_CONV_DIFF_RESTRICTION_HH
#define DUNE_ULTRAWEAK_CONV_DIFF_RESTRICTION_HH

#include "dune/ultraweak/restriction.hh"
#include "dune/ultraweak/convection-diffusion/visualization.hh"

// TODO: deduce order?
template<typename Subgrid, std::size_t order=1>
class Restriction {
public:
  using OversamplingGrid = typename Subgrid::HostGridType;
  using GV = typename Subgrid::LeafGridView;
  using OversamplingGV = typename OversamplingGrid::LeafGridView;

  using VTKWriter =  Dune::SubsamplingVTKWriter<GV>;

  using Traits = ConvDiffNormalEqTraits<GV,order>;
  using OversamplingTraits = ConvDiffNormalEqTraits<OversamplingGV,order>;

  using VectorType = typename Traits::CoefficientVector;
  using VectorTypeOS = typename OversamplingTraits::CoefficientVector;

  using NativeCoefficientType = Dune::PDELab::Backend::Native<VectorType>;


  Restriction(Subgrid& subgrid) :
    _subgrid(subgrid), gv(subgrid.leafGridView()),
    vectorFem(gv), vectorGfs(gv, vectorFem),
    scalarFem(gv), scalarGfs(gv, scalarFem),
    gfs(vectorGfs, scalarGfs),
    gv_os(subgrid.getHostGrid().leafGridView()),
    vectorFem_os(gv_os), vectorGfs_os(gv_os, vectorFem_os),
    scalarFem_os(gv_os), scalarGfs_os(gv_os, scalarFem_os),
    gfs_os(vectorGfs_os, scalarGfs_os) {}

  auto restrict(const VectorTypeOS& vecIn) const {
    VectorType vecOut(gfs, 0.0);
    auto restriction =
      PDELabSubGridRestriction(_subgrid,
                               gfs, vecOut,
                               gfs_os, vecIn);
    _subgrid.transfer(restriction);

    return vecOut;
  }

  // TODO: this feels unintended...
  auto gfsSize() const {
    std::array<std::size_t,2> sz{0, 0};

    auto temp = Dune::PDELab::Backend::native(VectorType(gfs, 0.0));
    for (std::size_t i=0; i<temp.N(); ++i)
      sz[i] = temp[i].N();
    return sz;
  }


  auto assemble() const {
    using LOP = Dune::Ultraweak::SubgridRestriction;
    using CC = Dune::PDELab::NoConstraints;
    using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    using GO = Dune::PDELab::GridOperator<
      typename OversamplingTraits::TensorGFS,
      typename Traits::TensorGFS,
      LOP, MBE,
      typename Traits::RF, typename Traits::RF, typename Traits::RF,
      CC,CC>;

    LOP lop;
    CC cc;
    MBE mbe;
    GO go(gfs_os,cc,gfs,cc,lop,mbe);

    typename GO::Jacobian restrictionMatrix(go, 0.0);
    typename GO::Domain tempEvalPoint(gfs_os, 0.0);
    go.jacobian(tempEvalPoint, restrictionMatrix);

    return restrictionMatrix;
  }

  template<typename Problem, typename Parametrization>
  void visualize(NativeCoefficientType rawCoeffs,
                 const Problem& problem,
                 const Parametrization& parametrization,
                 const std::string filename,
                 const int subsampling=1) const {
    const auto coeffs = typename Traits::CoefficientVector(gfs, rawCoeffs);
    writeVTK(coeffs, problem, parametrization, filename, subsampling);
  }

  template<typename Problem, typename Parametrization>
  void writeVTK(const VectorType& solution,
                const Problem& problem,
                const Parametrization& parametrization,
                const std::string filename,
                const std::size_t subsampling=1) const {
   visualizeConvDiffNormalEqSolution(gv, gfs, solution,
                                     problem, parametrization,
                                     filename, subsampling);
  }

  auto& getGfs() const {
    return gfs;
  }

protected:
  Subgrid& _subgrid;
  GV gv;
  const typename Traits::VectorFEM vectorFem;
  typename Traits::VectorGFS vectorGfs;
  const typename Traits::ScalarFEM scalarFem;
  typename Traits::ScalarGFS scalarGfs;
  typename Traits::TensorGFS gfs;

  OversamplingGV gv_os;
  const typename OversamplingTraits::VectorFEM vectorFem_os;
  typename OversamplingTraits::VectorGFS vectorGfs_os;
  const typename OversamplingTraits::ScalarFEM scalarFem_os;
  typename OversamplingTraits::ScalarGFS scalarGfs_os;
  typename OversamplingTraits::TensorGFS gfs_os;
};

#endif  // DUNE_ULTRAWEAK_CONV_DIFF_RESTRICTION_HH
