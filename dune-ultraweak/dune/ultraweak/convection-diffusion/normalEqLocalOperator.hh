#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORMAL_EQ_LOP_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORMAL_EQ_LOP_HH

#include <dune/pdelab.hh>

#include "dune/ultraweak/parametricCoefficientsWrapper.hh"

// TODO: additional flag to switch to diffusive flux implementation?
// TODO: implement 'classicTotalFlux'
enum NormalEquationType {
  classic,
  classicTotalFlux,
  adjoint
};

/**
   Local Operator assembling the normal equations for a convection-diffusion problem.

   For 'type==adjoint', the additional flux variable represents
   the total flux (A * grad u - b*u)!

   For 'type==classic', the flux is by default the diffusive flux (A * grad u).
*/
template<typename Problem, typename Parametrization, typename ScalarFEM,
         NormalEquationType type>
class SymmetricConvDiffMixedOperator :
  public Dune::PDELab::FullVolumePattern,
  public Dune::PDELab::LocalOperatorDefaultFlags
{
public:
  // define flags controlling global assembler
  enum { doPatternVolume = true };
  enum { doAlphaVolume = true };
  enum { doAlphaBoundary = true };
  enum { doLambdaVolume = true };
  enum { doLambdaBoundary = true };

  using LocalBasisTypeCG = typename
    ScalarFEM::Traits::FiniteElementType::Traits::LocalBasisType;
  using CacheTypeCG = Dune::PDELab::LocalBasisCache<LocalBasisTypeCG>;

  static constexpr std::size_t nA = Problem::nDiffusionComponents;
  static constexpr std::size_t nC = Problem::nReactionComponents;

  // TODO: make parameter
  static constexpr bool penalizeCurl = false;

  // always assumes LFSU == LFSV!
  SymmetricConvDiffMixedOperator (const Problem& problem_,
                                  const Parametrization& parametrization,
                                  const int intorder_,
                                  const double rescalingBoundary) :
    problem(problem_), parametrization_(parametrization),
    intorder(intorder_), rescalingBoundary_(rescalingBoundary) {}

  // volume integral depending on test and ansatz functions
  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume (const EG& eg, const LFSU& lfsu, const X& x,
                     const LFSV& lfsv, R& r) const {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using ScalarSpace = typename LFSU::template Child<1>::Type;

    using RF = typename
      FluxSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename FluxSpace::Traits::SizeType;

    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using ScalarRangeType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using ScalarJacobianType = typename
      ScalarSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::JacobianType;
    const int dim = EG::Geometry::mydimension;

    // select the subspaces
    const auto& fluxspace = lfsu.template child<0>();
    const auto& scalarspace = lfsu.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = eg.geometry();
    const auto& cell = eg.entity();

    auto gradtau = Dune::PDELab::makeJacobianContainer(fluxspace);
    auto gradtautransformed = Dune::PDELab::makeJacobianContainer(fluxspace);
    auto gradphi = Dune::PDELab::makeJacobianContainer(scalarspace);

    std::array<FluxRangeType,nA> Ainvsigma;
    std::array<FluxRangeType,nA> Ainvb;

    ScalarRangeType divsigma;
    ScalarRangeType curlsigma;
    ScalarRangeType u;
    ScalarJacobianType gradu;
    std::array<ScalarRangeType,nC> cu;

    std::vector<FluxRangeType> tau(fluxspace.size());
    std::vector<FluxRangeType> tautransformed(fluxspace.size());
    std::vector<std::array<FluxRangeType,nA>> Ainvtau(fluxspace.size());
    std::vector<ScalarRangeType> divtau(fluxspace.size());
    std::vector<ScalarRangeType> curltau(fluxspace.size());

    // TODO: not needed for classic version
    std::array<ScalarRangeType,nA> Ainvbsigma;
    std::vector<std::array<ScalarRangeType,nA>> Ainvbtau(fluxspace.size());

    for (const auto& qp : quadratureRule(geo, intorder)) {
      const auto S = geo.jacobianInverseTransposed(qp.position());

      // RT0 functions + gradient
      fluxspace.finiteElement().localBasis().evaluateFunction(qp.position(),tau);
      fluxspace.finiteElement().localBasis().evaluateJacobian(qp.position(),gradtau);

      // scalar functions + gradient
      auto& phi = cache_cg.evaluateFunction(qp.position(), scalarspace.finiteElement().localBasis());
      auto& jacScalar = cache_cg.evaluateJacobian(qp.position(), scalarspace.finiteElement().localBasis());

      // transform flux functions using the Piola transformation
      tautransformed = tau;

      // transform gradient
      for (size_type i=0; i<fluxspace.size(); i++)
        for (size_type k=0; k<dim; k++)
          S.mv(gradtau[i][k], gradtautransformed[i][k]);

      for (size_type j=0; j<scalarspace.size(); j++)
        S.mv(jacScalar[j][0], gradphi[j][0]);

      // evaluate parameter functions
      const auto reaction = problem.c(cell, qp.position());
      const FluxRangeType velocity = problem.b(cell, qp.position());
      auto invDiffusivity = problem.Ainv(cell, qp.position());
      for (std::size_t b=0; b<nA; ++b)
        invDiffusivity[b].mv(velocity, Ainvb[b]);

      // compute A^{-1} * tau, div tau and (A^{-1}b) * tau
      for (size_type i=0; i<fluxspace.size(); i++) {
        for (std::size_t b=0; b<nA; ++b) {
          Ainvtau[i][b] = 0.0;
          invDiffusivity[b].mv(tautransformed[i], Ainvtau[i][b]);
          Ainvbtau[i][b] = Ainvb[b].dot(tautransformed[i]);
        }
        divtau[i] = 0.0;
        for (int k=0; k<dim; k++)
          divtau[i] += gradtautransformed[i][k][k];
        curltau[i] = gradtautransformed[i][0][1] - gradtautransformed[i][1][0];
      }

      // compute A^{-1} * sigma, div sigma and (A^{-1}b)*sigma
      Ainvsigma.fill(FluxRangeType(0.0));
      Ainvbsigma.fill(ScalarRangeType(0.0));
      divsigma = 0.0;
      curlsigma = 0.0;
      for (size_type i=0; i<fluxspace.size(); i++) {
        for (std::size_t b=0; b<nA; ++b) {
          Ainvsigma[b].axpy(x(fluxspace,i), Ainvtau[i][b]);
          Ainvbsigma[b].axpy(x(fluxspace,i), Ainvbtau[i][b]);
        }
        divsigma.axpy(x(fluxspace,i), divtau[i]);
        curlsigma.axpy(x(fluxspace,i), curltau[i]);
      }

      // compute u and grad u
      u = 0.0;
      gradu = 0.0;
      for (size_type j=0; j<scalarspace.size(); j++) {
        u += x(scalarspace,j) * phi[j];
        gradu.axpy(x(scalarspace,j), gradphi[j]);
      }

      cu.fill(ScalarRangeType(0.0));
      for (std::size_t c=0; c<nC; ++c)
        cu[c] = reaction[c]*u;

      const RF factor = qp.weight() * geo.integrationElement(qp.position());

      const auto& bil = parametrization_.bilinear();

      std::array<ScalarRangeType,nC> cphi;
      cphi.fill(ScalarRangeType(0.0));

      if constexpr (type == NormalEquationType::classic) {
        // assemble terms dependent on the flux test function
        for (size_type i=0; i<fluxspace.size(); i++) {
          auto args1Vec = std::make_tuple(-gradu[0],Ainvsigma,zero);
          auto args2Vec = std::make_tuple(zero,Ainvtau[i],zero);
          r.accumulate(fluxspace, i, factor *
                       bil.makeParametricLincombVec(args1Vec,args2Vec));

          auto args1 = std::make_tuple(-divsigma + velocity.dot(gradu[0]),zeros(nA),cu);
          auto args2 = std::make_tuple(-divtau[i],zeros(nA),zeros(nC));
          r.accumulate(fluxspace, i, factor *
                       bil.template makeParametricLincomb(args1,args2));

          if constexpr (penalizeCurl) {
            // additional curl-constraint
            auto args1Curl = std::make_tuple(curlsigma,zeros(nA),zeros(nC));
            auto args2Curl = std::make_tuple(curltau[i],zeros(nA),zeros(nC));
            r.accumulate(scalarspace, i, factor *
                         bil.template makeParametricLincomb(args1Curl,args2Curl));
          }
        }

        // assemble terms dependent on the scalar test function
        for (size_type j=0; j<scalarspace.size(); j++) {
          auto args1Vec = std::make_tuple(-gradu[0],Ainvsigma,zeros(nC));
          auto args2Vec = std::make_tuple(-gradphi[j][0],zeros(nA),zeros(nC));
          r.accumulate(scalarspace, j, factor *
                       bil.makeParametricLincombVec(args1Vec,args2Vec));

          for (std::size_t c=0; c<nC; ++c)
            cphi[c] = reaction[c]*phi[j];

          auto args1 = std::make_tuple(-divsigma + velocity.dot(gradu[0]),zeros(nA),cu);
          auto args2 = std::make_tuple(velocity.dot(gradphi[j][0]),zeros(nA),cphi);
          r.accumulate(scalarspace, j, factor *
                       bil.template makeParametricLincomb(args1,args2));
        }
      }
      else if constexpr (type == NormalEquationType::adjoint) {
        // assemble terms dependent on the flux test function
        for (size_type i=0; i<fluxspace.size(); i++) {
          auto args1Vec = std::make_tuple(gradu[0],Ainvsigma,zero);
          auto args2Vec = std::make_tuple(zero,Ainvtau[i],zero);
          r.accumulate(fluxspace, i, factor *
                       bil.makeParametricLincombVec(args1Vec,args2Vec));

          auto args1 = std::make_tuple(divsigma,Ainvbsigma,cu);
          auto args2 = std::make_tuple(divtau[i],Ainvbtau[i],zeros(nC));
          r.accumulate(fluxspace, i, factor *
                       bil.template makeParametricLincomb(args1,args2));
        }

        // assemble terms dependent on the scalar test function
        for (size_type j=0; j<scalarspace.size(); j++) {
          auto args1Vec = std::make_tuple(gradu[0],Ainvsigma,zeros(nC));
          auto args2Vec = std::make_tuple(gradphi[j][0],zeros(nA),zeros(nC));
          r.accumulate(scalarspace, j, factor *
                       bil.makeParametricLincombVec(args1Vec,args2Vec));

          for (std::size_t c=0; c<nC; ++c)
            cphi[c] = reaction[c]*phi[j];

          auto args1 = std::make_tuple(divsigma,Ainvbsigma,cu);
          auto args2 = std::make_tuple(zero,zeros(nA),cphi);
          r.accumulate(scalarspace, j, factor *
                       bil.template makeParametricLincomb(args1,args2));
        }
      }
    }
  }

  // volume integral depending on test and ansatz functions
  template<typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume (const EG& eg, const LFSU& lfsu, const X& x,
                     const LFSV& lfsv, M& mat) const {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using ScalarSpace = typename LFSU::template Child<1>::Type;

    using RF = typename
      FluxSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename FluxSpace::Traits::SizeType;

    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using ScalarRangeType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    const int dim = EG::Geometry::mydimension;

    // select the subspaces
    const auto& fluxspace_u = lfsu.template child<0>();
    const auto& scalarspace_u = lfsu.template child<1>();
    const auto& fluxspace_v = lfsv.template child<0>();
    const auto& scalarspace_v = lfsv.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = eg.geometry();
    const auto& cell = eg.entity();

    auto gradtau = Dune::PDELab::makeJacobianContainer(fluxspace_u);
    auto gradtautransformed = Dune::PDELab::makeJacobianContainer(fluxspace_u);
    auto gradphi = Dune::PDELab::makeJacobianContainer(scalarspace_u);

    std::array<FluxRangeType,nA> Ainvb;
    std::vector<FluxRangeType> tau(fluxspace_u.size());
    std::vector<FluxRangeType> tautransformed(fluxspace_u.size());
    std::vector<std::array<FluxRangeType,nA>> Ainvtau(fluxspace_u.size());
    std::vector<std::array<ScalarRangeType,nC>> cphi(scalarspace_u.size());
    std::vector<ScalarRangeType> divtau(fluxspace_u.size());
    std::vector<ScalarRangeType> curltau(fluxspace_u.size());

    // TODO: not needed for classic version
    std::vector<std::array<ScalarRangeType,nA>> Ainvbtau(fluxspace_u.size());

    for (const auto& qp : quadratureRule(geo, intorder)) {
      const auto S = geo.jacobianInverseTransposed(qp.position());

      // RT0 functions + gradient
      fluxspace_u.finiteElement().localBasis().evaluateFunction(qp.position(),tau);
      fluxspace_u.finiteElement().localBasis().evaluateJacobian(qp.position(),gradtau);

      // scalar functions + gradient
      auto& phi = cache_cg.evaluateFunction(qp.position(), scalarspace_u.finiteElement().localBasis());
      auto& jacScalar = cache_cg.evaluateJacobian(qp.position(), scalarspace_u.finiteElement().localBasis());

      // transform flux functions using the Piola transformation
      tautransformed = tau;

      // evaluate parameter functions
      const auto reaction = problem.c(cell, qp.position());
      const FluxRangeType velocity = problem.b(cell, qp.position());
      auto invDiffusivity = problem.Ainv(cell, qp.position());
      for (std::size_t b=0; b<nA; ++b)
        invDiffusivity[b].mv(velocity, Ainvb[b]);

      for (size_type j=0; j<scalarspace_u.size(); j++) {
        S.mv(jacScalar[j][0], gradphi[j][0]);
        for (std::size_t c=0; c<nC; ++c)
          cphi[j][c] = reaction[c]*phi[j];
      }

      // compute A^{-1} * tau, div tau and (A^{-1}b) * tau
      for (size_type i=0; i<fluxspace_u.size(); i++) {
        for (std::size_t b=0; b<nA; ++b) {
          Ainvtau[i][b] = 0.0;
          invDiffusivity[b].mv(tautransformed[i], Ainvtau[i][b]);
          Ainvbtau[i][b] = Ainvb[b].dot(tautransformed[i]);
        }
        divtau[i] = 0.0;
        for (int k=0; k<dim; k++) {
          S.mv(gradtau[i][k], gradtautransformed[i][k]);
          divtau[i] += gradtautransformed[i][k][k];
        }
        curltau[i] = gradtautransformed[i][0][1] - gradtautransformed[i][1][0];
      }

      const RF factor = qp.weight() * geo.integrationElement(qp.position());

      const auto& bil = parametrization_.bilinear();


      if constexpr (type == NormalEquationType::classic) {
        for (size_type i1=0; i1<fluxspace_u.size(); i1++) {
          // assemble terms dependent on the flux test function
          for (size_type i2=0; i2<fluxspace_v.size(); i2++) {
            // (A^{-1}\tau_i, A^{-1}\tau_j)
            auto args1Vec = std::make_tuple(zero,Ainvtau[i1],zero);
            auto args2Vec = std::make_tuple(zero,Ainvtau[i2],zero);
            mat.accumulate(fluxspace_u, i1, fluxspace_v, i2, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            // (-\nabla\cdot\tau_i, -\nabla\cdot\tau_j)
            auto args1 = std::make_tuple(-divtau[i1],zeros(nA),zeros(nC));
            auto args2 = std::make_tuple(-divtau[i2],zeros(nA),zeros(nC));
            mat.accumulate(fluxspace_u, i1, fluxspace_v, i2, factor *
                           bil.makeParametricLincomb(args1,args2));
            if constexpr (penalizeCurl) {
              auto args1Curl = std::make_tuple(curltau[i1],zeros(nA),zeros(nC));
              auto args2Curl = std::make_tuple(curltau[i2],zeros(nA),zeros(nC));
              mat.accumulate(fluxspace_u, i1, fluxspace_v, i2,  factor *
                           bil.template makeParametricLincomb(args1Curl,args2Curl));
            }
          }
          // assemble terms dependent on the scalar test function
          for (size_type j=0; j<scalarspace_v.size(); j++) {
            // (A^{-1}\tau, -\nabla\phi)
            auto args1Vec = std::make_tuple(zero,Ainvtau[i1],zero);
            auto args2Vec = std::make_tuple(-gradphi[j][0],zeros(nA),zero);
            mat.accumulate(fluxspace_u, i1, scalarspace_v, j, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            // (-\nabla\cdot\tau, b\nabla\phi + c\phi)
            auto args1 = std::make_tuple(-divtau[i1],zeros(nA),zeros(nC));
            auto args2 = std::make_tuple(velocity.dot(gradphi[j][0]),zeros(nA),cphi[j]);
            mat.accumulate(fluxspace_u, i1, scalarspace_v, j, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
        }

        for (size_type j1=0; j1<scalarspace_u.size(); j1++) {
          // assemble terms dependent on the flux test function
          for (size_type i=0; i<fluxspace_v.size(); i++) {
            // (-\nabla\phi, A^{-1}\tau)
            auto args1Vec = std::make_tuple(-gradphi[j1][0],zeros(nA),zero);
            auto args2Vec = std::make_tuple(zero,Ainvtau[i],zero);
            mat.accumulate(scalarspace_u, j1, fluxspace_v, i, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            // (b\nabla\phi + c\phi, -\nabla\cdot\tau)
            auto args1 = std::make_tuple(velocity.dot(gradphi[j1][0]),zeros(nA),cphi[j1]);
            auto args2 = std::make_tuple(-divtau[i],zeros(nA),zeros(nC));
            mat.accumulate(scalarspace_u, j1, fluxspace_v, i, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
          // assemble terms dependent on the scalar test function
          for (size_type j2=0; j2<scalarspace_v.size(); j2++) {
            // (-\nabla\phi_i,-\nabla\phi_j)
            auto args1Vec = std::make_tuple(-gradphi[j1][0],zeros(nA),zero);
            auto args2Vec = std::make_tuple(-gradphi[j2][0],zeros(nA),zero);
            mat.accumulate(scalarspace_u, j1, scalarspace_v, j2, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            // (b\nabla\phi_i + c\phi_i, b\nabla\phi_j + c\phi_j)
            auto args1 = std::make_tuple(velocity.dot(gradphi[j1][0]),zeros(nA),cphi[j1]);
            auto args2 = std::make_tuple(velocity.dot(gradphi[j2][0]),zeros(nA),cphi[j2]);
            mat.accumulate(scalarspace_u, j1, scalarspace_v, j2, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
        }
      }
      else if constexpr (type == NormalEquationType::adjoint) {
        for (size_type i1=0; i1<fluxspace_u.size(); i1++) {
          // assemble terms dependent on the flux test function
          for (size_type i2=0; i2<fluxspace_v.size(); i2++) {
            auto args1Vec = std::make_tuple(zero,Ainvtau[i1],zero);
            auto args2Vec = std::make_tuple(zero,Ainvtau[i2],zero);
            mat.accumulate(fluxspace_u, i1, fluxspace_v, i2, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            auto args1 = std::make_tuple(divtau[i1],Ainvbtau[i1],zeros(nC));
            auto args2 = std::make_tuple(divtau[i2],Ainvbtau[i2],zeros(nC));
            mat.accumulate(fluxspace_u, i1, fluxspace_v, i2, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
          // assemble terms dependent on the scalar test function
          for (size_type j=0; j<scalarspace_v.size(); j++) {
            auto args1Vec = std::make_tuple(zero,Ainvtau[i1],zero);
            auto args2Vec = std::make_tuple(gradphi[j][0],zeros(nA),zero);
            mat.accumulate(fluxspace_u, i1, scalarspace_v, j, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            auto args1 = std::make_tuple(divtau[i1],Ainvbtau[i1],zeros(nC));
            auto args2 = std::make_tuple(zero,zeros(nA),cphi[j]);
            mat.accumulate(fluxspace_u, i1, scalarspace_v, j, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
        }

        for (size_type j1=0; j1<scalarspace_u.size(); j1++) {
          // assemble terms dependent on the flux test function
          for (size_type i=0; i<fluxspace_v.size(); i++) {
            auto args1Vec = std::make_tuple(gradphi[j1][0],zeros(nA),zero);
            auto args2Vec = std::make_tuple(zero,Ainvtau[i],zero);
            mat.accumulate(scalarspace_u, j1, fluxspace_v, i, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            auto args1 = std::make_tuple(zero,zeros(nA),cphi[j1]);
            auto args2 = std::make_tuple(divtau[i],Ainvbtau[i],zeros(nC));
            mat.accumulate(scalarspace_u, j1, fluxspace_v, i, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
          // assemble terms dependent on the scalar test function
          for (size_type j2=0; j2<scalarspace_v.size(); j2++) {
            auto args1Vec = std::make_tuple(gradphi[j1][0],zeros(nA),zero);
            auto args2Vec = std::make_tuple(gradphi[j2][0],zeros(nA),zero);
            mat.accumulate(scalarspace_u, j1, scalarspace_v, j2, factor *
                           bil.makeParametricLincombVec(args1Vec,args2Vec));

            auto args1 = std::make_tuple(zero,zeros(nA),cphi[j1]);
            auto args2 = std::make_tuple(zero,zeros(nA),cphi[j2]);
            mat.accumulate(scalarspace_u, j1, scalarspace_v, j2, factor *
                           bil.makeParametricLincomb(args1,args2));
          }
        }
      }
    }
  }

  template<typename IG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_boundary(const IG& ig, const LFSU& lfsu_s, const X& x,
                      const LFSV& lfsv_s, R& r_s) const {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using ScalarSpace = typename LFSU::template Child<1>::Type;

    using RF = typename
      FluxSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename FluxSpace::Traits::SizeType;

    using ScalarRangeType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;

    // select the subspaces
    const auto& fluxspace = lfsu_s.template child<0>();
    const auto& scalarspace = lfsu_s.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = ig.geometry();
    const auto geoInInside = ig.geometryInInside();

    std::vector<FluxRangeType> tau(fluxspace.size());
    std::vector<FluxRangeType> tautransformed(fluxspace.size());

    ScalarRangeType u(0.0);
    FluxRangeType sigma(0.0);
    RF sigmaN = 0.0;

    for (const auto& qp : quadratureRule(geo, intorder)) {
      auto bctype = problem.bctype(ig.intersection(), qp.position());
      const auto& insidepos = geoInInside.global(qp.position());
      const auto integrationElement = geo.integrationElement(qp.position());
      const RF factor = rescalingBoundary_ * qp.weight() * integrationElement
        * parametrization_.bilinear().theta(0);

      if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet) {
        // scalar functions
        auto& phi = cache_cg.evaluateFunction(insidepos, scalarspace.finiteElement().localBasis());

        // evaluate u
        u = 0.0;
        for (size_type j=0; j<scalarspace.size(); j++)
          u += x(scalarspace,j) * phi[j];

        for (size_type i=0; i<scalarspace.size(); i++)
          r_s.accumulate(scalarspace, i, factor * u * phi[i]);
      }
      else if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann) {
        const auto& normal = ig.unitOuterNormal(qp.position());
        fluxspace.finiteElement().localBasis().evaluateFunction(insidepos,tau);

        // transform flux functions using the Piola transformation
        tautransformed = tau;

        // evaluate sigma
        sigma = FluxRangeType(0.0);
        for (size_type i=0; i<fluxspace.size(); i++)
          sigma += x(scalarspace,i) * tautransformed[i].dot(normal);

        if constexpr (type == NormalEquationType::classic) {
          // assemble (\sigma\nu)(\tau\nu)
          for (size_type i=0; i<fluxspace.size(); i++)
            r_s.accumulate(fluxspace, i,
                           factor * sigma.dot(normal) * tautransformed[i].dot(normal));
        }
        else if constexpr (type == NormalEquationType::adjoint) {
          // assemble (\sigma\nu - b\nu u)(\tau\nu - b\nu v)
          auto& phi = cache_cg.evaluateFunction(insidepos,
                                                scalarspace.finiteElement().localBasis());
          // evaluate u
          u = 0.0;
          for (size_type j=0; j<scalarspace.size(); j++)
            u += x(scalarspace,j) * phi[j];

          const auto bn = problem.b(ig, qp.position()).dot(normal);
          sigmaN = sigma.dot(normal) - bn*u;

          for (size_type i=0; i<scalarspace.size(); i++)
            r_s.accumulate(scalarspace, i, factor * sigmaN * -bn * phi[i]);
          for (size_type i=0; i<fluxspace.size(); i++)
            r_s.accumulate(fluxspace, i, factor * sigmaN * tautransformed[i].dot(normal));
        }
      }
      else if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow) {
        if constexpr (type == NormalEquationType::classic) {
          // assemble (\sigma\nu - \nu\nabla u)(\tau\nu - \nu\nabla v)
        }
        else if constexpr (type == NormalEquationType::adjoint) {
          // TODO
        }
      }
    }
 }

  template<typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG& ig, const LFSU& lfsu_s, const X& x,
                      const LFSV& lfsv_s, M& mat_ss) const {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;

    using ScalarSpace = typename LFSU::template Child<1>::Type;
    using RF = typename
      ScalarSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename ScalarSpace::Traits::SizeType;

    // select the subspaces
    const auto& fluxspace_u = lfsu_s.template child<0>();
    const auto& scalarspace_u = lfsu_s.template child<1>();
    const auto& fluxspace_v = lfsv_s.template child<0>();
    const auto& scalarspace_v = lfsv_s.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = ig.geometry();
    const auto geoInInside = ig.geometryInInside();

    std::vector<FluxRangeType> tau(fluxspace_u.size());
    std::vector<FluxRangeType> tautransformed(fluxspace_u.size());

    for (const auto& qp : quadratureRule(geo, intorder)) {
      auto bctype = problem.bctype(ig.intersection(), qp.position());
      const auto& insidepos = geoInInside.global(qp.position());
      const auto integrationElement = geo.integrationElement(qp.position());
      const RF factor = rescalingBoundary_ * qp.weight() * integrationElement
        * parametrization_.bilinear().theta(0);

      if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet) {
        // scalar functions
        auto& phi = cache_cg.evaluateFunction(insidepos, scalarspace_u.finiteElement().localBasis());
        for (size_type i=0; i<scalarspace_u.size(); i++)
          for (size_type j=0; j<scalarspace_v.size(); j++)
            mat_ss.accumulate(scalarspace_u, i, scalarspace_v, j,
                              factor * phi[i] * phi[j]);
      }
      else if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann) {
        const auto& normal = ig.unitOuterNormal(qp.position());
        fluxspace_u.finiteElement().localBasis().evaluateFunction(insidepos,tau);

        // transform flux functions using the Piola transformation
        tautransformed = tau;

        if constexpr (type == NormalEquationType::classic) {
          // assemble (\sigma\nu)(\tau\nu)
          for (size_type i1=0; i1<fluxspace_u.size(); i1++)
            for (size_type i2=0; i2<fluxspace_v.size(); i2++)
              mat_ss.accumulate(fluxspace_u, i1, fluxspace_v, i2,
                                factor * tautransformed[i1].dot(normal)
                                * tautransformed[i2].dot(normal));
        }
        else if constexpr (type == NormalEquationType::adjoint) {
          // assemble (\sigma\nu - b\nu u)(\tau\nu - b\nu v)
          auto& phi = cache_cg.evaluateFunction(insidepos,
                                                scalarspace_u.finiteElement().localBasis());

          const auto bn = problem.b(ig, qp.position()).dot(normal);

          for (size_type i1=0; i1<scalarspace_u.size(); i1++) {
            for (size_type j1=0; j1<scalarspace_v.size(); j1++)
              mat_ss.accumulate(scalarspace_u, i1, scalarspace_v, j1,
                                factor * bn * phi[i1] * bn * phi[j1]);
            for (size_type j2=0; j2<fluxspace_v.size(); j2++)
              mat_ss.accumulate(scalarspace_u, i1, fluxspace_v, j2,
                                factor * -bn * phi[i1] * tautransformed[j2].dot(normal));
          }
          for (size_type i2=0; i2<fluxspace_u.size(); i2++) {
            for (size_type j1=0; j1<scalarspace_v.size(); j1++)
              mat_ss.accumulate(fluxspace_u, i2, scalarspace_v, j1,
                                factor * tautransformed[i2].dot(normal) * -bn * phi[j1]);
            for (size_type j2=0; j2<fluxspace_v.size(); j2++)
              mat_ss.accumulate(fluxspace_u, i2, fluxspace_v, j2,
                                factor * tautransformed[i2].dot(normal)
                                * tautransformed[j2].dot(normal));
          }
        }
      }
    }
  }


  template<typename EG, typename LFSV, typename R>
  void lambda_volume (const EG& eg, const LFSV& lfsv, R& r) const {
    using ScalarSpace = typename LFSV::template Child<1>::Type;

    using RF = typename
      ScalarSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename ScalarSpace::Traits::SizeType;
    using ScalarRangeType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;

    const int dim = EG::Geometry::mydimension;

    const auto geo = eg.geometry();
    const auto& cell = eg.entity();

    const auto& fluxspace = lfsv.template child<0>();
    const auto& scalarspace = lfsv.template child<1>();

    auto gradphi = Dune::PDELab::makeJacobianContainer(scalarspace);
    auto gradtau = Dune::PDELab::makeJacobianContainer(fluxspace);
    auto gradtautransformed = Dune::PDELab::makeJacobianContainer(fluxspace);
    std::vector<ScalarRangeType> divtau(fluxspace.size());

    for (const auto& qp : quadratureRule(geo, intorder)) {
      // evaluate scalar shape functions
      auto& phi = cache_cg.evaluateFunction(qp.position(),
                                            scalarspace.finiteElement().localBasis());
      // evaluate source function
      const auto source = problem.f(cell, qp.position());

      const RF integrationFactor = qp.weight() * geo.integrationElement(qp.position());

      if constexpr (type == NormalEquationType::classic) {
        // assemble (f, Av)
        fluxspace.finiteElement().localBasis().evaluateJacobian(qp.position(),gradtau);
        auto& jacScalar = cache_cg.evaluateJacobian(qp.position(),
                                                    scalarspace.finiteElement().localBasis());

        const auto S = geo.jacobianInverseTransposed(qp.position());
        for (size_type j=0; j<scalarspace.size(); j++)
          S.mv(jacScalar[j][0], gradphi[j][0]);

        for (size_type i=0; i<fluxspace.size(); i++) {
          divtau[i] = 0.0;
          for (int k=0; k<dim; k++) {
            S.mv(gradtau[i][k], gradtautransformed[i][k]);
            divtau[i] += gradtautransformed[i][k][k];
          }
        }

        /*
          TODO: only a workaround for non-parametrized problems! Everything should be in the
          rhs-parametrization for the offline-assembly to work!
        */
        const auto reaction = problem.c(cell, qp.position());
        double summedReaction = 0.0;
        for (std::size_t c=0; c<nC; ++c)
          summedReaction += parametrization_.bilinear().right().theta(1+nA+c)*reaction[c];
        const auto velocity = problem.b(cell, qp.position());

        for (size_type i=0; i<fluxspace.size(); i++)
          r.accumulate(fluxspace, i, integrationFactor *
                       parametrization_.rhs().theta(1) * -source * -divtau[i]);
        for (size_type j=0; j<scalarspace.size(); j++)
          r.accumulate(scalarspace, j, integrationFactor *
                       parametrization_.rhs().theta(1) *
                       -source * (velocity.dot(gradphi[j][0]) + summedReaction * phi[j]));
      }
      else if constexpr (type == NormalEquationType::adjoint) {
        // assemble (f,v)
        for (size_type i=0; i<scalarspace.size(); i++)
          r.accumulate(scalarspace, i, integrationFactor *
                       parametrization_.rhs().theta(1) * -source * phi[i]);
      }
    }
  }

  template<typename IG, typename LFSV, typename R>
  void lambda_boundary (const IG& ig, const LFSV& lfsv_s, R& r_s) const {
    using FluxSpace = typename LFSV::template Child<0>::Type;
    using ScalarSpace = typename LFSV::template Child<1>::Type;

    using RF = typename
      ScalarSpace::Traits::FiniteElementType::Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename ScalarSpace::Traits::SizeType;

    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;

    const auto geo = ig.geometry();
    const auto geoInInside = ig.geometryInInside();
    const auto& interface = ig.intersection();

    const auto& fluxspace = lfsv_s.template child<0>();
    const auto& scalarspace = lfsv_s.template child<1>();

    std::vector<FluxRangeType> tau(fluxspace.size());
    std::vector<FluxRangeType> tautransformed(fluxspace.size());

    for (const auto& qp : quadratureRule(geo, intorder)) {
      const auto& insidepos = geoInInside.global(qp.position());
      const auto integrationElement = geo.integrationElement(qp.position());
      const RF factor = rescalingBoundary_ * qp.weight() * integrationElement;

      auto bctype = problem.bctype(interface, qp.position());
      if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet) {
        // boundary value of the primary variable
        const auto gval = problem.g(interface, qp.position());
        auto& phi = cache_cg.evaluateFunction(insidepos,
                                              scalarspace.finiteElement().localBasis());

        if constexpr (type == NormalEquationType::classic) {
          for (size_type i=0; i<scalarspace.size(); i++)
            r_s.accumulate(scalarspace, i, factor *
                           parametrization_.rhs().theta(2) * -gval * phi[i]);
        }
        else if constexpr (type == NormalEquationType::adjoint) {
          const auto& normal = ig.unitOuterNormal(qp.position());
          fluxspace.finiteElement().localBasis().evaluateFunction(insidepos, tau);

          //transform flux functions using the Piola transform
          tautransformed = tau;

          // Primary boundary value
          // TODO: same rescaling factor?
          for (size_type j=0; j<fluxspace.size(); j++)
            r_s.accumulate(fluxspace, j, factor *
                           parametrization_.rhs().theta(2) *
                           -gval * tautransformed[j].dot(normal));

          // Additionally assemble boundary value of the adjoint variable
          const auto gvalAdjoint = problem.g(interface, qp.position());
          for (size_type i=0; i<scalarspace.size(); i++)
            r_s.accumulate(scalarspace, i, factor *
                           parametrization_.rhs().theta(0) * -gvalAdjoint * phi[i]);
        }
      }
      else if (bctype == Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann) {
        const auto jval = problem.j(interface, qp.position());

        if constexpr (type == NormalEquationType::classic) {
          const auto& normal = ig.unitOuterNormal(qp.position());
          fluxspace.finiteElement().localBasis().evaluateFunction(insidepos, tau);

          //transform flux functions using the Piola transform
          tautransformed = tau;

          for (size_type j=0; j<fluxspace.size(); j++)
            r_s.accumulate(fluxspace, j, factor *
                           parametrization_.rhs().theta(3) *
                           -jval * tautransformed[j].dot(normal));
        }
        else if constexpr (type == NormalEquationType::adjoint) {
          auto& phi = cache_cg.evaluateFunction(insidepos,
                                                scalarspace.finiteElement().localBasis());
          for (size_type i=0; i<scalarspace.size(); i++)
            r_s.accumulate(scalarspace, i, factor *
                           parametrization_.rhs().theta(3) * -jval * phi[i]);
        }
      }
    }
  }

private:
  const Problem& problem;
  const Parametrization& parametrization_;
  int intorder;
  const double rescalingBoundary_;

  CacheTypeCG cache_cg;
};

#endif // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORMAL_EQ_LOP_HH
