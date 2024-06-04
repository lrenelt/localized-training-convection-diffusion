#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORM_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORM_HH

template<typename ScalarFEM>
class HdivH1norm :
  public Dune::PDELab::FullVolumePattern,
  public Dune::PDELab::LocalOperatorDefaultFlags
{
public:
  enum { doPatternVolume = true };
  enum { doAlphaVolume = true };

  using ScalarLocalBasisType = typename
    ScalarFEM::Traits::FiniteElementType::Traits::LocalBasisType;
  using ScalarCacheType = Dune::PDELab::LocalBasisCache<ScalarLocalBasisType>;


  HdivH1norm(const std::size_t intorder = 2) : intorder_(intorder) {}

  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume(const EG& eg, const LFSU& lfsu,
                    const X& x, const LFSV& lfsv, R& r) const
  {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using ScalarSpace = typename LFSU::template Child<1>::Type;

    using RF = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeFieldType;
    using size_type = typename FluxSpace::Traits::SizeType;

    using FluxRangeType = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using ScalarRangeType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeType;
    using ScalarJacobianType = typename ScalarSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::JacobianType;

    const int dim = EG::Geometry::mydimension;

    // select the subspaces
    const auto& fluxspace = lfsu.template child<0>();
    const auto& scalarspace = lfsu.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = eg.geometry();

    FluxRangeType sigma;
    ScalarRangeType divsigma;

    std::vector<FluxRangeType> tau(fluxspace.size());
    std::vector<FluxRangeType> tautransformed(fluxspace.size());
    auto gradtau = Dune::PDELab::makeJacobianContainer(fluxspace);
    auto gradtautransformed = Dune::PDELab::makeJacobianContainer(fluxspace);
    std::vector<ScalarRangeType> divtau(fluxspace.size());

    ScalarRangeType u;
    ScalarJacobianType gradu;

    std::vector<ScalarRangeType> phi(scalarspace.size());
    auto gradphi = Dune::PDELab::makeJacobianContainer(scalarspace);

    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      const auto S = geo.jacobianInverseTransposed(qp.position());
      factor = qp.weight() * geo.integrationElement(qp.position());

      // assemble Hdiv-part of the norm:
      fluxspace.finiteElement().localBasis().evaluateFunction(qp.position(),tau);
      fluxspace.finiteElement().localBasis().evaluateJacobian(qp.position(),gradtau);

      // transform flux functions using the Piola transformation
      tautransformed = tau;

      // transform gradient
      for (size_type i=0; i<fluxspace.size(); i++)
        for (size_type k=0; k<dim; k++)
          S.mv(gradtau[i][k], gradtautransformed[i][k]);

      // compute divergence
      for (size_type i=0; i<fluxspace.size(); i++) {
        divtau[i] = 0.0;
        for (int k=0; k<dim; k++)
          divtau[i] += gradtautransformed[i][k][k];
      }

      // compute sigma and div sigma
      sigma = FluxRangeType(0.0);
      divsigma = 0.0;
      for (size_type i=0; i<fluxspace.size(); i++) {
        sigma += x(fluxspace,i), tau[i];
        divsigma.axpy(x(fluxspace,i), divtau[i]);
      }

      for (size_type j=0; j<fluxspace.size(); j++)
        r.accumulate(fluxspace, j, factor * (sigma.dot(tau[j]) + divsigma*divtau[j]));

      // assemble H1-part of the norm:
      phi = cache.evaluateFunction(qp.position(), scalarspace.finiteElement().localBasis());

      auto& js = cache.evaluateJacobian(qp.position(), scalarspace.finiteElement().localBasis());
      for (size_type i=0; i<scalarspace.size(); ++i)
        S.mv(js[i][0], gradphi[i][0]);

      // compute u and grad u
      u = 0.0;
      gradu = ScalarJacobianType(0.0);
      for (size_type i=0; i<scalarspace.size(); i++) {
        u += x(scalarspace,i)*phi[i];
        gradu.axpy(x(scalarspace,i),gradphi[i]);
      }

      for (size_type j=0; j<scalarspace.size(); j++)
        r.accumulate(scalarspace, j, factor * (gradu[0].dot(gradphi[j][0]) + u * phi[j]));
    }
  }

  template<typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG& eg, const LFSU& lfsu,
                       const X& x, const LFSV& lfsv, M& mat) const
  {
    using FluxSpace = typename LFSU::template Child<0>::Type;
    using ScalarSpace = typename LFSU::template Child<1>::Type;

    using RF = typename FluxSpace::Traits::FiniteElementType::
      Traits::LocalBasisType::Traits::RangeFieldType;
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

    std::vector<FluxRangeType> tau(fluxspace_u.size());
    std::vector<FluxRangeType> tautransformed(fluxspace_u.size());
    auto gradtau = Dune::PDELab::makeJacobianContainer(fluxspace_u);
    auto gradtautransformed = Dune::PDELab::makeJacobianContainer(fluxspace_u);
    std::vector<ScalarRangeType> divtau(fluxspace_u.size());

    std::vector<ScalarRangeType> phi(scalarspace_u.size());
    auto gradphi = Dune::PDELab::makeJacobianContainer(scalarspace_u);

    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      const auto S = geo.jacobianInverseTransposed(qp.position());
      factor = qp.weight() * geo.integrationElement(qp.position());

      // assemble Hdiv-part of the norm:
      fluxspace_u.finiteElement().localBasis().evaluateFunction(qp.position(),tau);
      fluxspace_u.finiteElement().localBasis().evaluateJacobian(qp.position(),gradtau);

      // transform flux functions using the Piola transformation
      tautransformed = tau;

      // transform gradient
      for (size_type i=0; i<fluxspace_u.size(); i++)
        for (size_type k=0; k<dim; k++)
          S.mv(gradtau[i][k], gradtautransformed[i][k]);

      // compute divergence
      for (size_type i=0; i<fluxspace_u.size(); i++) {
        divtau[i] = 0.0;
        for (int k=0; k<dim; k++)
          divtau[i] += gradtautransformed[i][k][k];
      }

      for (size_type i=0; i<fluxspace_u.size(); i++)
        for (size_type j=0; j<fluxspace_v.size(); j++)
          mat.accumulate(fluxspace_v, j, fluxspace_u, i, factor *
                         (tau[i].dot(tau[j]) + divtau[i] * divtau[j]));

      // assemble H1-part of the norm:
      phi = cache.evaluateFunction(qp.position(), scalarspace_u.finiteElement().localBasis());

      auto& js = cache.evaluateJacobian(qp.position(), scalarspace_u.finiteElement().localBasis());
      for (size_type i=0; i<scalarspace_u.size(); ++i)
        S.mv(js[i][0], gradphi[i][0]);

      for (size_type i=0; i<scalarspace_u.size(); i++)
        for (size_type j=0; j<scalarspace_v.size(); j++)
          mat.accumulate(scalarspace_v, j, scalarspace_u, i,  factor *
                         (gradphi[i][0].dot(gradphi[j][0]) + phi[i]*phi[j]));
    }
  }

  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void jacobian_apply_volume (const EG& eg, const LFSU& lfsu,
                              const X& z, const LFSV& lfsv,
                              R& r) const {
    alpha_volume(eg,lfsu,z,lfsv,r);
  }

private:
  const std::size_t intorder_;
  ScalarCacheType cache;
};



#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_NORM_HH
