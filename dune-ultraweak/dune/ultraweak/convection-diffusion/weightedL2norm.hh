#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_WEIGHTEDL2NORM_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_WEIGHTEDL2NORM_HH

template<typename ScalarFEM, typename WeightingFunction>
class WeightedL2Norm :
  public Dune::PDELab::FullVolumePattern,
  public Dune::PDELab::LocalOperatorDefaultFlags
{
public:
  enum { doPatternVolume = true };
  enum { doAlphaVolume = true };

  using ScalarLocalBasisType = typename
    ScalarFEM::Traits::FiniteElementType::Traits::LocalBasisType;
  using ScalarCacheType = Dune::PDELab::LocalBasisCache<ScalarLocalBasisType>;

  // using DomainType = ScalarLocalBasisType::Traits::DomainType;
  // using WeightingFunction = std::function<MatrixType(const DomainType&)>;

  WeightedL2Norm(WeightingFunction Afunc,
                 const std::size_t intorder = 2)
    : Afunc_(Afunc), intorder_(intorder) {}

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

    // select the subspaces
    const auto& fluxspace = lfsu.template child<0>();
    const auto& scalarspace = lfsu.template child<1>();

    // get Jacobian and determinant
    // assume the transformation is linear
    const auto geo = eg.geometry();
    const auto& cell = eg.entity();

    FluxRangeType sigma;
    std::vector<FluxRangeType> tau(fluxspace.size());
    std::vector<FluxRangeType> tautransformed(fluxspace.size());
    std::vector<FluxRangeType> weightedTau(fluxspace.size());

    ScalarRangeType u;
    std::vector<ScalarRangeType> phi(scalarspace.size());

    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      factor = qp.weight() * geo.integrationElement(qp.position());

      fluxspace.finiteElement().localBasis().evaluateFunction(qp.position(),tau);

      // TODO: transform flux functions using the Piola transformation
      tautransformed = tau;

      const auto weight = Afunc_(cell, qp.position());
      for (size_type i=0; i<fluxspace.size(); i++)
        weight.mv(tautransformed[i], weightedTau[i]);

      // compute sigma
      sigma = FluxRangeType(0.0);
      for (size_type i=0; i<fluxspace.size(); i++)
        sigma += x(fluxspace,i) * weightedTau[i];

      for (size_type j=0; j<fluxspace.size(); j++)
        r.accumulate(fluxspace, j, factor * sigma.dot(weightedTau[j]));

      phi = cache.evaluateFunction(qp.position(), scalarspace.finiteElement().localBasis());
      // compute u
      u = 0.0;
      for (size_type i=0; i<scalarspace.size(); i++)
        u += x(scalarspace,i)*phi[i];

      for (size_type j=0; j<scalarspace.size(); j++)
        r.accumulate(scalarspace, j, factor * u * phi[j] );
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

    // select the subspaces
    const auto& fluxspace_u = lfsu.template child<0>();
    const auto& scalarspace_u = lfsu.template child<1>();
    const auto& fluxspace_v = lfsv.template child<0>();
    const auto& scalarspace_v = lfsv.template child<1>();

    const auto geo = eg.geometry();
    const auto& cell = eg.entity();

    std::vector<FluxRangeType> tau(fluxspace_u.size());
    std::vector<FluxRangeType> tautransformed(fluxspace_u.size());
    std::vector<FluxRangeType> weightedTau(fluxspace_u.size());
    std::vector<ScalarRangeType> phi(scalarspace_u.size());

    RF factor;

    for (const auto& qp: quadratureRule(geo, intorder_)) {
      factor = qp.weight() * geo.integrationElement(qp.position());

      fluxspace_u.finiteElement().localBasis().evaluateFunction(qp.position(),tau);

      // transform flux functions using the Piola transformation
      tautransformed = tau;

      const auto weight = Afunc_(cell, qp.position());
      for (size_type i=0; i<fluxspace_u.size(); i++)
        weight.mv(tautransformed[i], weightedTau[i]);

      for (size_type i=0; i<fluxspace_u.size(); i++)
        for (size_type j=0; j<fluxspace_v.size(); j++)
          mat.accumulate(fluxspace_v, j, fluxspace_u, i,
                         factor * weightedTau[i].dot(weightedTau[j]) );

      phi = cache.evaluateFunction(qp.position(), scalarspace_u.finiteElement().localBasis());

      for (size_type i=0; i<scalarspace_u.size(); i++)
        for (size_type j=0; j<scalarspace_v.size(); j++)
          mat.accumulate(scalarspace_v, j, scalarspace_u, i,
                         factor * phi[i]*phi[j] );
    }
  }

  template<typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void jacobian_apply_volume (const EG& eg, const LFSU& lfsu,
                              const X& z, const LFSV& lfsv,
                              R& r) const {
    alpha_volume(eg,lfsu,z,lfsv,r);
  }

private:
  WeightingFunction Afunc_;
  const std::size_t intorder_;
  ScalarCacheType cache;
};

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_WEIGHTEDL2NORM_HH
