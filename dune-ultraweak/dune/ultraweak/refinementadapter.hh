// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ULTRAWEAK_REFINEMENTADAPTER_HH
#define DUNE_ULTRAWEAK_REFINEMENTADAPTER_HH

/*! \brief Adapter evaluating a grid function on a refined grid view

  \tparam GF  the grid function type
  \tparam GV  the refined grid view type
*/
template<typename GF, typename GV>
class RefinementAdapter : public Dune::PDELab::GridFunctionBase<
  Dune::PDELab::GridFunctionTraits<GV,
                                   typename GF::Traits::RangeFieldType,
                                   GF::Traits::dimRange,
                                   typename GF::Traits::RangeType>,
  RefinementAdapter<GF, GV> >
{
public:
  using Traits = Dune::PDELab::GridFunctionTraits<GV,typename GF::Traits::RangeFieldType,
                                                  GF::Traits::dimRange,
                                                  typename GF::Traits::RangeType>;

  //! constructor
  RefinementAdapter (const GF& gf_, const GV& fine_gv_, int refinements_) : gf(gf_), fine_gv(fine_gv_), refinements(refinements_) {}

  //! \copydoc GridFunctionBase::evaluate()
  inline void evaluate (const typename Traits::ElementType& e,
                        const typename Traits::DomainType& x,
                        typename Traits::RangeType& y) const
  {
    typename Traits::ElementType coarseElement = e;
    for (int i=0; i<refinements; i++)
      coarseElement = coarseElement.father();
    const auto xcoarse = coarseElement.geometry().local(e.geometry().global(x));
    gf.evaluate(coarseElement,xcoarse,y);
  }

  inline const typename Traits::GridViewType& getGridView () const
  {
    return fine_gv;
  }

private:
  const GF& gf;
  const GV& fine_gv;
  const int refinements;
};

#endif // DUNE_ULTRAWEAK_REFINEMENTADAPTER_HH
