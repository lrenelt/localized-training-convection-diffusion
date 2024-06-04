#ifndef DUNE_ULTRAWEAK_DISCRETEGRIDFUNCTIONDIVERGENCE_HH
#define DUNE_ULTRAWEAK_DISCRETEGRIDFUNCTIONDIVERGENCE_HH

#include <vector>

#include <dune/pdelab.hh>

namespace Dune {
  namespace PDELab {

    template<typename GV, typename RangeFieldType, int dimR>
    struct DiscreteGridFunctionDivergenceTraits
      : public GridFunctionTraits<GV,RangeFieldType,dimR,
                                  FieldVector<RangeFieldType,dimR>> {};

    /** \brief DiscreteGridFunction for divergence with Piola transformation
     *
     * \copydetails DiscreteGridFunction
     */
    template<typename GFS, typename X>
    class DiscreteGridFunctionDivergence : public GridFunctionBase<
      DiscreteGridFunctionDivergenceTraits<
        typename GFS::Traits::GridViewType,
        typename GFS::Traits::FiniteElementType::Traits::LocalBasisType
          ::Traits::RangeFieldType,
        1>,
      DiscreteGridFunctionDivergence<GFS,X>>
    {
    private:
      using GV = typename GFS::Traits::GridViewType;
      using LBTraits = typename GFS::Traits::FiniteElementType::Traits::LocalBasisType::Traits;
      using RF = typename LBTraits::RangeFieldType;

      static const int dim = GV::dimension;
      static const int dimRange = 1;

    public:
      using Traits = DiscreteGridFunctionDivergenceTraits<GV,RF,dimRange>;

    private:
      using BaseT = GridFunctionBase<Traits, DiscreteGridFunctionDivergence<GFS,X>>;

    public:
      /** \brief Construct a DiscreteGridFunctionDivergence
       *
       * \copydetails DiscreteGridFunction::DiscreteGridFunction(const GFS&, const X&)
       */
      DiscreteGridFunctionDivergence(const GFS& gfs, const X& x_) :
        pgfs(stackobject_to_shared_ptr(gfs)),
        lfs(gfs), lfs_cache(lfs), x_view(x_),
        xl(pgfs->maxLocalSize()) {}

      /** \brief Construct a DiscreteGridFunctionDivergence
       *
       * \param gfs shared pointer to the GridFunctionsSpace
       * \param x_  shared pointer to the coefficients vector
       */
      DiscreteGridFunctionDivergence(std::shared_ptr<const GFS> gfs, std::shared_ptr<const X> x_) :
        pgfs(gfs), lfs(*gfs), lfs_cache(lfs), x_view(*x_),
        xl(pgfs->maxLocalSize()) {}

      inline void evaluate (const typename Traits::ElementType& e,
                            const typename Traits::DomainType& x,
                            typename Traits::RangeType& y) const
      {
        // bind local function space
        lfs.bind(e);
        lfs_cache.update();
        x_view.bind(lfs_cache);

        // get loal coefficients
        x_view.read(xl);
        x_view.unbind();

        // get local Jacobians of the shape functions
        std::vector<typename LBTraits::JacobianType> J(lfs.size());
        lfs.finiteElement().localBasis().evaluateJacobian(x,J);

        // get geometry Jacobian
        const auto jacIT = e.geometry().jacobianInverseTransposed(x);

        y = 0.0;
        typename LBTraits::JacobianType gradphi;
        for (unsigned int i=0; i<xl.size(); i++) {
          gradphi = 0;
          jacIT.umv(J[i], gradphi);
          for (unsigned int k=0; k<dim; k++) //divergence of i-th shape function
            y += xl[i]*gradphi[k][k];
        }
      }

      //! get a reference to the GridView
      inline const typename Traits::GridViewType& getGridView () const
      {
        return pgfs->gridView();
      }

    private:
      using LFS = LocalFunctionSpace<GFS>;
      using LFSCache = LFSIndexCache<LFS>;
      using XView = typename X::template ConstLocalView<LFSCache>;

      std::shared_ptr<const GFS> pgfs;
      mutable LFS lfs;
      mutable LFSCache lfs_cache;
      mutable XView x_view;
      mutable std::vector<typename Traits::RangeFieldType> xl;
    };
  } // namespace PDELab
} // namespace Dune

#endif  // DUNE_ULTRAWEAK_DISCRETEGRIDFUNCTIONDIVERGENCE_HH
