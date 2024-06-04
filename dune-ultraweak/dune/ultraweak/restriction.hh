#ifndef DUNE_ULTRAWEAK_RESTRICTION_HH
#define DUNE_ULTRAWEAK_RESTRICTION_HH

#include <dune/pdelab.hh>

namespace Dune {
  namespace Ultraweak {
    namespace Impl {
      template<typename LFS, typename CoefficientFieldType>
      class WrappedLocalFunction {
      protected:
        using FESwitch = Dune::FiniteElementInterfaceSwitch<
          typename LFS::Traits::FiniteElementType>;
        using BasisSwitch = Dune::BasisInterfaceSwitch<typename FESwitch::Basis>;
        using RangeType = typename BasisSwitch::Range;
      public:
        WrappedLocalFunction(const LFS& lfs,
                             const Dune::PDELab::LocalVector<CoefficientFieldType>& coeffs) :
          _lfs(lfs), _coeffs(coeffs) {}

        template<typename X>
        auto operator()(const X& x) const {
          std::vector<RangeType> values;
          _lfs.finiteElement().localBasis().evaluateFunction(x,values);
          RangeType result(0.0);
          for (std::size_t i=0; i<_lfs.size(); ++i)
            result += _coeffs(_lfs,i) * values[i];
          return result;
        }

      private:
        const LFS& _lfs;
        const Dune::PDELab::LocalVector<CoefficientFieldType>& _coeffs;
      };

      template<typename LocalCoeffs>
      class RestrictionVisitor :
        public Dune::TypeTree::DefaultPairVisitor,
        public Dune::TypeTree::DynamicTraversal,
        public Dune::TypeTree::VisitTree {
      public:
        RestrictionVisitor(const LocalCoeffs& localCoeffsHost_,
                           LocalCoeffs& localCoeffsSub_) :
          localCoeffsHost(localCoeffsHost_),
          localCoeffsSub(localCoeffsSub_) {}

        template<typename HostLFS, typename SubLFS, typename TreePath>
        void leaf(HostLFS& hostLfs, SubLFS& subLfs, TreePath treePath) const {
          // wrap coefficients into a function on the local element
          auto func = Impl::WrappedLocalFunction(hostLfs, localCoeffsHost);

          // interpolate into local subgrid function space
          using FESwitch = Dune::FiniteElementInterfaceSwitch<
            typename SubLFS::Traits::FiniteElementType>;
          using BasisSwitch = Dune::BasisInterfaceSwitch<typename FESwitch::Basis>;
          using DomainFieldType = typename BasisSwitch::DomainField;

          std::vector<DomainFieldType> coeffs;
          FESwitch::interpolation(subLfs.finiteElement())
            .interpolate(func, coeffs);
          for (std::size_t i=0; i<subLfs.size(); ++i)
            localCoeffsSub(subLfs,i) = coeffs[i];
        }

      private:
        // references to the full local coefficient vectors
        const LocalCoeffs& localCoeffsHost;
        LocalCoeffs& localCoeffsSub;
      };

      template<typename LFS>
      class WrappedLocalBasisFunction {
      protected:
        using FESwitch = Dune::FiniteElementInterfaceSwitch<
          typename LFS::Traits::FiniteElementType>;
        using BasisSwitch = Dune::BasisInterfaceSwitch<typename FESwitch::Basis>;
        using RangeType = typename BasisSwitch::Range;
      public:
        WrappedLocalBasisFunction(const LFS& lfs,
                             std::size_t basisFunctionIdx) :
          _lfs(lfs), _idx(basisFunctionIdx) {}

        template<typename X>
        auto operator()(const X& x) const {
          std::vector<RangeType> values;
          _lfs.finiteElement().localBasis().evaluateFunction(x,values);
          return values[_idx];
        }

      private:
        const LFS& _lfs;
        std::size_t _idx;
      };

      template<typename LocalMatrixType>
      class RestrictionAssemblyVisitor :
        public Dune::TypeTree::DefaultPairVisitor,
        public Dune::TypeTree::DynamicTraversal,
        public Dune::TypeTree::VisitTree {
      public:
        RestrictionAssemblyVisitor(LocalMatrixType& mat) : _mat(mat) {}

        template<typename HostLFS, typename SubLFS, typename TreePath>
        void leaf(HostLFS& hostLfs, SubLFS& subLfs, TreePath treePath) const {
          for (std::size_t i=0; i<hostLfs.size(); ++i) {
            // wrap hostgrid basis function into a callable
            auto hostBasisFunction = Impl::WrappedLocalBasisFunction(hostLfs, i);

            // interpolate into local subgrid function space
            using FESwitch = Dune::FiniteElementInterfaceSwitch<
              typename SubLFS::Traits::FiniteElementType>;
            using BasisSwitch = Dune::BasisInterfaceSwitch<typename FESwitch::Basis>;
            using DomainFieldType = typename BasisSwitch::DomainField;

            std::vector<DomainFieldType> coeffs;
            FESwitch::interpolation(subLfs.finiteElement())
              .interpolate(hostBasisFunction, coeffs);
            for (std::size_t j=0; j<subLfs.size(); ++j)
              _mat(hostLfs,i, subLfs,j) = coeffs[j];
          }
        }

      private:
        // reference to the local matrix
        LocalMatrixType& _mat;
      };
    } // namespace Impl

    // local operator to assemble the restriction matrix
    class SubgridRestriction :
      public Dune::PDELab::FullVolumePattern,
      public Dune::PDELab::LocalOperatorDefaultFlags
    {
    public:
      enum { doPatternVolume = true };
      enum { doAlphaVolume  = true };

      SubgridRestriction () {}

      template<typename EG, typename LFSU, typename X, typename LFSV, typename M>
      void jacobian_volume (const EG&, const LFSU& hostLfs, const X&, const LFSV& subLfs,
                            M & mat) const {
        auto restrict = Dune::Ultraweak::Impl::RestrictionAssemblyVisitor(mat.container());
        Dune::TypeTree::applyToTreePair(hostLfs, subLfs, restrict);
      }
    };
  } // namespace Ultraweak
} // namespace Dune

template<typename SubGridType,
         typename SubGFS, typename SubCoeffs,
         typename HostGFS, typename HostCoeffs>
class PDELabSubGridRestriction
{
public:
  PDELabSubGridRestriction(const SubGridType& _subgrid,
                           const SubGFS& _subGfs, SubCoeffs& _subCoeffs,
                           const HostGFS& _hostGfs, const HostCoeffs& _hostCoeffs) :
      subgrid(_subgrid),
      hostgrid(_subgrid.getHostGrid()),
      subCoeffs(_subCoeffs),
      hostCoeffs(_hostCoeffs),
      subLfs(_subGfs),
      hostLfs(_hostGfs),
      con(),
      subLfs_cache(subLfs,con,false),
      hostLfs_cache(hostLfs,con,false) {}

  // grid typedefs
  using SubElement = SubGridType::template Codim<0>::Entity;
  using HostGridType = typename SubGridType::HostGridType;
  using HostElement = HostGridType::template Codim<0>::Entity;

  // local space defs
  using LFSSub = Dune::PDELab::LocalFunctionSpace<SubGFS, Dune::PDELab::TrialSpaceTag>;
  using LFSHost = Dune::PDELab::LocalFunctionSpace<HostGFS, Dune::PDELab::TestSpaceTag>;

  // TODO: constraints are currently not being considered!
  using Constraints = Dune::PDELab::EmptyTransformation;
  using LFSSubCache = Dune::PDELab::LFSIndexCache<LFSSub,Constraints>;
  using LFSHostCache = Dune::PDELab::LFSIndexCache<LFSHost,Constraints>;

  // local vector
  using LocalCoeffsFieldType = double;
  using LocalCoeffs = Dune::PDELab::LocalVector<LocalCoeffsFieldType>;

  // views into global vectors
  using SubVectorView = SubCoeffs::template LocalView<LFSSubCache>;
  using HostVectorView = HostCoeffs::template ConstLocalView<LFSHostCache>;

  void pre() {
    subCoeffs *= 0.0;
    global_view_sub.attach(subCoeffs);
    global_view_host.attach(hostCoeffs);
  }

  void transfer(const SubElement& subElement, const HostElement& hostElement)
  {
    subLfs.bind(subElement);
    subLfs_cache.update();
    global_view_sub.bind(subLfs_cache);

    hostLfs.bind(hostElement);
    hostLfs_cache.update();
    global_view_host.bind(hostLfs_cache);

    localCoeffsHost.assign(hostLfs.size(), 0.0);
    localCoeffsSub.assign(subLfs.size(), 0.0);

    global_view_host.read(localCoeffsHost);

    auto restrict = Dune::Ultraweak::Impl::RestrictionVisitor(localCoeffsHost, localCoeffsSub);
    Dune::TypeTree::applyToTreePair(hostLfs, subLfs, restrict);

    global_view_sub.write(localCoeffsSub);
  }

  void post()
  {
    global_view_sub.commit();
    global_view_sub.detach();
    global_view_host.detach();
  }

private:
  const SubGridType& subgrid;
  const HostGridType& hostgrid;

  // the global coefficients
  SubCoeffs& subCoeffs;
  const HostCoeffs& hostCoeffs;

  // the local coefficients
  LocalCoeffs localCoeffsHost;
  LocalCoeffs localCoeffsSub;

  // views into the global vectors
  SubVectorView global_view_sub;
  HostVectorView global_view_host;

  LFSSub subLfs;
  LFSHost hostLfs;
  Constraints con;

  LFSSubCache subLfs_cache;
  LFSHostCache hostLfs_cache;
};

#endif  // DUNE_ULTRAWEAK_RESTRICTION_HH
