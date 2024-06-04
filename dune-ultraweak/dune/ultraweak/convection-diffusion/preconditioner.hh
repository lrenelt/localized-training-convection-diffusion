// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PRECONDITIONER_HH
#define DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PRECONDITIONER_HH

#include <dune/istl/solverregistry.hh>

namespace Dune {

  template<typename M, typename X, typename Y>
  class BlockJacobiAMG : public Dune::Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

  private:
    using MatrixBlock = typename M::block_type;
    using XBlock = typename X::block_type;
    using YBlock = typename Y::block_type;

    using InverseOperatorType = Dune::SeqILU<MatrixBlock,XBlock,YBlock>;

    using rowIt = M::ConstRowIterator;
    using colIt = M::ConstColIterator;

    static constexpr int N = 2;

  public:
    BlockJacobiAMG(const M& A, const int n, const real_field_type relaxation) :
      A_(A), n_(n), relaxation_(relaxation)
    {
      // initialize the block inversions
      for (std::size_t i=0; i<A_.N(); ++i) {
        try {
          blockInv[i] = std::make_shared<InverseOperatorType>(A_[i][i], 1.0);
        }
        catch (Dune::BCRSMatrixError& e) {
          std::cerr << "Diagonal block with index " << i << " is empty!" << std::endl;
        }
      }
    }

    BlockJacobiAMG(const M& A, const ParameterTree& configuration)
      : BlockJacobiAMG(A, configuration.get<int>("iterations",1),
               configuration.get<real_field_type>("relaxation",1.0)) {}

    BlockJacobiAMG(const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A,
                   const ParameterTree& configuration)
      : BlockJacobiAMG(A->getmat(), configuration) {}

    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b) {}

    virtual void apply(X& v, const Y& d) {
      for(int i=0; i<n_; i++)
        jacIteration(v,d);
    }

    virtual void post ([[maybe_unused]] X& x) {}

    virtual SolverCategory::Category category() const {
      return SolverCategory::sequential;
    }

  private:
    // standard Jacobi-iteration that uses the AMG for the diagonal block inversion
    void jacIteration(X& x, const Y& d) const {
      X update(x);
      for (std::size_t i=0; i<A_.N(); ++i)
        blockInv[i]->apply(update[i],d[i]);
      x.axpy(relaxation_,update);
    }

    const M& A_;
    const int n_;
    const real_field_type relaxation_;

    std::array<std::shared_ptr<InverseOperatorType>,N> blockInv;
  };


  DUNE_REGISTER_PRECONDITIONER("jacobiAMG",
                               defaultPreconditionerCreator<BlockJacobiAMG>());
}

#endif  // DUNE_ULTRAWEAK_CONVECTION_DIFFUSION_PRECONDITIONER_HH
