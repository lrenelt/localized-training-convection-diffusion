#ifndef DUNE_ULTRAWEAK_INNER_PRODUCT_ASSEMBLER_HH
#define DUNE_ULTRAWEAK_INNER_PRODUCT_ASSEMBLER_HH

#include <dune/pdelab.hh>

template<typename ProductLOP, typename GFS>
class InnerProductAssembler {
  static constexpr int dim = 2; //TODO: deduce
  using RF = double; //TODO: deduce
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;

  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<
    GFS, GFS, ProductLOP, MBE, RF, RF, RF, CC, CC>;
  using MatrixType = Dune::PDELab::Backend::Native<typename GO::Jacobian>;

public:
  InnerProductAssembler(ProductLOP& lop, const GFS& gfs)
    : lop_(lop), gfs_(gfs), isAssembled_(false) {
    MBE mbe(1<<(dim+1));
    go_ = std::make_shared<GO>(gfs_, cc_, gfs_, cc_, lop_, mbe);
  }

  void assemble() {
    typename GO::Domain tempEvalPoint(gfs_, 0.0);
    typename GO::Jacobian wrappedMat(*go_, 0.0);
    go_->jacobian(tempEvalPoint, wrappedMat);
    mat_ = Dune::PDELab::Backend::native(wrappedMat);
    isAssembled_ = true;
  }

  const MatrixType& getAssembledMatrix() {
    if (!isAssembled_)
      assemble();
    return mat_;
  }

protected:
  CC cc_;
  ProductLOP& lop_;
  const GFS& gfs_;
  std::shared_ptr<GO> go_;
  MatrixType mat_;
  bool isAssembled_;
};

#endif  // DUNE_ULTRAWEAK_INNER_PRODUCT_ASSEMBLER_HH
