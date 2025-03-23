#ifdef NOT_LONGER_NEED_THIS
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Create classes for matrix-free operations for weighted tensors
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;
using Eigen::SparseMatrix;

template<int N> class MatrixFree;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Matrix-free class for weighted ALS
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Inherits traits of SparseMatrix (from example)
// Traits
namespace Eigen {
namespace internal {

template<> struct traits<MatrixFree<2>> :public Eigen::internal::traits<Eigen::SparseMatrix<double>>{};
template<> struct traits<MatrixFree<3>> :public Eigen::internal::traits<Eigen::SparseMatrix<double>>{};
template<> struct traits<MatrixFree<4>> :public Eigen::internal::traits<Eigen::SparseMatrix<double>>{};
template<> struct traits<MatrixFree<5>> :public Eigen::internal::traits<Eigen::SparseMatrix<double>>{};

}
}

template<int N> class MatrixFree : public Eigen::EigenBase<MatrixFree<N>> {

public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  
  Index rows() const { return nz_nA;}
  Index cols() const { return nz_nA;}
  
  template<typename Rhs>
  Eigen::Product<MatrixFree<N>,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixFree<N>,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }
  
  // Custom API:

  // Point pointers in right direction
  void Initialize(Eigen::MatrixXd * as[],
                  Eigen::VectorXd * z,
                  const Eigen::MatrixXi * cood, 
                  const Eigen::VectorXd * Wt,
                  int nz_na,
                  int whichA,
                  double Lambda) {
    
    // copy stuff over
    nz_nA = nz_na;
    As = as;
    Z = z;
    Cood = cood;
    wt = Wt;
    lambda= Lambda;
    length = Cood->cols();
    Nz = Z->size();
    A_it=whichA;
  }
  void UpdateLambda(double lambda_lhs) {
    lambda = lambda_lhs;
  }
  

public:
  Index nz_nA;
  Eigen::MatrixXd ** As;
  Eigen::VectorXd * Z;
  const Eigen::MatrixXi * Cood;
  const Eigen::VectorXd * wt;
  double lambda;
  int length;
  int Nz;
  int A_it;

};



// Implementation of MatrixFree * Eigen::DenseVector though a specialization of internal::generic_product_impl:
// Mostly copied from example on Eigen website
namespace Eigen {
namespace internal {

// Specialization for 2D tensors
template<typename Rhs>
struct generic_product_impl<MatrixFree<2>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MatrixFree<2>,Rhs,generic_product_impl<MatrixFree<2>,Rhs> >
{
  typedef typename Product<MatrixFree<2>,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixFree<2>& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    
    // copy some stuff over
    constexpr int DIM = 2;
    int dm = lhs.A_it;
    int length = lhs.length;
    int Nz = lhs.Nz;
    double lambda = lhs.lambda;
    Eigen::VectorXd * Z = lhs.Z;
    const Eigen::MatrixXi * Cood = lhs.Cood;
    const Eigen::VectorXd * wt = lhs.wt;
    // temp
    Eigen::VectorXd halfway(length);
    
    // get indices
    int ci = (dm+0)%DIM;
    int cj = (dm+1)%DIM;

    
    
    // get matrices
    Eigen::MatrixXd * Aj = lhs.As[cj];

    // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
    dst = alpha*lambda*rhs;
    // The weighted MTTKRP part (first half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      double temp=0;
      for(int z=0;z<Nz;z++){
        temp += (*Aj)(z,jay)*(*Z)(z)*rhs(Iind+z);
      }
      halfway(i) = (*wt)(i)*temp;
    }
    
    // The weighted MTTKRP part (second half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      double forward = alpha*halfway(i);
      for(int z=0;z<Nz;z++){
        dst(Iind+z) += (*Aj)(z,jay)*(*Z)(z)*forward;
      }
    }
    
  }
  
};

// Specialization for 3D tensors
template<typename Rhs>
struct generic_product_impl<MatrixFree<3>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MatrixFree<3>,Rhs,generic_product_impl<MatrixFree<3>,Rhs> >
{
  typedef typename Product<MatrixFree<3>,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixFree<3>& lhs, const Rhs& rhs, const Scalar& alpha)
  {
   
  // copy some stuff over
  constexpr int DIM = 3;
  int dm = lhs.A_it;
  int length = lhs.length;
  int Nz = lhs.Nz;
  double lambda = lhs.lambda;
  Eigen::VectorXd * Z = lhs.Z;
  const Eigen::MatrixXi * Cood = lhs.Cood;
  const Eigen::VectorXd * wt = lhs.wt;
  // temp
  Eigen::VectorXd halfway(length);
  
  // get indices
  int ci = (dm+0)%DIM;
  int cj = (dm+1)%DIM;
  int ck = (dm+2)%DIM;


  // get matrices
  Eigen::MatrixXd * Aj = lhs.As[cj];
  Eigen::MatrixXd * Ak = lhs.As[ck];
  
  // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
  dst = alpha*lambda*rhs;
  // The weighted MTTKRP part (first half)
  for(int i=0;i<length;i++){
    int Iind = (*Cood)(ci,i)*Nz;
    int jay = (*Cood)(cj,i);
    int kay = (*Cood)(ck,i);
    double temp=0;
    for(int z=0;z<Nz;z++){
      temp += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Z)(z)*rhs(Iind+z);
    }
    halfway(i) = (*wt)(i)*temp;
  }
  
  // The weighted MTTKRP part (second half)
  for(int i=0;i<length;i++){
    int Iind = (*Cood)(ci,i)*Nz;
    int jay = (*Cood)(cj,i);
    int kay = (*Cood)(ck,i);
    double forward = alpha*halfway(i);
    for(int z=0;z<Nz;z++){
      dst(Iind+z) += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Z)(z)*forward;
    }
  }
}

};

// Specialization for 4D tensors
template<typename Rhs>
struct generic_product_impl<MatrixFree<4>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MatrixFree<4>,Rhs,generic_product_impl<MatrixFree<4>,Rhs> >
{
  typedef typename Product<MatrixFree<4>,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixFree<4>& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    
    // copy some stuff over
    constexpr int DIM = 4;
    int dm = lhs.A_it;
    int length = lhs.length;
    int Nz = lhs.Nz;
    double lambda = lhs.lambda;
    Eigen::VectorXd * Z = lhs.Z;
    const Eigen::MatrixXi * Cood = lhs.Cood;
    const Eigen::VectorXd * wt = lhs.wt;
    // temp
    Eigen::VectorXd halfway(length);
    

    // get indices
    int ci = (dm+0)%DIM;
    int cj = (dm+1)%DIM;
    int ck = (dm+2)%DIM;
    int cl = (dm+3)%DIM;
    
    
    // get matrices
    Eigen::MatrixXd * Aj = lhs.As[cj];
    Eigen::MatrixXd * Ak = lhs.As[ck];
    Eigen::MatrixXd * Al = lhs.As[cl];
    
    // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
    dst = alpha*lambda*rhs;
    // The weighted MTTKRP part (first half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      int kay = (*Cood)(ck,i);
      int ell = (*Cood)(cl,i);
      double temp=0;
      for(int z=0;z<Nz;z++){
        temp += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Al)(z,ell)*(*Z)(z)*rhs(Iind+z);
      }
      halfway(i) = (*wt)(i)*temp;
    }
    
    // The weighted MTTKRP part (second half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      int kay = (*Cood)(ck,i);
      int ell = (*Cood)(cl,i);
      double forward = alpha*halfway(i);
      for(int z=0;z<Nz;z++){
        dst(Iind+z) += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Al)(z,ell)*(*Z)(z)*forward;
      }
    }
  }
};


// Specialization for 5D tensors
template<typename Rhs>
struct generic_product_impl<MatrixFree<5>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MatrixFree<5>,Rhs,generic_product_impl<MatrixFree<5>,Rhs> >
{
  typedef typename Product<MatrixFree<5>,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MatrixFree<5>& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    
    // copy some stuff over
    constexpr int DIM = 5;
    int dm = lhs.A_it;
    int length = lhs.length;
    int Nz = lhs.Nz;
    double lambda = lhs.lambda;
    Eigen::VectorXd * Z = lhs.Z;
    const Eigen::MatrixXi * Cood = lhs.Cood;
    const Eigen::VectorXd * wt = lhs.wt;
    // temp
    Eigen::VectorXd halfway(length);
    

    // get indices
    int ci = (dm+0)%DIM;
    int cj = (dm+1)%DIM;
    int ck = (dm+2)%DIM;
    int cl = (dm+3)%DIM;
    int cm = (dm+4)%DIM;
    
    
    // get matrices
    Eigen::MatrixXd * Aj = lhs.As[cj];
    Eigen::MatrixXd * Ak = lhs.As[ck];
    Eigen::MatrixXd * Al = lhs.As[cl];
    Eigen::MatrixXd * Am = lhs.As[cm];
    
    // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
    dst = alpha*lambda*rhs;
    // The weighted MTTKRP part (first half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      int kay = (*Cood)(ck,i);
      int ell = (*Cood)(cl,i);
      int emm = (*Cood)(cm,i);
      double temp=0;
      for(int z=0;z<Nz;z++){
        temp += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Al)(z,ell)*(*Am)(z,emm)*(*Z)(z)*rhs(Iind+z);
      }
      halfway(i) = (*wt)(i)*temp;
    }
    
    // The weighted MTTKRP part (second half)
    for(int i=0;i<length;i++){
      int Iind = (*Cood)(ci,i)*Nz;
      int jay = (*Cood)(cj,i);
      int kay = (*Cood)(ck,i);
      int ell = (*Cood)(cl,i);
      int emm = (*Cood)(cm,i);
      double forward = alpha*halfway(i);
      for(int z=0;z<Nz;z++){
        dst(Iind+z) += (*Aj)(z,jay)*(*Ak)(z,kay)*(*Al)(z,ell)*(*Am)(z,emm)*(*Z)(z)*forward;
      }
    }
  }
};

}


}




#endif














