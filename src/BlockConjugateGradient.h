//https://math.aalto.fi/opetus/inv/CGalgorithm.pdf
#ifndef BLOCK_CGRADIENT_HEADER
#define BLOCK_CGRADIENT_HEADER


//#include "set_debug.h"
#include <RcppEigen.h>
#include <math.h> 

template<int N> class BlockConjugateGradient{
  
  
public:
  
  BlockConjugateGradient<N>(){};
  BlockConjugateGradient<N>(double nz,double na,
                  Eigen::VectorXd * Halfway,
                  double whichA,
                  double tol) {

    // copy stuff over
    nZ = nz;
    nA = na;
    halfway=Halfway;
    epsilon = tol;

    // initialize temporary variables
    Rk = Eigen::MatrixXd::Zero(nz,nA);
    Pk = Eigen::MatrixXd::Zero(nz,nA);
    Apk = Eigen::MatrixXd::Zero(nz,nA);
    rTr = Eigen::ArrayXd(nA);
    beta = Eigen::ArrayXd(nA);
    alpha = Eigen::ArrayXd(nA);
    skip = Eigen::ArrayXd::Ones(nA);
  };
  
  
  
  void Solve(Eigen::MatrixXd & RHS,
             Eigen::MatrixXd (& A)[N],
             Eigen::VectorXd & Z,
             const Eigen::MatrixXi & coord, 
             const Eigen::VectorXd & wt,
             int dm,double lambda) {
    
    Eigen::MatrixXd & X = A[dm];

    // First step
    this->SparseMatrixProduct(Apk,X,A,Z,coord,wt,dm,lambda);
    Rk = RHS - Apk;
    Pk = Rk;
    double relative = 0.5*(RHS.squaredNorm()+Rk.squaredNorm())/double(this->nA); // don't want this to be zero if outer iteration is almost converged
    rTr = Rk.colwise().squaredNorm().array();
    
    // conjugate gradient iterations
    for(int k=0;k<(nZ+2);k++){
      this->SparseMatrixProduct(Apk,Pk,A,Z,coord,wt,dm,lambda);  // calculate A*p_k
      beta = (Pk.cwiseProduct(Apk)).colwise().sum();  // p_k^T*A*p_k
      alpha = rTr*beta.inverse(); // (r_k^T*r_k)/(p_k^T*A*p_k)
      this->removeNA(alpha,0.0); // if coordinate has converged, might have NaN, remove
      X += Pk*alpha.matrix().asDiagonal();  //x_{k+1} = x_k + alpha*p_k
      Rk -= Apk*alpha.matrix().asDiagonal(); //r_{k+1} = r_k - alpha*A*p_k
      
      beta = rTr.inverse().eval(); //  1/(r_k^T*r_k)
      rTr = Rk.colwise().squaredNorm().array();  // r_{k+1}^T*r_{k+1}
      if(rTr.maxCoeff() <= this->epsilon*relative){
        break; // stop early
      }
      this->removeNA(beta,1.0); // if coordinate has converged, might have NaN, remove
      beta = rTr*beta; // beta =  (r_{k+1}^T*r_{k+1})/(r_k^T*r_k)
      Pk.noalias() = Rk + Pk*beta.matrix().asDiagonal(); //p_k = r_{k+1} + beta*p_k
    }
  };
  

private:
  // Variables
  int nZ;
  int nA;
  int dm;
  double lambda;
  double epsilon;



  Eigen::VectorXd * halfway;

  Eigen::MatrixXd * RHS;
  Eigen::MatrixXd Rk;
  Eigen::MatrixXd Pk;
  Eigen::MatrixXd Apk;
  Eigen::ArrayXd beta;
  Eigen::ArrayXd alpha;
  Eigen::ArrayXd rTr;
  Eigen::ArrayXd skip;
  
  // Functions
  
  // Safe inverse (doesn't check they are same size)
  inline void removeNA(Eigen::ArrayXd & In, double replace){
    auto ptrIn = In.data(); 
    for (auto i = 0; i < In.size(); i++){
      auto ptr = ptrIn+i; 
      *ptr  = (isnan(*ptr)|isinf(*ptr))?replace:*ptr;
    }
  }
 
  
  
  // Template matrix multiplication (Warning use care when multiplying an A matrix)
  void SparseMatrixProduct(Eigen::MatrixXd & Out,Eigen::MatrixXd & In,
                           Eigen::MatrixXd (& A)[N],Eigen::VectorXd & Z,
                           const Eigen::MatrixXi & Coord, 
                           const Eigen::VectorXd & wt,
                           int dm,double lambda){};

};

// Template declaration of functions
  template<> void BlockConjugateGradient<2>::SparseMatrixProduct(Eigen::MatrixXd & Out,Eigen::MatrixXd & In,
                                                                 Eigen::MatrixXd (& A)[2],Eigen::VectorXd & Z,
                                                                 const Eigen::MatrixXi & coord, 
                                                                 const Eigen::VectorXd & wt,
                                                                 int dm,double lambda){
    // get indices
    constexpr int DIM = 2;
    int ci = (dm + 0)%DIM;
    int cj = (dm + 1)%DIM;
    
    // get matrices
      // Eigen::MatrixXd & Ai = A[ci];
    Eigen::MatrixXd & Aj = A[cj];
    
    auto length = wt.size();
    
    // The lambda0*||x||^2+lambdaK*||x-x_x||^2 part
    Out = lambda*In;

    // The weighted MTTKRP part (first half)
    for(auto u=0;u<length;u++){
      int eye = coord(ci,u);
      int jay = coord(cj,u);
      double temp = 0;
      for(int z=0;z<nZ;z++){
        temp += In(z,eye)*Aj(z,jay)*Z(z);
      }
      (*halfway)(u) = wt(u)*temp;
    }
    // The weighted MTTKRP part (second half)
    for(int u=0;u<length;u++){
      int eye = coord(ci,u);
      int jay = coord(cj,u);
      double forward = (*halfway)(u);
      for(int z=0;z<nZ;z++){
        Out(z,eye) += Aj(z,jay)*Z(z)*forward;
      }
  }
}
template<> void BlockConjugateGradient<3>::SparseMatrixProduct(Eigen::MatrixXd & Out,Eigen::MatrixXd & In,
                                                               Eigen::MatrixXd (& A)[3],Eigen::VectorXd & Z,
                                                               const Eigen::MatrixXi & coord, 
                                                               const Eigen::VectorXd & wt,
                                                               int dm,double lambda){
  
  // get indices
  constexpr int DIM = 3;
  int ci = (dm + 0)%DIM;
  int cj = (dm + 1)%DIM;
  int ck = (dm + 2)%DIM;
  
  // get matrices
   // Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  
  auto length = wt.size();
  
  // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
  Out = lambda*In;
  
  // The weighted MTTKRP part (first half)
  for(auto u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    double temp = 0;
    for(int z=0;z<nZ;z++){
      temp += In(z,eye)*Aj(z,jay)*Ak(z,kay)*Z(z);
    }
    (*halfway)(u) = wt(u)*temp;
  }
  // The weighted MTTKRP part (second half)
  for(int u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    
    double forward = (*halfway)(u);
    for(int z=0;z<nZ;z++){
      Out(z,eye) += Aj(z,jay)*Ak(z,kay)*Z(z)*forward;
    }
  }
}

template<> void BlockConjugateGradient<4>::SparseMatrixProduct(Eigen::MatrixXd & Out,Eigen::MatrixXd & In,
                                                               Eigen::MatrixXd (& A)[4],Eigen::VectorXd & Z,
                                                               const Eigen::MatrixXi & coord, 
                                                               const Eigen::VectorXd & wt,
                                                               int dm,double lambda){
  
  // get indices
  constexpr int DIM = 4;
  int ci = (dm + 0)%DIM;
  int cj = (dm + 1)%DIM;
  int ck = (dm + 2)%DIM;
  int cl = (dm + 3)%DIM;
  
  // get matrices
  // Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  Eigen::MatrixXd & Al = A[cl];
  
  auto length = wt.size();
  
  // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
  Out = lambda*In;
  
  // The weighted MTTKRP part (first half)
  for(auto u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    int ell = coord(cl,u);
    double temp = 0;
    for(int z=0;z<nZ;z++){
      temp += In(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Z(z);
    }
    (*halfway)(u) = wt(u)*temp;
  }
  // The weighted MTTKRP part (second half)
  for(int u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    int ell = coord(cl,u);
    double forward = (*halfway)(u);
    for(int z=0;z<nZ;z++){
      Out(z,eye) += Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Z(z)*forward;
    }
  }
}

template<> void BlockConjugateGradient<5>::SparseMatrixProduct(Eigen::MatrixXd & Out,Eigen::MatrixXd & In,
                                                               Eigen::MatrixXd (& A)[5],Eigen::VectorXd & Z,
                                                               const Eigen::MatrixXi & coord, 
                                                               const Eigen::VectorXd & wt,
                                                               int dm,double lambda){
  
  // get indices
  constexpr int DIM = 5;
  int ci = (dm + 0)%DIM;
  int cj = (dm + 1)%DIM;
  int ck = (dm + 2)%DIM;
  int cl = (dm + 3)%DIM;
  int cm = (dm + 4)%DIM;
  
  // get matrices
  // Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  Eigen::MatrixXd & Al = A[cl];
  Eigen::MatrixXd & Am = A[cm];
  
  auto length = wt.size();
  
  // The \lambda0*||x||^2+lambdaK*||x-x_x||^2 part
  Out = lambda*In;
  
  // The weighted MTTKRP part (first half)
  for(auto u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    int ell = coord(cl,u);
    int emm = coord(cm,u);
    double temp = 0;
    for(int z=0;z<nZ;z++){
      temp += In(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Am(z,emm)*Z(z);
    }
    (*halfway)(u) = wt(u)*temp;
  }
  // The weighted MTTKRP part (second half)
  for(int u=0;u<length;u++){
    int eye = coord(ci,u);
    int jay = coord(cj,u);
    int kay = coord(ck,u);
    int ell = coord(cl,u);
    int emm = coord(cm,u);
    double forward = (*halfway)(u);
    for(int z=0;z<nZ;z++){
      Out(z,eye) += Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Am(z,emm)*Z(z)*forward;
    }
  }
}

#endif
 





















