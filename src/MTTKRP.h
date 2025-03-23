#ifndef MTTKRP_HEADER
#define MTTKRP_HEADER

//#include "set_debug.h"
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]


template<int D> inline void MTTKRP(Eigen::MatrixXd & RHS,
                            Eigen::MatrixXd (& A)[D],
                              Eigen::VectorXd & Z,
                              const Eigen::MatrixXi & Cood, 
                              const Eigen::VectorXd & qty,
                              int dm,
                              double lambda_k){};



// Rank 2 tensor (matrix)
template<> inline void MTTKRP<2>(Eigen::MatrixXd & RHS,
                                 Eigen::MatrixXd (& A)[2],
                                Eigen::VectorXd & Z,
                                const Eigen::MatrixXi & Cood, 
                                const Eigen::VectorXd & qty,
                                int dm,
                                double lambda_k){
  
  int ci = dm%2;
  int cj = (dm+1)%2;

  
  Eigen::MatrixXd & Ai = A[ci]; 
  Eigen::MatrixXd & Aj = A[cj];

  
  // initialize
  int length = Cood.cols();
  int Nz = Z.size();
  RHS = lambda_k*Ai; 
  
  // Get MTTKRP
  for(auto i=0;i<length;i++){
    int eye = Cood(ci,i);
    int jay = Cood(cj,i);
    double Qty = qty(i);
    for(int z=0;z<Nz;z++){
      RHS(z,eye) += Aj(z,jay)*Z(z)*Qty;
    }
  }
}
// Rank 3 Tensor
template<> inline void MTTKRP<3>(Eigen::MatrixXd & RHS,
                                 Eigen::MatrixXd (& A)[3],
                                      Eigen::VectorXd & Z,
                                      const Eigen::MatrixXi & Cood, 
                                      const Eigen::VectorXd & qty,
                                      int dm,
                                      double lambda_k){
  
  int ci = dm%3;
  int cj = (dm+1)%3;
  int ck = (dm+2)%3;
  
  Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  
  // initialize
  int length = Cood.cols();
  int Nz = Z.size();
  RHS = lambda_k*Ai; 
  
  // Get MTTKRP
  for(auto i=0;i<length;i++){
    int eye = Cood(ci,i);
    int jay = Cood(cj,i);
    int kay = Cood(ck,i);
    double Qty = qty(i);
    for(int z=0;z<Nz;z++){
      RHS(z,eye) += Aj(z,jay)*Ak(z,kay)*Z(z)*Qty;
    }
  }
  
  
}

// Rank 4 Tensor
template<> inline void MTTKRP<4>(Eigen::MatrixXd & RHS,
                                 Eigen::MatrixXd (& A)[4],
                                Eigen::VectorXd & Z,
                                const Eigen::MatrixXi & Cood, 
                                const Eigen::VectorXd & qty,
                                int dm,
                                double lambda_k){
  
  int ci = dm%4;
  int cj = (dm+1)%4;
  int ck = (dm+2)%4;
  int cl = (dm+3)%4;
  
  Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  Eigen::MatrixXd & Al = A[cl];
  
  // initialize
  int length = Cood.cols();
  int Nz = Z.size();
  RHS = lambda_k*Ai; 
  
  // Get MTTKRP
  for(auto i=0;i<length;i++){
    int eye = Cood(ci,i);
    int jay = Cood(cj,i);
    int kay = Cood(ck,i);
    int ell = Cood(cl,i);
    double Qty = qty(i);
    for(int z=0;z<Nz;z++){
      RHS(z,eye) += Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Z(z)*Qty;
    }
  }
  
  
}

// Rank 5 Tensor
template<> inline void MTTKRP<5>(Eigen::MatrixXd & RHS,
                                 Eigen::MatrixXd (& A)[5],
                                  Eigen::VectorXd & Z,
                                  const Eigen::MatrixXi & Cood, 
                                  const Eigen::VectorXd & qty,
                                  int dm,
                                  double lambda_k){
  
  int ci = dm%5;
  int cj = (dm+1)%5;
  int ck = (dm+2)%5;
  int cl = (dm+3)%5;
  int cm = (dm+4)%5;
  
  Eigen::MatrixXd & Ai = A[ci];
  Eigen::MatrixXd & Aj = A[cj];
  Eigen::MatrixXd & Ak = A[ck];
  Eigen::MatrixXd & Al = A[cl];
  Eigen::MatrixXd & Am = A[cm];
  
  // initialize
  int length = Cood.cols();
  int Nz = Z.size();
  RHS = lambda_k*Ai; 
  
  // Get MTTKRP
  for(auto i=0;i<length;i++){
    int eye = Cood(ci,i);
    int jay = Cood(cj,i);
    int kay = Cood(ck,i);
    int ell = Cood(cl,i);
    int emm = Cood(cm,i);
    double Qty = qty(i);
    for(int z=0;z<Nz;z++){
      RHS(z,eye) += Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Am(z,emm)*Z(z)*Qty;
    }
  }
  
  
}

//====================================================
// Remove mean from LHS
template<int DIM,bool wMean,bool All> inline void Remove_mean_RHS(Eigen::MatrixXd & RHS,
                                                                  Eigen::ArrayXd (& A1)[DIM],
                                                                  Eigen::VectorXd & Z,int dim,double mean){
  if constexpr (wMean){
    if constexpr (All){
      A1[dim] *= (mean*Z).array();  
    }else{
        A1[dim] = (mean*Z).array(); // include all but this one
    }
    for(int d=1;d<DIM;d++){
      A1[dim] *= A1[(dim+d)%DIM];
    }
    RHS.colwise() -= A1[dim].matrix();
  }
}
    

#endif
