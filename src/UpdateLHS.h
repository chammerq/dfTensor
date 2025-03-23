#ifndef UPDATE_LHS_HEADER
#define UPDATE_LHS_HEADER

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

template<int DIM> inline void Calculate_LHS(Eigen::MatrixXd & LHS,
                               Eigen::MatrixXd (& AtA)[DIM],
                               Eigen::MatrixXd & ZtZ,int dim,double lambda){}


template<> inline void Calculate_LHS<2>(Eigen::MatrixXd & LHS,
                                        Eigen::MatrixXd (& AtA)[2],
                                        Eigen::MatrixXd & ZtZ,int dim,double lambda){
  LHS.setIdentity();
  LHS *= lambda;
  LHS +=ZtZ.cwiseProduct(AtA[(dim+1)%2]);
  
}


template<> inline void Calculate_LHS<3>(Eigen::MatrixXd & LHS,
                          Eigen::MatrixXd (& AtA)[3],
                           Eigen::MatrixXd & ZtZ,int dim,double lambda){
  LHS.setIdentity();
  LHS *= lambda;
  LHS +=(ZtZ.cwiseProduct(AtA[(dim+1)%3])).cwiseProduct(AtA[(dim+2)%3]);
  
}
  
template<> inline void Calculate_LHS<4>(Eigen::MatrixXd & LHS,
                                        Eigen::MatrixXd (& AtA)[4],
                                        Eigen::MatrixXd & ZtZ,int dim,double lambda){
  LHS.setIdentity();
  LHS *= lambda;
  LHS +=((ZtZ.cwiseProduct(AtA[(dim+1)%4])).cwiseProduct(AtA[(dim+2)%4])).cwiseProduct(AtA[(dim+3)%4]);
  
}
  
template<> inline void Calculate_LHS<5>(Eigen::MatrixXd & LHS,
                                        Eigen::MatrixXd (& AtA)[5],
                                        Eigen::MatrixXd & ZtZ,int dim,double lambda){
  LHS.setIdentity();
  LHS *= lambda;
  LHS +=(((ZtZ.cwiseProduct(AtA[(dim+1)%5])).cwiseProduct(AtA[(dim+2)%5])).cwiseProduct(AtA[(dim+3)%5])).cwiseProduct(AtA[(dim+4)%5]);
  //LHS += (ZtZ.array()*AtA[(dim+1)%5].array()*AtA[(dim+2)%5].array()*AtA[(dim+3)%5].array()*AtA[(dim+4)%5]).matrix();
  
}


#endif