#ifndef DFTENSOR_HEADER_1
#define DFTENSOR_HEADER_1

//#include "set_debug.h"
#include <RcppEigen.h>

//#include <quadmath.h>
//typedef __float128 quad;

inline void orthogonalize_columns(Eigen::MatrixXd & A){
    auto nz = A.rows();
    auto nd = A.cols();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A.transpose());
    A = (qr.householderQ()*Eigen::MatrixXd::Identity(nd,nz)).transpose();
}


// Function rescale rows of matrix to be chi-square
// Maybe orthogonalize columns
inline void RescaleA(Eigen::MatrixXd & A,bool orthogonalize){

  double threshold =1e-12; 
  double n = 1./sqrt(double(A.cols()));
  
  // get column norm
  Eigen::VectorXd rwnorm = A.rowwise().norm();

  // loop over multiply by n and make sure we don't have zeros (safe division)
  auto ptr = rwnorm.data();
  for (auto i = 0; i < rwnorm.size(); i++){
    double tmp =(*(ptr+i))*n;
    *(ptr+i) = (tmp >= threshold) ? tmp : threshold;
  }

  // normalize matrices
  A = (rwnorm.cwiseInverse().asDiagonal())*A; // make the columns equal chi-square mean
  
  if(orthogonalize){
    orthogonalize_columns(A);
    A *= sqrt(double(A.cols()));
  }
}

// Function rescale rows of matrix to be chi-square, copy to Z
// Maybe orthogonalize columns
//inline void RescaleA(Eigen::MatrixXd & A,bool orthogonalize,Eigen::VectorXd * Z = NULL){
inline void RescaleA(Eigen::MatrixXd & A,Eigen::VectorXd & Z,bool orthogonalize,bool scaleZ){
  
  if(scaleZ){
    double threshold =1e-12; 
    double n = 1./sqrt(double(A.cols()));
    
    // get column norm
    Eigen::VectorXd rwnorm = A.rowwise().norm();
  
    // loop over multiply by n and make sure we don't have zeros (safe division)
    auto ptr = rwnorm.data();
    for (auto i = 0; i < rwnorm.size(); i++){
      double tmp =(*(ptr+i))*n;
      *(ptr+i) = (tmp >= threshold) ? tmp : threshold;
    }
    
    // normalize matrices
    Z = Z.cwiseProduct(rwnorm);
    A = (rwnorm.cwiseInverse().asDiagonal())*A;
    
    // Orthogonalize matrices
    if(orthogonalize){
      orthogonalize_columns(A);
      A *= sqrt(double(A.cols())); // scale back to chi-square
    }
    
  }else if(orthogonalize){
      orthogonalize_columns(A);
  }
}


// Function to rescale columns of a matrix so they can represent probabilities
inline void rescale_back2_prob(Eigen::MatrixXd & pM,double cutoff){
  // loop over and make sure we don't have zeros
  double threshold = cutoff/double(pM.rows());
  auto ptr = pM.data();
  for (auto i = 0; i < pM.size(); i++){
    *(ptr+i) = (*(ptr+i) >= threshold) ? *(ptr+i) : threshold;
  }
  // use kahan alogorithm to get stable sums
  Eigen::ArrayXd Sum = Eigen::ArrayXd::Zero(pM.rows());
  Eigen::ArrayXd Res= Eigen::ArrayXd::Zero(pM.rows());
  for(auto i = 0; i<pM.cols(); i++){
    for(auto j = 0;j<pM.rows();j++){
      double y = pM(j,i) - Res[j];
      double increment = Sum[j]+y;
      Res[j]  =  (increment -  Sum[j]) - y;
      Sum[j] = increment;
    } 
  }
  pM = Sum.matrix().cwiseInverse().asDiagonal()*pM;
  
  
}

// Function to rescale a vector so it can represent probabilities
inline void rescale_back2_prob(Eigen::VectorXd & pV,double cutoff){
  // loop over and make sure we don't have zeros
  auto ptr = pV.data();
  double threshold = cutoff/double(pV.size());
  for (int i = 0; i < pV.size(); i++){
    *(ptr+i) = (*(ptr+i) >= threshold) ? *(ptr+i) : threshold;
  }
  
  double inv_sum =1.0/ pV.sum();
  pV *= inv_sum;
}

// A function to try to adjust lambda to handle the rescaling of Z
double update_regularization_parameters(double & lambda0,double & lambda_k, Eigen::VectorXd & Z,bool & scaleZ ){
  if(scaleZ){
    return(lambda0*Z.squaredNorm()+lambda_k);
  }
  return(lambda0+lambda_k); 
}

#endif