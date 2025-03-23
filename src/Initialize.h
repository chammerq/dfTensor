#ifndef INITIALIZATION_HEADER
#define INITIALIZATION_HEADER


#include "set_debug.h"

#include <RcppEigen.h>
#include "common_functions.h"
#include <cmath>
// [[Rcpp::depends(RcppEigen)]]

// Some enums
enum {RANDOM_INIT=0,DSKETCH_INIT=1,RSKETCH_INIT=2};
enum {PLSA_INIT=0,CPD_INIT=1};

 
 
 
// Create a list of random initialization, normalize or orthogonalize as needed
 template<int DIM> Rcpp::List  random_initial_factors(const Eigen::VectorXd & qty,int & decomp_type, int & nlatent,const Rcpp::NumericVector & tdims){
   Rcpp::List ReturnMe(DIM+1);
   for(int dm = 0;dm<DIM;dm++){
     Eigen::MatrixXd RandM = Eigen::MatrixXd::Random(nlatent,tdims(dm));
     if(decomp_type==PLSA_INIT){
       RandM = RandM.array().abs();
       rescale_back2_prob(RandM, 0.001); // numbers initially between 0.001 and 1
     }
     if(decomp_type==CPD_INIT){
       RescaleA(RandM,true);
     }
     ReturnMe(dm) = RandM;
   }
   Eigen::VectorXd Z = Eigen::VectorXd::Ones(nlatent);
   if(decomp_type==PLSA_INIT){
     rescale_back2_prob(Z, 1);
   }else if(decomp_type==CPD_INIT){
     double XYZ_norm_bound = nlatent;
     for(int i=0;i<DIM;i++){
       XYZ_norm_bound *= double(nlatent*tdims(i));
     }
     Z *= qty.norm()/sqrt(XYZ_norm_bound);
   }
   
   ReturnMe(DIM)=Z;
   
   return ReturnMe;
 };
 

 
// Create a more advanced initialization
// Matricizes tensor in all the different ways, then contracts with a
// partial  hadamard-like matrix, then does qr on contracted matrix
// Formula contracting matrix: ((floor(k*(j+1)/ncol)%%2)==0)? -1 : 1
// where K is row index, J is column index, ncol the number of columns
template<int DIM> Rcpp::List  sketch_initial_factors(const Eigen::MatrixXi & coord,
                                                 const Eigen::VectorXd & qty,
                                                 int & decomp_type, int & nlatent,
                                                 const Rcpp::NumericVector & tdims){
  // Create and initialize list 
  Eigen::MatrixXd As[DIM];
  for(int dm = 0;dm<DIM;dm++){
    As[dm] = Eigen::MatrixXd::Zero(nlatent,tdims(dm));
  }
 
 // Calculate matricized dimensions
 int totald=1;
 for(int d=0;d<DIM;d++){
   totald *= tdims(d);
 }
 float one_over_dim[DIM];
 for(int d=0;d<DIM;d++){
   one_over_dim[d] = 1.0/float(totald/tdims(d));
 }
 // Create a matrix for converting tensor indices to linear indices
 // JKL*i + KL*j + L*k + 1*l + 0*m for each row
 Eigen::Matrix<int, DIM, DIM> index_coeff = Eigen::Matrix<int, DIM, DIM>::Zero(DIM,DIM);
 for(int dm=0;dm<DIM;dm++){
   index_coeff(dm,dm) = 0;
   index_coeff(dm,(dm+1)%DIM) = 1;
   if(DIM>2){
     index_coeff(dm,(dm+2)%DIM) = tdims((dm+2)%DIM);
   }
   if(DIM>3){
     index_coeff(dm,(dm+3)%DIM) = tdims((dm+2)%DIM)*tdims((dm+3)%DIM);
   }
   if(DIM>4){
     index_coeff(dm,(dm+4)%DIM) = tdims((dm+2)%DIM)*tdims((dm+3)%DIM)*tdims((dm+4)%DIM);
   }
   // Add more dimensions as needed
 }

  int lng = qty.size();
 
 // loop over data
 for(int i=0;i<lng;i++){
   Eigen::Matrix<int, DIM, 1> crd_slice = coord.col(i);
   Eigen::Matrix<int, DIM, 1> flat_ind = index_coeff*crd_slice;// Ex: flat_ind = JKL*i + KL*j + L*k + 1*l + 0*m for each row
   double Qty = qty(i);
   for(int d=0;d<DIM;d++){
     Eigen::MatrixXd & A =  As[d];
     int Ad_col_ind =crd_slice(d); // column index of A[d]
     float lind_col = float(flat_ind(d))*one_over_dim[d];

     for(int z=0;z<nlatent;z++){
       bool toggle = int(floor(float(z+1)*lind_col))%2 == 0;
       A(z,Ad_col_ind) += toggle?(-Qty):Qty;
     }
   }
 }
 
 // Loop over, orthogonalize, scale matrices
 Eigen::VectorXd Z = Eigen::VectorXd::Ones(nlatent);
 for(int dm = 0;dm<DIM;dm++){
   Eigen::MatrixXd & A =  As[dm];
   RescaleA(A,Z,true,decomp_type==PLSA_INIT); //orthogonalize
   if(decomp_type==PLSA_INIT){
    A *= 0.2; // reduce variance of A under exponential
    A = (A.array().exp()).matrix();
    rescale_back2_prob(A, 0.001);
   }
 }
 // Calculate Z
 if(decomp_type==PLSA_INIT){
   rescale_back2_prob(Z, Z.mean()+0.1);
 }else if(decomp_type==CPD_INIT){
   double XYZ_norm_bound = nlatent;

   for(int i=0;i<DIM;i++){
     XYZ_norm_bound *= double(nlatent*tdims(i));
   }

  Z *= qty.norm()/sqrt(XYZ_norm_bound);
 }
 //
 Rcpp::List ReturnMe(DIM+1);
 for(int dm = 0;dm<DIM;dm++){
   ReturnMe(dm) = As[dm];
 }
 ReturnMe(DIM)=Z;
  return ReturnMe;
};

 
// Initialization function
template<int DIM> Rcpp::List create_initial_factors(const Eigen::MatrixXi & coord,
                                                    const Eigen::VectorXd & qty,
                                                    int & initialize_type,
                                                    int & decomp_type,int & nlatent,
                                                    const Rcpp::NumericVector & tdims){
  if(initialize_type == DSKETCH_INIT){
    return sketch_initial_factors<DIM>(coord,qty,decomp_type,nlatent,tdims);  
  }
  return random_initial_factors<DIM>(qty,decomp_type,nlatent,tdims);
};


#endif