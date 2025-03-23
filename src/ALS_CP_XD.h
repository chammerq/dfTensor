#ifndef ALS_CP_XD
#define ALS_CP_XD

#include "set_debug.h"
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "MTTKRP.h"
#include "common_functions.h"
#include "UpdateLHS.h"
#include "Reconstruct.h"
#include "BlockConjugateGradient.h"

// Template definition of ALS
template<int DIM> Rcpp::List sparse_als_cp_xd(Rcpp::List & Factors,
                                        const Eigen::MatrixXi & coord,
                                        const Eigen::VectorXd & qty,
                                        const Rcpp::NumericVector & number_iterations,
                                        const Rcpp::NumericVector & lambda_etc,
                                        const Rcpp::NumericVector & UpdateThese){

  
  // rename
  int als_it = number_iterations(0);
  int orth_it = number_iterations(1);
  bool TrackError = UpdateThese(DIM+1)>0;
  bool scaleZ = UpdateThese(DIM)>0; // if Z is not fixed

  // rename lambdas
  double lambda0 = lambda_etc(0);
  double lambdak = lambda_etc(1);
  double abs_tol = lambda_etc(2);
  double rel_tol = lambda_etc(3);
  double DtD = 0;
  
  // // Initialize vectors  
  Eigen::VectorXd Z = Factors(DIM);
  int nz = Z.size();

  Eigen::ArrayXd sse;
  Eigen::VectorXd resid;
  Eigen::VectorXd fit = Eigen::VectorXd::Zero(qty.size());
  if(TrackError){
    sse = Eigen::ArrayXd::Zero(als_it+1); 
    resid = Eigen::ArrayXd::Zero(qty.size());
    DtD = qty.squaredNorm();
  }else{    
    sse = Eigen::ArrayXd::Zero(1); 
    resid = Eigen::ArrayXd::Zero(1);
    sse(0) = NAN; // if not needed, return na
    resid(0)=NAN;
  }

  // Initialize Matrices
  Eigen::MatrixXd A[DIM];
  Eigen::MatrixXd AtA[DIM];
  Eigen::MatrixXd RHS[DIM];

  for(int i=0;i<DIM;i++){
    A[i] = Factors(i);
    AtA[i] = A[i]*A[i].transpose().eval();
    RHS[i] = Eigen::MatrixXd::Zero(nz,A[i].cols());
  }
  Eigen::MatrixXd ZtZ =Z*Z.transpose().eval(); 
  Eigen::MatrixXd LHS(nz,nz);

  //_____________________________
  // Iterate the ALS algorithm
  //-----------------------------
  int numit = 0;
  for(int it=0;it<als_it;it++){
    
    Rcpp::checkUserInterrupt();// should we exit?
    
    bool orthogonalize = it<orth_it; // should we orthogonalize this iteration
    
    // loop over dimensions
    for(int dm=0;dm<DIM;dm++){
      if(UpdateThese(dm)>0){
        double lambda_lhs = update_regularization_parameters(lambda0,lambdak,Z,scaleZ);
        Calculate_LHS<DIM>(LHS,AtA,ZtZ,dm,lambda_lhs); // left hand side of least squares
        MTTKRP<DIM>(RHS[dm],A,Z,coord,qty,dm,lambdak); // get MTTKRP
        A[dm] = LHS.completeOrthogonalDecomposition().solve(RHS[dm]);    // then solve ALS
        RescaleA(A[dm],Z,orthogonalize,scaleZ);            // re-normalize
        ZtZ = Z*Z.transpose();                     // update cross products
        AtA[dm] = A[dm]*A[dm].transpose(); 
      }
    }
    
    // check fit if asked
    if(TrackError){
     
      int dm = DIM-1;
      // get full AtA
      Calculate_LHS<DIM>(LHS,AtA,ZtZ,dm,0); // get most of AtA
      double sumAtA  = (LHS*AtA[dm]).trace(); //include last A, get norm
   
      //Get full AtD
      MTTKRP<DIM>(RHS[dm],A,Z,coord,qty,dm,0); // get  most of AtD
      double sumAtD = (RHS[dm]*(A[dm].transpose())).trace();  //include last A, get norm
      
      sse(it) = sumAtA - 2*sumAtD + DtD;
      numit=it+1;
      double abs_err = abs(sse(it));
      if(abs_err<abs_tol){
        break;
      }
      if(it>0){
        double rel_err = abs(sse(it-1)-abs_err)/(abs_err+ 1e-6);
        if(rel_err<rel_tol){
          break;
        }
      }
    }
  }
  
  ReconstructSparse_w_resid<DIM>(A,Z,coord,fit,qty,resid);// get fit
  
  // copy back to list
  Rcpp::List factor_matrices(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    factor_matrices(dm)=A[dm];
  }    
  factor_matrices(DIM) = Z;

  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("factors") =  factor_matrices,
                                      Rcpp::Named("fitted.values") =  fit,
                                      Rcpp::Named("residuals") =  resid,
                                      Rcpp::Named("sse") =  sse.head(numit));
  return ReturnMe;
}


//++++++++++++++++++++++++++++++++++++++++
// Template definition of weighted ALS
//=++++++++++++++++++++++++++++++++++++++++
template<int DIM> Rcpp::List sparse_als_wt_cp_xd(Rcpp::List & Factors,
                                                 const Eigen::MatrixXi & coord,
                                                 const Eigen::VectorXd & qty,
                                                 const Eigen::VectorXd & wt,
                                                 const Rcpp::NumericVector & number_iterations,
                                                 const Rcpp::NumericVector & lambda_etc,
                                                 const Rcpp::NumericVector & UpdateThese){
  
  
  int als_it = number_iterations(0);
  int orth_it = number_iterations(1);
  //int wtd_it = number_iterations(2);
  bool TrackError = UpdateThese(DIM+1)>0;
  bool scaleZ = UpdateThese(DIM)>0;
  
  // rename lambdas
  double lambda0 = lambda_etc(0);
  double lambdak = lambda_etc(1);
  double abs_tol = lambda_etc(2);
  double rel_tol = lambda_etc(3);
  double cg_tol = 1e-7;
  
  // // Initialize vectors
  Eigen::VectorXd Z = Factors(DIM);
  int nz = Z.size();
  auto leng = qty.size();
  
  // CG iterations
  Eigen::VectorXd temp(leng);
  Eigen::VectorXd sse;
  Eigen::VectorXd resid;
  Eigen::VectorXd fit = Eigen::VectorXd::Zero(leng);
  
  if(TrackError){
    sse = Eigen::VectorXd::Zero(als_it); 
    resid = Eigen::VectorXd::Zero(leng);
  }else{    
    sse = Eigen::VectorXd::Zero(1); 
    resid = Eigen::VectorXd::Zero(1);
    sse(0) = NAN; // if not needed, return na
    resid(0) = NAN;
  }
  
  // Initialize Matrices
  Eigen::MatrixXd A[DIM];
  Eigen::MatrixXd RHS[DIM];
  BlockConjugateGradient<DIM> BCG[DIM];

  // copy over A
  for(int dm=0;dm<DIM;dm++){
    A[dm] = Factors(dm);
    RHS[dm] = Eigen::MatrixXd::Zero(nz,A[dm].cols());
  }
  
  // Initialize BCG, contains temporary variables
  for(int dm=0;dm<DIM;dm++){
    if(UpdateThese(dm)>0){ 
      BCG[dm] = BlockConjugateGradient<DIM>(nz,A[dm].cols(),&temp,dm,cg_tol);
    }
  }
  
  // create a weighted qty vector
  Eigen::VectorXd wtqty = qty.cwiseProduct(wt);
  
  //===================================
  // Iterate the ALS algorithm
  //===================================
  int numit=0;
  for(int it=0;it<als_it;it++){
    Rcpp::checkUserInterrupt();// should we exit?
    
    bool orthogonalize = it<orth_it; // should we orthogonalize this iteration
    // loop over dimensions
    for(int dm=0;dm<DIM;dm++){
      if(UpdateThese(dm)>0){

        // Update reg parameter
        double lambda_lhs = update_regularization_parameters(lambda0,lambdak,Z,scaleZ);
        
        // Get RHS (i.e. A^Tb)
        MTTKRP<DIM>(RHS[dm],A,Z,coord,wtqty,dm,lambdak); //get MTTKRP

        // Solve normal equations with Block Conjugate Gradient
        BCG[dm].Solve(RHS[dm],A,Z,coord, wt,dm,lambda_lhs);
        
        // re-normalize
        RescaleA(A[dm],Z,orthogonalize,scaleZ); 
      }
    }
    
    // check fit if asked
    if(TrackError){
      numit=it+1; 
      sse(it) = ReconstructSparse_w_resid<DIM>(A,Z,coord,fit,qty,resid,wt);
      double abs_err = sse[it];
      if(abs_err<abs_tol){
        break;
      }
      if(it>0){
        double rel_err = abs(sse[it-1]-abs_err)/(abs_err+ 1e-6);
        if(rel_err<rel_tol){
          break;
        }
      }
    }
  }
  
  // Calculate fit if needed
  if(!TrackError){
    ReconstructSparse<DIM>(A,Z,coord,fit);
  }
  
  // copy back to list
  Rcpp::List factor_matrices(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    factor_matrices(dm)=A[dm];
  }    
  factor_matrices(DIM) = Z;

  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("factors") =  factor_matrices,
                                           Rcpp::Named("fitted.values") =  fit,
                                           Rcpp::Named("residuals") =  resid,
                                           Rcpp::Named("sse") =  sse.head(numit));

  return ReturnMe;
} 
#endif
