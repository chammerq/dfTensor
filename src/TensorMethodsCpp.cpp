// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "set_debug.h"
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "common_functions.h"
#include "ALS_CP_XD.h"
#include "PLSA.h"
#include "Reconstruct.h"
#include "Initialize.h"
#include "Polish.h"
#include "PoissonRates.h"



// ----------------------------------------------
// CP decomposition for sparse tensors
// ----------------------------------------------
// [[Rcpp::export]]
Rcpp::List sparse_als_cp(Rcpp::List & Factors,
                            const Eigen::MatrixXi & coord,
                            const Eigen::VectorXd & qty,
                            const Rcpp::NumericVector & number_iterations,
                            const Rcpp::NumericVector & lambdas_etc,
                            const Rcpp::NumericVector & UpdateThese,int DIM){
  if(DIM==2){
    return sparse_als_cp_xd<2>(Factors,coord,qty, number_iterations, lambdas_etc, UpdateThese);    
  }
  if(DIM==3){
    return sparse_als_cp_xd<3>(Factors,coord,qty, number_iterations, lambdas_etc, UpdateThese);    
  }
  if(DIM==4){
    return sparse_als_cp_xd<4>(Factors,coord,qty, number_iterations, lambdas_etc, UpdateThese);    
  }
  if(DIM==5){
    return sparse_als_cp_xd<5>(Factors,coord,qty, number_iterations, lambdas_etc, UpdateThese);    
  }
  
  Rcpp::stop("sparse_als_cp: Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
}


// ----------------------------------------------
// Weighted CP decomposition 
// (Also for sparse values viewed as missing)
// ----------------------------------------------

// [[Rcpp::export]]
Rcpp::List sparse_als_wt_cp(Rcpp::List & Factors,
                            const Eigen::MatrixXi & coord,
                            const Eigen::VectorXd & qty,
                            const Eigen::VectorXd & wt,
                            const Rcpp::NumericVector & number_iterations,
                            const Rcpp::NumericVector & lambdas_etc,
                            const Rcpp::NumericVector & UpdateThese,int DIM){
  if(DIM==2){
    return sparse_als_wt_cp_xd<2>(Factors,coord,qty,wt,number_iterations, lambdas_etc, UpdateThese);
  }
  if(DIM==3){
    return sparse_als_wt_cp_xd<3>(Factors,coord,qty,wt,number_iterations, lambdas_etc, UpdateThese);
  }
  if(DIM==4){
    return sparse_als_wt_cp_xd<4>(Factors,coord,qty,wt,number_iterations, lambdas_etc, UpdateThese);
  }
  if(DIM==5){
    return sparse_als_wt_cp_xd<5>(Factors,coord,qty,wt,number_iterations, lambdas_etc, UpdateThese);
  }
   
  Rcpp::stop("sparse_als_wt_cp: Not yet implemented for rank %i tensors ",DIM);
 // Never get here
 Rcpp::List ReturnMe(DIM);
  return ReturnMe;
}


// ----------------------------------------------
// PLSA decomposition for sparse tensors
// ----------------------------------------------

// [[Rcpp::export]]
Rcpp::List sparse_plsa_xd(Rcpp::List & Factors,
                            const Eigen::MatrixXi & coord,
                            const Eigen::VectorXd & qty,
                            int  number_iterations,
                            const Rcpp::NumericVector & tolerances,
                            const Rcpp::NumericVector & UpdateThese,
                            const Rcpp::NumericVector & prior,int DIM){
  switch(DIM){
    case 2: 
      return sparse_plsa_xd<2>(Factors,coord,qty, number_iterations, tolerances, UpdateThese,prior);
    case 3: 
      return sparse_plsa_xd<3>(Factors,coord,qty, number_iterations, tolerances, UpdateThese,prior);
    case 4: 
      return sparse_plsa_xd<4>(Factors,coord,qty, number_iterations, tolerances, UpdateThese,prior);
    case 5: 
      return sparse_plsa_xd<5>(Factors,coord,qty, number_iterations, tolerances, UpdateThese,prior);
  }
   
   Rcpp::stop("sparse_plsa_xd: Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
  
  
}


// ----------------------------------------------
// Function to multiply factor matrices at non-zero tensor locations 
// ----------------------------------------------

// [[Rcpp::export]]
Eigen::VectorXd ReconstructSparseTensor(Rcpp::List & Factors,const Eigen::MatrixXi & coord,int DIM){
  
    // Call functions
  if(DIM==2){
    return ReconstructSparseFromList<2>(Factors,coord);
  }
  if(DIM==3){
    return ReconstructSparseFromList<3>(Factors,coord);
  }
  if(DIM==4){
    return ReconstructSparseFromList<4>(Factors,coord);
  }
  if(DIM==5){
    return ReconstructSparseFromList<5>(Factors,coord);
  }
  
  Rcpp::stop("ReconstructSparseTensor: Not yet implemented for rank %i tensors ",DIM);
  
  // never get here
  Eigen::VectorXd fit;
  return fit;
}

// ----------------------------------------------
// Function to find random initialization
// ----------------------------------------------
// [[Rcpp::export]]
Rcpp::List Initialize(const Eigen::MatrixXi & coord,const Eigen::VectorXd & qty,
                      int initialize_type,int decomp_type, int nlatent,
                      const Rcpp::NumericVector & tdims,int DIM){
   
  switch(DIM){
    case 2: 
      return create_initial_factors<2>(coord,qty,initialize_type,decomp_type,nlatent,tdims);
    case 3: 
      return create_initial_factors<3>(coord,qty,initialize_type,decomp_type,nlatent,tdims);
    case 4: 
      return create_initial_factors<4>(coord,qty,initialize_type,decomp_type,nlatent,tdims);
    case 5: 
      return create_initial_factors<5>(coord,qty,initialize_type,decomp_type,nlatent,tdims);
  }
    
  Rcpp::stop("Initialize(): Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
  
}

// ----------------------------------------------
// Function to find get means across dimensions
// ----------------------------------------------
// [[Rcpp::export]]
Rcpp::List Mean_polish(const Rcpp::NumericVector & dims,
                       const Eigen::MatrixXi & coord,
                       const Eigen::ArrayXd & qty,
                       int numit,double abs_tol, double rel_tol,double stepsize,int DIM){
    
  switch(DIM){
  case 2: 
    return means_xd<2,false>(dims,coord,qty,nullptr,numit,abs_tol,rel_tol,stepsize);
  case 3: 
    return means_xd<3,false>(dims,coord,qty,nullptr,numit,abs_tol,rel_tol,stepsize);
  case 4: 
    return means_xd<4,false>(dims,coord,qty,nullptr,numit,abs_tol,rel_tol,stepsize);
  case 5:
    return means_xd<5,false>(dims,coord,qty,nullptr,numit,abs_tol,rel_tol,stepsize);
  }
     
  Rcpp::stop("Mean_polish: Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
  
}

// [[Rcpp::export]]
Rcpp::List Mean_polish_wtd(const Rcpp::NumericVector & dims,
                     const Eigen::MatrixXi & coord,
                     const Eigen::ArrayXd & qty,
                     const Eigen::ArrayXd & wt,
                     int numit,double abs_tol, double rel_tol,double stepsize,int DIM){
    
  switch(DIM){
  case 2: 
    return means_xd<2,true>(dims,coord,qty,&wt,numit,abs_tol,rel_tol,stepsize);
  case 3: 
    return means_xd<3,true>(dims,coord,qty,&wt,numit,abs_tol,rel_tol,stepsize);
  case 4: 
    return means_xd<4,true>(dims,coord,qty,&wt,numit,abs_tol,rel_tol,stepsize);
  case 5:
    return means_xd<5,true>(dims,coord,qty,&wt,numit,abs_tol,rel_tol,stepsize);
  }
  Rcpp::stop("Mean_polish_wtd: Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
  
}

// ----------------------------------------------
// Find Poisson Rates
// ----------------------------------------------
// [[Rcpp::export]]
Rcpp::List poisson_rates(const Rcpp::NumericVector & dims,
                         const Eigen::MatrixXi & coord,
                         const Eigen::VectorXd & qty,
                         int numit,double rel_tol,
                         double threshold,int DIM){
  
  switch(DIM){
  case 2:
    return poisson_rates_xd<2>(dims,coord,qty,numit,rel_tol,threshold);
  case 3:
    return poisson_rates_xd<3>(dims,coord,qty,numit,rel_tol,threshold);
  case 4:
    return poisson_rates_xd<4>(dims,coord,qty,numit,rel_tol,threshold);
  case 5:
    return poisson_rates_xd<5>(dims,coord,qty,numit,rel_tol,threshold);
  }
  Rcpp::stop("poisson_rates: Not yet implemented for rank %i tensors ",DIM);
  // Never get here
  Rcpp::List ReturnMe(DIM);
  return ReturnMe;
  
}

// ----------------------------------------------
// Function to sort tensor
// ----------------------------------------------

// [[Rcpp::export]]
Rcpp::IntegerVector sort_indices( const Eigen::MatrixXi & coord,Rcpp::List & is_unique){
  is_unique(0) = true;
  Rcpp::IntegerVector index = Rcpp::seq(0, coord.cols() - 1);
  std::sort(index.begin(),index.end(),[&](size_t a,size_t b){
                      for(int dm=0;dm<coord.rows();dm++){
                        if(coord(dm,a)!= coord(dm,b)){
                          return (coord(dm,a) < coord(dm,b));
                        }
                      }
                      is_unique(0) = false;
                      return false;
                    });
  return index + 1;
}
 