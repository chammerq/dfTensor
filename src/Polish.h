#ifndef DIM_SCALE_XD_HEADER
#define DIM_SCALE_XD_HEADER

#include "set_debug.h"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]
template<int DIM,bool isWtd> Rcpp::List means_xd(const Rcpp::NumericVector & dims,
                                      const Eigen::MatrixXi & coord,
                                      const Eigen::ArrayXd & qty,
                                      const Eigen::ArrayXd * weight,
                                      int numit,double abs_tol, double rel_tol,double stepsize)
{
  
  
  // initialize stuff
  Eigen::ArrayXd sse = Eigen::ArrayXd::Zero(numit+1);
  Eigen::ArrayXd fit = Eigen::ArrayXd::Zero(qty.size());
  Eigen::ArrayXd Mu[DIM];
  Eigen::ArrayXd dMu[DIM];
  Eigen::ArrayXd isumwt[DIM]; // Inverse of sum weights

  double inv_sum_wt = 0;
  double mu0 = 0;
  double dmu0 = 0;
  Eigen::ArrayXd Resid = qty; // Initialize as all means are zero
  
  // Initialize means, inverse counts
  for(int i=0;i<DIM;i++){
      Mu[i] = Eigen::ArrayXd::Zero(dims(i));
      dMu[i] = Eigen::ArrayXd::Zero(dims(i));
      isumwt[i] = Eigen::ArrayXd::Zero(dims(i));
  }
  
  // Calculate denominators
  if constexpr(isWtd){ // first weighted version
    if (weight == nullptr) {
      throw std::invalid_argument("Calling weighted of means version without a weight");
    }
    const Eigen::ArrayXd & wt = *weight;
    // loop over and get normalizing constants
    double sumwt0 = 0;
    for(auto i=0;i<wt.size();i++){
      for(int dm=0;dm<DIM;dm++){
        (isumwt[dm])(coord(dm,i)) += wt(i);
      }
      sumwt0 += wt(i);
    }
    
    // Get inverse of sum of weights
    inv_sum_wt  = 1.0/sumwt0;
    double threshold = (1e-12)*sumwt0/static_cast<double>(qty.size());
    for(int dm=0;dm<DIM;dm++){
      for(auto i =0;i<isumwt[dm].size();i++){
        double temp = isumwt[dm](i);
        isumwt[dm](i) = temp<=threshold?0:1.0/temp;
      }
    }
  }else{ // unweighted version
    
    // Get total dimension
    double ntotal = 1.0;
    for(int i=0;i<DIM;i++){
      ntotal *= static_cast<double>(dims(i));
    }
    
    // Get inverse of sum of weights
    inv_sum_wt  = 1.0/ntotal;
    for(int i=0;i<DIM;i++){
      double invwt = static_cast<double>(dims(i))/ntotal; // divide by all the other dimensions but this one
      isumwt[i] = Eigen::ArrayXd::Constant(dims(i),invwt);
    }
  }

  //_____________________________
  // Iterate 
  //-----------------------------
  int it;
  for(it=0;it<numit;it++){
    Rcpp::checkUserInterrupt();// should we exit?
    
    // loop over data, get mean updates
    //--------------------------------
    for(auto i=0;i<qty.size();i++){
      double residual = Resid(i);
      
      // weight residual if needed
      if constexpr(isWtd){
        residual *= (*weight)(i);
      }
      
      for(int dm=0;dm<DIM;dm++){
        (dMu[dm])(coord(dm,i)) += residual;
      }
      dmu0 += residual;
    }
    
    // Update means with step, recenter
    //_________________________________
    double step_length = (it==0)?1.0:stepsize;
    for(int dm=0;dm<DIM;dm++){
      // increment mean
      Mu[dm] += step_length*(dMu[dm]*isumwt[dm]);
      dMu[dm].setZero();
      
      // re-center
      double meanM = Mu[dm].mean();
      Mu[dm] -= meanM;
    }
    // update global mean
    mu0 += step_length*(dmu0*inv_sum_wt);
    dmu0=0;
    
    // loop over data, get fit, residual
    //'''''''''''''''''''''''''''''''''''
    double sum_squared_error=0;
    for(auto i=0;i<qty.size();i++){
      double estimate = mu0;
      for(int dm=0;dm<DIM;dm++){
        estimate += (Mu[dm])(coord(dm,i)); 
      }
      fit(i) = estimate;
      double residual = qty(i) - estimate;
      Resid(i) = residual;
      
      // increment sum of residuals
      double r2 = residual*residual;
      if constexpr(isWtd){
        r2 *= (*weight)(i);
      }
      sum_squared_error += r2;
    }
    
    // Check early stopping
    //--------------------------------
    sse[it] = sum_squared_error; // update estimate
    if(sse[it]<abs_tol){
      break;
    }
    if(it>0){
      double rel_err = abs(sse[it-1]-sse[it])/(sse[it]+ 1e-10);
      if(rel_err<rel_tol){
        break;
      }
    }
  }
  
  // Copy over structures and return
  Rcpp::List dim_means(DIM+1);
  Rcpp::List inverse_counts(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    dim_means(dm)=Mu[dm];
    inverse_counts(dm) = isumwt[dm];
  } 
  dim_means(DIM) = mu0;
  inverse_counts(DIM) = inv_sum_wt;

  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("means") =  dim_means,
                                           Rcpp::Named("fitted.values") =  fit,
                                           Rcpp::Named("residuals") = Resid,
                                           Rcpp::Named("sse") =  sse.head(it),
                                           Rcpp::Named("inv.counts") = inverse_counts);
  return ReturnMe;
}

/*

// Function to center means
inline void update_means(Eigen::VectorXd & M,Eigen::VectorXd & dM,double inv_n){
  M += inv_n*dM;
  dM.setZero();
  double tempM = M.sum();
  M.array() -= (tempM/double(M.size()));

}

// Function to center means
inline void update_means(Eigen::VectorXd & M,Eigen::VectorXd & dM,Eigen::VectorXd & invW){
  M += dM.cwiseProduct(invW);
  dM.setZero();
  double tempM = M.sum();
  M.array() -= (tempM/double(M.size()));
  
}

// dim_scale_xd

// Template definition of ALS
template<int DIM> Rcpp::List means_xd(const Rcpp::NumericVector & dims,
                                              const Eigen::MatrixXi & coord,
                                              const Eigen::ArrayXd & qty,
                                              int numit,double abs_tol, double rel_tol,double stepsize)
{
  

  // initialize stuff
  auto lng = qty.size();
  Eigen::ArrayXd sse = Eigen::ArrayXd::Zero(numit+1); 
  Eigen::ArrayXd err = Eigen::ArrayXd::Zero(qty.size());
  Eigen::ArrayXd fit = Eigen::ArrayXd::Zero(qty.size());
  Eigen::VectorXd Mu[DIM];
  Eigen::VectorXd dMu[DIM];

  for(int i=0;i<DIM;i++){
    Mu[i] = Eigen::VectorXd::Zero(dims(i));
    dMu[i] = Eigen::VectorXd::Zero(dims(i));
  }
  
  // Get normalizing constant
  double ntotal = 1;
  for(int i=0;i<DIM;i++){
    ntotal *= dims(i);
  }
  double invn[DIM];
  for(int i=0;i<DIM;i++){
    invn[i] = dims(i)/ntotal;
  }
  
  double mu0 = (qty.sum())/ntotal;
  double dmu0=0;
  err = qty.array() - mu0;
  //_____________________________
  // Iterate 
  //-----------------------------
  int it;
  for(it=0;it<numit;it++){
    Rcpp::checkUserInterrupt();// should we exit?
    // maybe reduce time step
    if(it==1){
      for(int dm=0;dm<DIM;dm++){
        invn[dm] *= stepsize;
      } 
    }
    
    // re-estimate
    for(auto i=0;i<lng;i++){
      for(int dm=0;dm<DIM;dm++){
        (dMu[dm])(coord(dm,i)) += err(i);
      }
      dmu0 += err(i);
    }
    
    // Update Means
    for(int dm=0;dm<DIM;dm++){
      update_means(Mu[dm],dMu[dm],invn[dm]);
    }
    mu0 += dmu0*(stepsize/ntotal);
    dmu0=0;
    
    // fit 
    for(auto i=0;i<lng;i++){
      double est = mu0;
      for(int dm=0;dm<DIM;dm++){
        est += (Mu[dm])(coord(dm,i)); 
      }
      fit(i) = est;
    }
    
    // residual
    err = qty-fit;
    sse(it) = err.square().sum();
    double abs_err = sse[it];
    if(abs_err<abs_tol){break;}
    if(it>0){
      double rel_err = abs(sse[it-1]-abs_err)/(abs_err+ 1e-6);
      if(rel_err<rel_tol){ break;}
    }
  }

  // copy over
  Rcpp::List tubal_means(DIM+1);
  Eigen::VectorXd dim_counts = Eigen::VectorXd::Zero(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    tubal_means(dm)=Mu[dm];
    dim_counts(dm) = invn[dm];
  } 
  tubal_means(DIM) = mu0;
  dim_counts(DIM) = 1.0/double(ntotal);
  
  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("means") =  tubal_means,
                                           Rcpp::Named("fitted.values") =  fit,
                                           Rcpp::Named("residuals") =  err,
                                           Rcpp::Named("sse") =  sse.head(it),
                                           Rcpp::Named("inv.counts") = dim_counts);
  return ReturnMe;
}


// For weighed means
template<int DIM> Rcpp::List wtd_means_xd(const Rcpp::NumericVector & dims,
                                          const Eigen::MatrixXi & coord,
                                          const Eigen::ArrayXd & qty,
                                          const Eigen::ArrayXd & wt,
                                          int numit,double abs_tol, double rel_tol,double stepsize)
{
  
  
  // initialize stuff
  auto lng = qty.size();
  Eigen::ArrayXd sse = Eigen::ArrayXd::Zero(numit+1); 
  Eigen::ArrayXd err = Eigen::ArrayXd::Zero(qty.size());
  Eigen::ArrayXd fit = Eigen::ArrayXd::Zero(qty.size());
  Eigen::VectorXd Mu[DIM];
  Eigen::VectorXd dMu[DIM];
  Eigen::VectorXd isumwt[DIM];

  for(int i=0;i<DIM;i++){
    Mu[i] = Eigen::VectorXd::Zero(dims(i));
    dMu[i] = Eigen::VectorXd::Zero(dims(i));
    isumwt[i] = Eigen::VectorXd::Zero(dims(i));
  }
  
  // loop over and get normalizing constants
  double sumwt0 = 0;
  for(auto i=0;i<lng;i++){
    for(int dm=0;dm<DIM;dm++){
      (isumwt[dm])(coord(dm,i)) += wt(i);
    }
    sumwt0 += wt(i);
  }
  
  double threshold = (1e-9)*sumwt0/double(lng);
  for(int dm=0;dm<DIM;dm++){
    for(auto i =0;i<isumwt[dm].size();i++){
      double temp = isumwt[dm](i);
      isumwt[dm](i) = temp<=threshold?0:1.0/temp;
    }
   // isumwt[dm] = (isumwt[dm]).cwiseInverse().eval();

  }
  double mu0 = (wt.cwiseProduct(qty)).sum()/sumwt0;
  double dmu0=0;

  err = qty - mu0;
  
  
  //_____________________________
  // Iterate 
  //-----------------------------
  int it;
  for(it=0;it<numit;it++){
    Rcpp::checkUserInterrupt();// should we exit?
    if(it==1){
      for(int dm=0;dm<DIM;dm++){
       isumwt[dm] *= stepsize;
      } 
    }
    
    // re-estimate
    for(auto i=0;i<lng;i++){
      double wterr = wt(i)*err(i);
      for(int dm=0;dm<DIM;dm++){
        (dMu[dm])(coord(dm,i)) += wterr;
      }
      dmu0 += wterr;
    }
    
    // Update Means
    for(int dm=0;dm<DIM;dm++){
      update_means(Mu[dm],dMu[dm],isumwt[dm]);
    }
    mu0 += stepsize*(dmu0/sumwt0);
    dmu0=0;
    
    // fit 
    for(auto i=0;i<lng;i++){
      double est = mu0;
      for(int dm=0;dm<DIM;dm++){
        est += (Mu[dm])(coord(dm,i)); 
      }
      fit(i) = est;
    }
    
    // residual
    err = qty-fit;
    sse(it) = (wt*(err.square())).sum();
    double abs_err = sse[it];
    if(abs_err<abs_tol){break;}
    if(it>0){
      double rel_err = abs(sse[it-1]-abs_err)/(abs_err+ 1e-6);
      if(rel_err<rel_tol){ break;}
    }
  }

  // copy over
  Rcpp::List tubal_means(DIM+1);
  Rcpp::List dim_counts(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    tubal_means(dm)=Mu[dm];
    dim_counts(dm) = isumwt[dm];
  } 
  tubal_means(DIM) = mu0;
  dim_counts(DIM) = 1./sumwt0;
  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("means") =  tubal_means,
                                           Rcpp::Named("fitted.values") =  fit,
                                           Rcpp::Named("residuals") =  err,
                                           Rcpp::Named("sse") =  sse.head(it),
                                           Rcpp::Named("inv.counts") = dim_counts);
  return ReturnMe;
}
*/
#endif