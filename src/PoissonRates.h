#ifndef POISSON_RATES_HEADER
#define POISSON_RATES_HEADER

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// Function to calculate lambdas and rescale
inline void Calculate_Lambdas(Eigen::ArrayXd & Lambda,Eigen::ArrayXd & SumK,Eigen::ArrayXd & SumL,
                     double & threshold, double & lambda){

  Lambda *= (SumK+threshold)/(SumL+threshold);
  double muL = Lambda.sum()/double(SumK.size());
  Lambda /= muL;
  lambda *= muL;
  SumK.setZero();
  SumL.setZero();
}
// Function
template<int DIM> Rcpp::List poisson_rates_xd(const Rcpp::NumericVector & dims,
                                              const Eigen::MatrixXi & coord,
                                              const Eigen::ArrayXd & qty,
                                              int numit,double rel_tol,double threshold)
{

//_____________________________
// initialize stuff
//-----------------------------
auto lng = qty.size();
Eigen::ArrayXd lle = Eigen::ArrayXd::Zero(numit+1);
Eigen::ArrayXd fit = Eigen::ArrayXd::Zero(qty.size());
Eigen::ArrayXd SumK[DIM];
Eigen::ArrayXd SumL[DIM];
Eigen::ArrayXd Lambda[DIM];
Eigen::ArrayXi counts[DIM];

for(int dm=0;dm<DIM;dm++){
  SumK[dm] = Eigen::ArrayXd::Zero(dims(dm));
  SumL[dm] = Eigen::ArrayXd::Zero(dims(dm));
  counts[dm] = Eigen::ArrayXi::Zero(dims(dm));
  Lambda[dm] = Eigen::ArrayXd::Ones(dims(dm));

}
double sumk = qty.sum();
double lambda = sumk/double(lng);
//_____________________________
// Get counts

for(auto i=0;i<lng;i++){
  for(int dm=0;dm<DIM;dm++){
    (counts[dm])(coord(dm,i)) += 1;
  }
}
Rcpp::checkUserInterrupt();
//_____________________________
// Iterate
//-----------------------------
int it;
for(it=0;it<numit;it++){
  Rcpp::checkUserInterrupt();
  // Reset stuff
  double suml=0;
  double LLE = 0;

  // Loop over data
  for(auto i=0;i<lng;i++){
    // Predict
    double prodL = lambda;
    for(int dm=0;dm<DIM;dm++){
      prodL *= Lambda[dm](coord(dm,i));
    }
    fit(i) = prodL;
    LLE += prodL - log(prodL)*qty(i);

   // estimate rate
   for(int dm=0;dm<DIM;dm++){
     int j = coord(dm,i);
     SumK[dm](j) += qty(i);
     SumL[dm](j) += prodL; //Lambda[dm](j);
   }
   suml += prodL/lambda;
  }

  // MLE estimate
  lambda = sumk/suml;
  for(int dm=0;dm<DIM;dm++){
    Calculate_Lambdas(Lambda[dm],SumK[dm],SumL[dm],lambda,threshold);
  }
  // Check convergence
  lle(it) = LLE;
  if(it>0){
    double rel_err = (lle[it-1]-LLE)/(abs(LLE)+ 1e-6);
    if(rel_err<rel_tol){ break;}
  }
}

// copy over
Rcpp::List dims_rates(DIM+1);
Rcpp::List dim_counts(DIM+1);
for(int dm=0;dm<DIM;dm++){
  dims_rates(dm)=Lambda[dm];
  dim_counts(dm) = counts[dm];
}
dims_rates(DIM) = lambda;
dim_counts(DIM) = lng;

Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("rates") =  dims_rates,
                                         Rcpp::Named("fitted.values") =  fit,
                                         Rcpp::Named("lle") =  lle.head(it),
                                         Rcpp::Named("counts") = dim_counts);
return ReturnMe;
}

#endif
