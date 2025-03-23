#ifndef PLSA_HEADER
#define PLSA_HEADER

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include "common_functions.h"

//***************************************************
// Note: This code is based on the equations 3-6 in the original PLSA paper.
// Probablistic Latent Semantic Analysis by Thomas Hofmann, https://arxiv.org/pdf/1301.6705
// Some of the calculations are shifted around to different steps to reduce memory usage
// Regularization is based on:
// CONJUGATE-PRIOR-REGULARIZED MULTINOMIAL PLSA FOR COLLABORATIVE FILTERING
// by Marcus Klasson
// https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8879554&fileOId=8879567

// https://course.ccs.neu.edu/cs6220f16/sec3/assets/pdf/hong-plsa-tutorial.pdf (Section 5.2 can be helpful)
//***************************************************

// Template function to update shift
template<int DIM> inline void rotate_pointers(Eigen::MatrixXd* (&A)[DIM],Eigen::MatrixXd* (&tA)[DIM],
                                              Eigen::VectorXd* PZ,Eigen::VectorXd* tPZ,
                                              const Rcpp::NumericVector & UpdateThese){
  for(int dm =0;dm<DIM;dm++){
    if(UpdateThese(dm)>0){
      std::swap(A[dm],tA[dm]);
    }
  }
  if(UpdateThese(DIM)>0){
    std::swap(PZ,tPZ);
  }
}

// Reset array
template<int DIM> inline void reset_array(Eigen::MatrixXd* (& A)[DIM], Eigen::VectorXd* PZ){
  for(int dm =0;dm<DIM;dm++){
    A[dm]->setZero();
  }
  PZ->setZero();
}

// add constant to array
inline void add2array(Eigen::MatrixXd & A, double gamma){
  auto ptr = A.data();
  for (auto i = 0; i < A.size(); i++){
    *(ptr+i) += gamma;
  }
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Template to calculate expectation step
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<int DIM> inline double calculate_Expectation( Eigen::MatrixXd* (& PA)[DIM],
                                                        Eigen::VectorXd* PZ,
                                                        const Eigen::MatrixXi & Cood,
                                                        int nz,int i){};

template<> inline double calculate_Expectation<2>( Eigen::MatrixXd* (& PA)[2],
                                                   Eigen::VectorXd* PZ,
                                                   const Eigen::MatrixXi & Cood,
                                                   int nz,int i){
  
  int eye = Cood(0,i);
  int jay = Cood(1,i);

  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];

  double tP_ai =0;
  for(int z=0;z<nz;z++){
    tP_ai += Ai(z,eye)*Aj(z,jay)*Pz(z);
  }
  
  return tP_ai;
  
};



template<> inline double calculate_Expectation<3>(Eigen::MatrixXd* (& PA)[3],
                                                  Eigen::VectorXd* PZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  int nz,int i){
  
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);
  
  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];

  double tP_ai =0;
  for(int z=0;z<nz;z++){
    tP_ai += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Pz(z);
  }
  
  return tP_ai;
};

template<> inline double calculate_Expectation<4>(Eigen::MatrixXd* (& PA)[4],
                                                  Eigen::VectorXd* PZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  int nz,int i){
  
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);
  int ell = Cood(3,i);

  
  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];
  Eigen::MatrixXd & Al = *PA[3];

  
  double tP_ai =0;
  for(int z=0;z<nz;z++){
    tP_ai += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Pz(z);
  }
  
  return tP_ai;
};

template<> inline double calculate_Expectation<5>(Eigen::MatrixXd* (& PA)[5],
                                                  Eigen::VectorXd* PZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  int nz,int i){
  
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);
  int ell = Cood(3,i);
  int emm = Cood(4,i);
  
  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];
  Eigen::MatrixXd & Al = *PA[3];
  Eigen::MatrixXd & Am = *PA[4];
  
  double tP_ai =0;
  for(int z=0;z<nz;z++){
    tP_ai += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Am(z,emm)*Pz(z);
  }
  
  return tP_ai;
  
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Template to calculate maximization step
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<int DIM> inline void calculate_Maximization(Eigen::MatrixXd* (& PA)[DIM],
                                                        Eigen::MatrixXd* (& tPA)[DIM],
                                                        Eigen::VectorXd* PZ,
                                                        Eigen::VectorXd* tPZ,
                                                        const Eigen::MatrixXi & Cood,
                                                        double ratio,int nz,int i){};




template<> inline void calculate_Maximization<2>( Eigen::MatrixXd* (& PA)[2],
                                                  Eigen::MatrixXd* (& tPA)[2],
                                                  Eigen::VectorXd* PZ,
                                                  Eigen::VectorXd* tPZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  double ratio,int nz,int i){
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  
  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  
  Eigen::VectorXd & tPz = *tPZ;
  Eigen::MatrixXd & tAi = *tPA[0];
  Eigen::MatrixXd & tAj = *tPA[1];
  
  for(int z=0;z<nz;z++){
    double temp = tAi(z,eye)*tAj(z,jay)*tPz(z)*ratio;
    Pz(z) += temp; 
    Ai(z,eye) += temp;
    Aj(z,jay) += temp;
  }
};


template<> inline void calculate_Maximization<3>( Eigen::MatrixXd* (& PA)[3],
                                                  Eigen::MatrixXd* (& tPA)[3],
                                                  Eigen::VectorXd* PZ,
                                                  Eigen::VectorXd* tPZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  double ratio,int nz,int i){
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);

  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];
  
  Eigen::VectorXd & tPz = *tPZ;
  Eigen::MatrixXd & tAi = *tPA[0];
  Eigen::MatrixXd & tAj = *tPA[1];
  Eigen::MatrixXd & tAk = *tPA[2];
  
  for(int z=0;z<nz;z++){
    double temp = tAi(z,eye)*tAj(z,jay)*tAk(z,kay)*tPz(z)*ratio;
    Pz(z) += temp; 
    Ai(z,eye) += temp;
    Aj(z,jay) += temp;
    Ak(z,kay) += temp;
  }
};



template<> inline void calculate_Maximization<4>( Eigen::MatrixXd* (& PA)[4],
                                                  Eigen::MatrixXd* (& tPA)[4],
                                                  Eigen::VectorXd* PZ,
                                                  Eigen::VectorXd* tPZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  double ratio,int nz,int i){
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);
  int ell = Cood(3,i);

  
  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];
  Eigen::MatrixXd & Al = *PA[3];

  Eigen::VectorXd & tPz = *tPZ;
  Eigen::MatrixXd & tAi = *tPA[0];
  Eigen::MatrixXd & tAj = *tPA[1];
  Eigen::MatrixXd & tAk = *tPA[2];
  Eigen::MatrixXd & tAl = *tPA[3];

  for(int z=0;z<nz;z++){
    double temp = tAi(z,eye)*tAj(z,jay)*tAk(z,kay)*tAl(z,ell)*tPz(z)*ratio;
    Pz(z) += temp; 
    Ai(z,eye) += temp;
    Aj(z,jay) += temp;
    Ak(z,kay) += temp;
    Al(z,ell) += temp;
  }
};


template<> inline void calculate_Maximization<5>( Eigen::MatrixXd* (& PA)[5],
                                                  Eigen::MatrixXd* (& tPA)[5],
                                                  Eigen::VectorXd* PZ,
                                                  Eigen::VectorXd* tPZ,
                                                  const Eigen::MatrixXi & Cood,
                                                  double ratio,int nz,int i){
  int eye = Cood(0,i);
  int jay = Cood(1,i);
  int kay = Cood(2,i);
  int ell = Cood(3,i);
  int emm = Cood(4,i);

  Eigen::VectorXd & Pz = *PZ;
  Eigen::MatrixXd & Ai = *PA[0];
  Eigen::MatrixXd & Aj = *PA[1];
  Eigen::MatrixXd & Ak = *PA[2];
  Eigen::MatrixXd & Al = *PA[3];
  Eigen::MatrixXd & Am = *PA[4];
  
  Eigen::VectorXd & tPz = *tPZ;
  Eigen::MatrixXd & tAi = *tPA[0];
  Eigen::MatrixXd & tAj = *tPA[1];
  Eigen::MatrixXd & tAk = *tPA[2];
  Eigen::MatrixXd & tAl = *tPA[3];
  Eigen::MatrixXd & tAm = *tPA[4];
  

  for(int z=0;z<nz;z++){
    double temp = tAi(z,eye)*tAj(z,jay)*tAk(z,kay)*tAl(z,ell)*tAm(z,emm)*tPz(z)*ratio;
    Pz(z) += temp; 
    Ai(z,eye) += temp;
    Aj(z,jay) += temp;
    Ak(z,kay) += temp;
    Al(z,ell) += temp;
    Am(z,emm) += temp;
  }
  
};





// Template definition of ALS
template<int DIM> Rcpp::List sparse_plsa_xd(Rcpp::List & Factors,
                                              const Eigen::MatrixXi & cood,
                                              const Eigen::VectorXd & qty,
                                              int  number_it,
                                              const Rcpp::NumericVector & tolerances,
                                              const Rcpp::NumericVector & UpdateThese,
                                              const Rcpp::NumericVector & prior){
  

  // rename 
  double abs_tol = tolerances(0);
  double rel_tol = tolerances(1);

  bool TrackError = UpdateThese(DIM+1)>0;
  int num = cood.cols();

  // Initialize Matrices and vectors
  Eigen::VectorXd conv = Eigen::VectorXd::Zero(number_it+1); 
  Eigen::MatrixXd Pa_z[DIM];
  Eigen::MatrixXd tPa_z[DIM];

  Eigen::MatrixXd * ptr_Pa_z[DIM];
  Eigen::MatrixXd * ptr_tPa_z[DIM];
  Eigen::VectorXd * ptr_Pz;
  Eigen::VectorXd * ptr_tPz;
  
  
  Eigen::VectorXd P_ten(qty.size());

  double tensize = 1.0; // total size of full tensor
  for(int i=0;i<DIM;i++){
    Pa_z[i] = Factors(i);
    tPa_z[i] = Factors(i);
    ptr_Pa_z[i] = &Pa_z[i];
    ptr_tPa_z[i] = &tPa_z[i];
    tensize *= double(Pa_z[i].cols());
  }
  Eigen::VectorXd Pz = Factors(DIM);
  Eigen::VectorXd tPz = Factors(DIM);
  ptr_Pz = &Pz;
  ptr_tPz = &tPz;
  int nz = Pz.size();
  
  // Set threshold for expectation
  double threshhold = tolerances(2)/tensize;
  
  // Scale error to make sense
  double min_cost = 0.0;
  double max_cost = 1.0;
  if(TrackError){
    double qsum = qty.sum();
    max_cost = qsum*log(tensize);
    Eigen::ArrayXd tmp = (qty.array()+1e-9).log() - log(qsum);
    min_cost = -(qty.array()*tmp).sum();
  }
  double scale_cost = 1.0/(abs(max_cost - min_cost) + 1e-14);

  //===================================
  // Iterate the EM algorithm
  //===================================
  int it;
  for(it=0;it<number_it;it++){
    Rcpp::checkUserInterrupt();// should we exit?
    //------------------------
    // E step
    //------------------------
    double cost=0;
    // loop over data
    for(auto i=0;i<num;i++){
      double tP_ijk =calculate_Expectation<DIM>(ptr_Pa_z,ptr_Pz,cood,nz,i);
      tP_ijk = (tP_ijk<threshhold)?threshhold:tP_ijk;
      
      // Update cost
      cost -= log(tP_ijk)*qty(i); 

      // store updated tensor fit
      P_ten(i) = tP_ijk;
    }
 
    conv[it] = scale_cost*(cost - min_cost);
    //------------------------
    // Check Convergence
    //------------------------
    if(TrackError && (it>0)){
      if(conv[it]<abs_tol){
        break;
      }
      if((abs(conv[it]-conv[it-1])/(conv[it-1]+1e-12))<rel_tol){
       break;
      }
    }
    //------------------------
    // Toggle temporary arrays
    //------------------------
    rotate_pointers(ptr_Pa_z,ptr_tPa_z,ptr_Pz,ptr_tPz,UpdateThese);
    //------------------------
    // M step
    //------------------------
    reset_array(ptr_Pa_z,ptr_Pz); // set temporary arrays to zero
    // loop over data,    
    for(auto i=0;i<num;i++){
      double commonRatio = qty(i)/P_ten(i);
      calculate_Maximization<DIM>(ptr_Pa_z,ptr_tPa_z,ptr_Pz,ptr_tPz,cood,commonRatio,nz,i);
    }
    
    //-----------------------------
    // Maybe regularize?
    //-----------------------------
    for(int dm =0;dm<DIM;dm++){
      double gamma = prior(dm);
      if(gamma>0.0){
        add2array(*ptr_Pa_z[dm],gamma);    
      }
    }
    
      
    //------------------------
    //Normalize
    //------------------------
    for(int dm =0;dm<DIM;dm++){
      rescale_back2_prob(*ptr_Pa_z[dm],threshhold);
    }
    rescale_back2_prob(*ptr_Pz,threshhold);

  }
  
  // copy back to list
  Rcpp::List factor_matrices(DIM+1);
  for(int dm=0;dm<DIM;dm++){
    factor_matrices(dm) = *ptr_Pa_z[dm]; 
  }
  factor_matrices(DIM) = *ptr_Pz;
  Rcpp::List ReturnMe = Rcpp::List::create(Rcpp::Named("factors") =  factor_matrices,
                                           Rcpp::Named("scaleLLE") =  conv.head(it));
  
  return ReturnMe;

}

  
#endif