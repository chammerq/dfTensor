#ifndef RECONSTRUCT_HEADER
#define RECONSTRUCT_HEADER

#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

template<int D> inline void ReconstructSparse(Eigen::MatrixXd (& A)[D],
                                        Eigen::VectorXd & Z,
                                        const Eigen::MatrixXi & Coord,
                                        Eigen::VectorXd & fit){};



// Rank 2 Tensor
template<> inline void ReconstructSparse<2>(Eigen::MatrixXd (& A)[2],
                                            Eigen::VectorXd & Z,
                                            const Eigen::MatrixXi & Coord,
                                            Eigen::VectorXd & fit){
  
  
  Eigen::MatrixXd & Ai = A[0];
  Eigen::MatrixXd & Aj = A[1];

  // initialize
  int length = Coord.cols();
  int Nz = Z.size();
  // Outer product
  for(auto i=0;i<length;i++){
    int eye = Coord(0,i);
    int jay = Coord(1,i);
    double temp=0;
    for(int z=0;z<Nz;z++){
      temp += Ai(z,eye)*Aj(z,jay)*Z(z);
    }
    
    fit(i)=temp;
  }
  
}


// Rank 3 Tensor
template<> inline void ReconstructSparse<3>(Eigen::MatrixXd (& A)[3],
                                            Eigen::VectorXd & Z,
                                            const Eigen::MatrixXi & Coord,
                                            Eigen::VectorXd & fit){
  
  
  Eigen::MatrixXd & Ai = A[0];
  Eigen::MatrixXd & Aj = A[1];
  Eigen::MatrixXd & Ak = A[2];

  // initialize
  int length = Coord.cols();
  int Nz = Z.size();
  
  // Outer product
  for(auto i=0;i<length;i++){
    int eye = Coord(0,i);
    int jay = Coord(1,i);
    int kay = Coord(2,i);
 
    double temp=0;
    for(int z=0;z<Nz;z++){
      temp += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Z(z);
    }
    fit(i)=temp;
  }
  
}


// Rank 4 Tensor
template<> inline void ReconstructSparse<4>(Eigen::MatrixXd (& A)[4],
                                            Eigen::VectorXd & Z,
                                            const Eigen::MatrixXi & Coord,
                                            Eigen::VectorXd & fit){
  
  
  Eigen::MatrixXd & Ai = A[0];
  Eigen::MatrixXd & Aj = A[1];
  Eigen::MatrixXd & Ak = A[2];
  Eigen::MatrixXd & Al = A[3];

  // initialize
  int length = Coord.cols();
  int Nz = Z.size();
  
  // Outer product
  for(auto i=0;i<length;i++){
    int eye = Coord(0,i);
    int jay = Coord(1,i);
    int kay = Coord(2,i);
    int ell = Coord(3,i);
    double temp=0;
    for(int z=0;z<Nz;z++){
      temp += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Z(z);
    }
    fit(i)=temp;
  }
  
}

// Rank 5 Tensor
template<> inline void ReconstructSparse<5>(Eigen::MatrixXd (& A)[5],
                                   Eigen::VectorXd & Z,
                                   const Eigen::MatrixXi & Coord,
                                   Eigen::VectorXd & fit){
  

  Eigen::MatrixXd & Ai = A[0];
  Eigen::MatrixXd & Aj = A[1];
  Eigen::MatrixXd & Ak = A[2];
  Eigen::MatrixXd & Al = A[3];
  Eigen::MatrixXd & Am = A[4];
  
  // initialize
  int length = Coord.cols();
  int Nz = Z.size();

  // Outer product
  for(auto i=0;i<length;i++){
    int eye = Coord(0,i);
    int jay = Coord(1,i);
    int kay = Coord(2,i);
    int ell = Coord(3,i);
    int emm = Coord(4,i);
    double temp=0;
    for(int z=0;z<Nz;z++){
      temp += Ai(z,eye)*Aj(z,jay)*Ak(z,kay)*Al(z,ell)*Am(z,emm)*Z(z);
    }
    fit(i)=temp;
  }
  
  
}

template<int DIM> Eigen::VectorXd  ReconstructSparseFromList(const Rcpp::List & Factors,
                                                const Eigen::MatrixXi & Coord){
  
  // Initialize Matrices, copy over
  Eigen::MatrixXd A[DIM];
  for(int i=0;i<DIM;i++){
    A[i] = Factors(i);
  }
  Eigen::VectorXd Z = Factors(DIM);
  Eigen::VectorXd fit = Eigen::VectorXd::Zero(Coord.cols());
  ReconstructSparse<DIM>(A,Z,Coord,fit);
  return fit;

};




template<int DIM> double ReconstructSparse_w_resid(Eigen::MatrixXd (& A)[DIM],
                                              Eigen::VectorXd & Z,
                                              const Eigen::MatrixXi & Coord,
                                              Eigen::VectorXd & fit,
                                              const Eigen::VectorXd & qty,
                                              Eigen::VectorXd & resid){
  
  ReconstructSparse<DIM>(A,Z,Coord,fit);
  resid = qty-fit;
  return resid.norm();

};

template<int DIM> double ReconstructSparse_w_resid(Eigen::MatrixXd (& A)[DIM],
                                                 Eigen::VectorXd & Z,
                                                 const Eigen::MatrixXi & Coord,
                                                 Eigen::VectorXd & fit,
                                                 const Eigen::VectorXd & qty,
                                                 Eigen::VectorXd & resid,
                                                 const Eigen::VectorXd & wt){
  
  ReconstructSparse<DIM>(A,Z,Coord,fit);
  resid = qty-fit;
  double sqnorm = (resid.array()*wt.array()*resid.array()).sum();
  return sqrt(sqnorm);
  
};



#endif