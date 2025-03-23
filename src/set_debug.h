// Use this file to turn on/off Eigen debug mode
#ifndef SETMYDEBUG
#define SETMYDEBUG


#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#include <chrono>
#include <thread>


#define MYDEBUG(stuff)   Rcpp::Rcerr<<"Debug: " <<stuff<<std::endl;      \
std::this_thread::sleep_for(std::chrono::milliseconds(10));     


#endif
