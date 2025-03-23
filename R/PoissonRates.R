poisson_rates = function(obj,numit,conv_tol=1e-6,threshold=1e-12){
  # Try to screen for bad entries
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor")
  }
  
  if(!inherits(obj,"incompleteTensor")){
    stop("Currently only implemented for Tensor where sparse values are missing values",call.=FALSE)
  }
  if(!inherits(obj,"weightedTensor")){
    warning("poisson_rates: Weights are currently ignored",call.=FALSE)
  }
  # Run C++ code to fit
  cout = .Call("_dfTensorPlus_poisson_rates", PACKAGE = 'dfTensorPlus',
               obj$kdim, obj$coord, obj$value,numit,conv_tol,threshold,obj$rank)
  
  # Name and format out
  for(k in 1:obj$rank){
    names(cout$rates[[k]]) = obj$Dimnames[[k]]
    names(cout$counts[[k]]) = obj$Dimnames[[k]]
  }
  names(cout$rates) = c(obj$mode_names,"lambda0")
  
  class(cout) = c("dfTensor:poisson_rates")
  return(cout)
}
