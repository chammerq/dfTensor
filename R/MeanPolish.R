
# mean polish
mean_polish = function(obj,numit,conv_tol=c(1e-6,1e-6),stepsize=1){
  # Try to screen for bad entries
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor")
  }

 # verify_vector(conv_tol,"CenterDimensions","conv_tol",TRUE)
  if(length(conv_tol)<2){
    conv_tol[2] = 1e-12
  }

  isweighted = (inherits(obj,"weightedTensor")||inherits(obj,"incompleteTensor"))
  # Run C++ code to fit
  if(isweighted){
    cout = .Call("_dfTensor_Mean_polish_wtd", PACKAGE = 'dfTensor',
                 obj$kdim, obj$coord, obj$value,obj$weights,numit,conv_tol[1],conv_tol[2],stepsize,obj$rank)  
  }else{
    cout = .Call("_dfTensor_Mean_polish", PACKAGE = 'dfTensor',
                 obj$kdim, obj$coord, obj$value,numit,conv_tol[1],conv_tol[2],stepsize,obj$rank)    
  }

  # Name and format out
  for(k in 1:obj$rank){
    names(cout$means[[k]]) = obj$Dimnames[[k]]
    names(cout$inv.counts[[k]]) = obj$Dimnames[[k]]
  }
  names(cout$means) = c(obj$mode_names,"mu0")
  names(cout$inv.counts) = c(obj$mode_names,"mu0")

  class(cout) = c("dfTensor:mean_polish")
  return(cout)
}
