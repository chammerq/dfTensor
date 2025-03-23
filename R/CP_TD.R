
CP_TD= function(obj,nlatent=2,als_it=2,lambdas=c(1e-6,0.001),conv_tol=c(1e-6,1e-6),
                initialize = c('random','dsketch'),ortho_it=-1,wtd_it=-1,warm_start=NULL){
  # Try to screen for bad entries
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor")
  }
  
  verify_scalar(nlatent,"CP_TD","nlatent",TRUE)
  verify_scalar(als_it,"CP_TD","als_it",FALSE)
  verify_scalar(wtd_it,"CP_TD","wtd_it",FALSE)
  verify_vector(lambdas,"CP_TD","lambdas",TRUE)
  verify_vector(conv_tol,"CP_TD","conv_tol",TRUE)
  if(length(lambdas)<2){
    lambdas[2] = 1e-8
  }
  if(length(conv_tol)<2){
    conv_tol[2] = 1e-12
  }
  lambdas_etc = c(lambdas,conv_tol)
  
  if(!is.null(warm_start)){
    stop("warm start option not currently available",call.=FALSE)
  }

  # Get initial factors 
  initialize = match.arg(initialize)
  if(initialize == 'random'){
    init_type = 0
  }else if(initialize == 'dsketch' || initialize == 'svd'){
    init_type = 1
  }else{
    stop(paste0("Method for initialization ",initialize," not currently implemented"),call. = FALSE)
  }
  factors = .Call('_dfTensor_Initialize',PACKAGE = 'dfTensor',obj$coord,obj$value,init_type,1,nlatent,obj$kdim,obj$rank) 

  # Get number of iterations, check weighted
  is_weighted=FALSE
  number_it = c(als_it,ortho_it,wtd_it)
  if(inherits(obj,"weightedTensor")||inherits(obj,"incompleteTensor")){
    if(wtd_it != 0){
      is_weighted = TRUE
    }
  }
  
  # Run C++ code to fit
  UpdateThese  = rep(1,obj$rank+2) # first fit, update all
  if(is_weighted){
    cout = .Call("_dfTensor_sparse_als_wt_cp", PACKAGE = 'dfTensor',
                 factors, obj$coord, obj$value,obj$weights,number_it,lambdas_etc,UpdateThese,obj$rank)      
  }else{
      cout = .Call("_dfTensor_sparse_als_cp", PACKAGE = 'dfTensor',
                   factors, obj$coord, obj$value,number_it,lambdas_etc,UpdateThese,obj$rank)   
  }

  # Name and format out
  gnames = paste("Group",1:nlatent)
  for(k in 1:obj$rank){
    colnames(cout$factors[[k]]) = obj$Dimnames[[k]]
    rownames(cout$factors[[k]]) = gnames 
  }
  names(cout$factors[[k+1]]) = gnames
  names(cout$factors) = c(obj$mode_names,"Z")

  # add tensor stuff
  cout$Tensor = list(rank = obj$rank,kdim = obj$kdim,mode_names =obj$mode_names,
                     Dimnames = obj$Dimnames,class_list = class(obj),
                     nlatent = length(cout$factors[[k+1]]))
  
  class(cout) = c("dfTensor:Factorization","CP_TD:Factorization")
    
    
    
  # return
  return(cout)
}
