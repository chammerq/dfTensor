
PLSA = function(obj,nlatent=2,niter=2,tolerances = c(1e-5,1e-5,1e-12),initialize = c('random','dsketch','svd'),warm_start=NULL,prior=NULL){
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor")
  }
  
  if(inherits(obj,"incompleteTensor")){
    stop("PLSA doesn't currently support missing values (obj inherited class: incompleteTensor)",call.=FALSE)
  }
  
  
  if(inherits(obj,"weightedTensor")){
    warning("PLSA doesn't currently support weightedTensors, ignoring weights",call.=FALSE)
  }
  
  verify_scalar(nlatent,"PLSA","n_latent",TRUE)
  verify_scalar(niter,"PLSA","niter",TRUE)
  verify_vector(tolerances,"PLSA","tolerances",TRUE)
  if(length(tolerances) != 3){
    stop("PLSA: tolerances must be a vector of length 3",call.=FALSE)
  }
  
  if(!is.null(warm_start)){
    stop("PLSA: warm start option not currently available",call.=FALSE)
  }
  
  if(!is.null(prior)){
    if(length(prior) != obj$rank){
      stop("PLSA: prior is not null but length is not equal to tensor rank")
    }
  }else{
    prior = rep(0,obj$rank)
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
  factors = .Call('_dfTensor_Initialize',PACKAGE = 'dfTensor',obj$coord,obj$value,init_type,0,nlatent,obj$kdim,obj$rank) 

  # Run C++ code to fit
  UpdateThese  = rep(1,obj$rank+5) # first fit, update all
  cout = .Call("_dfTensor_sparse_plsa_xd", PACKAGE = 'dfTensor',factors, obj$coord, obj$value,niter,tolerances,UpdateThese,prior,obj$rank)
  

  # Name and format out
  gnames = paste("Group",1:nlatent)
  for(k in 1:obj$rank){
    colnames(cout$factors[[k]]) = obj$Dimnames[[k]]
    rownames(cout$factors[[k]]) = gnames 
  }
  names(cout$factors[[k+1]]) = gnames
  names(cout$factors) = c(obj$mode_names,"Z")
  
  # add tensor stuff
  cout$factor_names = c(paste0("P(",obj$mode_names,"|z)"),"Pz")
  cout$Tensor = list(rank = obj$rank,kdim = obj$kdim,mode_names =obj$mode_names,
                     Dimnames = obj$Dimnames,class_list = class(obj),
                     nlatent = length(cout$factors[[k+1]]))
  class(cout) = c("dfTensor:Factorization","PLSA:Factorization")
  
  return(cout)
}

