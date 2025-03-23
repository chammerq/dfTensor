recommend = function(obj,Factored,predict_mode=NULL,predict_ind=NULL,type=c("PLSA","CP_TD","WCP_TD"),lambda=1e-6,wtd_it=-1){
  # Try to screen for bad entries
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor")
  }

  # Get initial factors
  tRank = obj$rank
  factors = list()
  Z = Factored[[tRank]]
  factors[[tRank + 1]] = Z
  for(k in 1:(tRank-1)){
    factors[[k]] = Factored[[k]]  
  }
  factors[[tRank]] = Z%o%rep(1,obj$kdim[tRank])*(.5+0.1*runif(length(Z)*obj$kdim[tRank]))
  
  UpdateThese  = rep(0,obj$rank+2) # first fit, update all
  UpdateThese[tRank] = 1
  
  # Fit one
  type = match.arg(type)
  if(type=="WCP_TD"){
    if(!inherits(obj,"incompleteTensor")&!inherits(obj,"weightedTensor")){
      stop("recommend: Can't use type='WCP_TD' for this type of tensor, try 'CP_TD' instead",call.=FALSE)
    }

    lambdas_etc = c(lambda,0,0,0,0,0)
    cout = .Call("_dfTensor_sparse_als_wt_cp", PACKAGE = 'dfTensor',factors,
                 obj$coord, obj$value,obj$weights,c(1,0,wtd_it),lambdas_etc,UpdateThese,obj$rank)      
  }else if(type=="CP_TD"){
    lambdas_etc = c(lambda,0,0,0,0,0)
    cout = .Call("_dfTensor_sparse_als_cp", PACKAGE = 'dfTensor',factors,
                 obj$coord, obj$value,c(1,0,-1),lambdas_etc,UpdateThese,obj$rank)  
  }else if(type=="PLSA"){
    tolerances = c(0,0,lambda)
    cout = .Call("_dfTensor_sparse_plsa_xd", PACKAGE = 'dfTensor',factors,
                 obj$coord,obj$value,2,tolerances,UpdateThese,obj$rank)
  }
  
  # output
  predict_matrix = rowSums(Factored[[predict_mode]][,predict_ind,drop=FALSE])
  return(t(cout$factors[[tRank]])%*%predict_matrix)
}
