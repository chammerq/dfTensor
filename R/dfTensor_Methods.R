df2Tensor = function(df,SparseAsMissing,values_from=NULL,coord_from=NULL,weights_from=NULL,checkUnique=FALSE){
  
  Reorder=TRUE # option to rearrange columns
  this = list2env(list())
  
  if(!inherits(df,"data.frame")){
    stop("df2Tensor: df must be a data.frame",call.=FALSE)
  }
   if(inherits(df,"tbl")||inherits(df,"tbl_df")){
   df = as.data.frame(df)
    #stop("df2Tensor: Tibbles are not currently supported",call.=FALSE)
  }
  df_colnames = colnames(df)
  nrows = dim(df)[1]
  if(nrows==0){
    stop("df2Tensor: data.frame is has zero rows",call. = FALSE)
  }
  
  verify_scalar(SparseAsMissing,"df2Tensor","SparseAsMissing")
  
  # process values column
  if(length(values_from)!=1){
    stop("df2Tensor: Only one column can be used as value column. (length(values_from)!=1)",call.=FALSE)
  }
  if(!is.numeric(values_from)){
    values_from = match(values_from,df_colnames)
    if(any(is.na(values_from))){
      stop("dfTensor: Some of the names given in coord_from don't exist as columns in df",call.=FALSE)
    }
  }else if(max(values_from)>length(df_colnames) || min(values_from)<=0){
    stop("dfTensor: Some of the values in coord_from are outside the dimension of df",call.=FALSE)
  }
  Values = df[,values_from]
  #verify_vector(this$value,"dfTensor","values")
  
  # process weights column
  weighted=FALSE
  Weights = NULL
  if(!is.null(weights_from)){
    weighted=TRUE
    if(length(weights_from)!=1){
      stop("df2Tensor: Only one column can be used as weights column. (length(weights_from)!=1)",call.=FALSE)
    }
    if(!is.numeric(weights_from)){
      weights_from = match(weights_from,df_colnames)
    }
    Weights= df[,weights_from]
  }else{
    if(SparseAsMissing){
      Weights = rep(1.0,nrows) 
    }
  }
  
  # process coordinate column
  tRank = length(coord_from)
  this$rank = tRank
  if(tRank>5){
    stop("df2Tensor: Currently supports tensors up to rank 5. (length(coord_from)>5)",call.=FALSE)
  }
  if(tRank<=1){
    stop("dfTensor: rank 1 tensors (vectors) not supported (length(coord_from)<=1)")
  }
  if(!is.numeric(coord_from)){
    coord_from = match(coord_from,df_colnames)
    if(any(is.na(coord_from))){
      stop("dfTensor: Some of the names given in coord_from don't exist as columns in df",call.=FALSE)
    }
  }else if(max(coord_from)>length(df_colnames) || min(coord_from)<=0){
      stop("dfTensor: Some of the values in coord_from are outside the dimension of df",call.=FALSE)
  }
  
  ijk = matrix(0L,ncol=tRank,nrow=nrows)
  this$kdim = 1:tRank
  for(k in 1:tRank){
    coord_col = df[,coord_from[k]]
    dimA = sort(unique(coord_col))
    this$kdim[k] = length(dimA)
    this$Dimnames[[k]] = dimA
    ijk[,k] = match(coord_col,dimA)-1 # convert to zero-based
  }

  # sort indices, starting with the longest
  if(Reorder){
    orderN = order(this$kdim,decreasing = TRUE)  
  }else{
    orderN = 1:this$rank
  }
  
  this$original_column_order = orderN
  this$kdim = this$kdim[orderN]

  # sort columns
  is_unique = list(TRUE) # pass in by reference
  ijk = t(ijk[,orderN]) 
  sorted_ind = .Call("_dfTensor_sort_indices", PACKAGE = 'dfTensor',ijk,is_unique)
  this$coord=ijk[,sorted_ind]
  
  this$value=Values[sorted_ind]
  if(is.null(Weights)){
    this$weights = -1
  }else{
    this$weights = Weights[sorted_ind]
  }
  
  # Check uniques
  is_unique = unlist(is_unique)
  if(checkUnique & !is_unique){
      stop("dfTensor: input rows (for the indices columns) are not unique and a fail was requested (by you)",call.=FALSE)
  }
  this$is_unique = is_unique
  
  # get column names for coordinates
  mode_names = (df_colnames[coord_from])[orderN]
  if(is.null(mode_names)){
    mode_names = paste("Mode ",1:tRank)
  }
  rownames(this$coord) = mode_names
  this$mode_names = mode_names
  # re-order dimnames
  this$Dimnames = this$Dimnames[orderN]
  names(this$Dimnames) = mode_names
  
  # get multiple class
  class_list = c("dfTensor",paste0("Rank",tRank,"Tensor"))
  if(SparseAsMissing){
    class_list = c(class_list,"incompleteTensor")
  }
  if(weighted){
    class_list = c(class_list,"weightedTensor")
  }
  class(this)=class_list
  
  # misc
  this$sparsity = 1.0-nrows/prod(this$kdim)
  if((this$sparsity<0|!this$is_unique)&checkUnique){
    warning("dfTensor: input rows (for the indices columns) are not unique, might cause some problems...",call.=FALSE)
  }
  return(this)
  
}

print.dfTensor= function(x){
  cat("A rank",x$rank,"dfTensor\n\t with dimensions:")
  cat("[",x$kdim,"]\n")
  cat("\t Density (1.0-Sparse) : ",signif(100*(1-x$sparsity),3),"%\n")
  if(inherits(x,"incompleteTensor")){
    cat("\t Sparse (zero) values are considered as missing values\n")
  }
  if(inherits(x,"weightedTensor")){
    cat("\t Includes data weights")
  }
  if(!x$is_unique){
    cat("\t\t Indices not unique (sparsity is probably understated)\n")
  }
}

# make a deep copy
`%=%` = function(lhs,rhs){
  lhs = deparse(substitute(lhs))
  if(inherits(rhs,"dfTensor")){
    tmp = as.environment(as.list.environment(rhs, all.names=TRUE))
    class(tmp) = 'dfTensor'
    assign(lhs,tmp,envir=parent.frame())
  }else{
    assign(lhs,rhs,envir=parent.frame())
  }
}

# Convert tensor back
tensor2df = function(obj){
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor",call.=FALSE)
  }
  df = data.frame(values = obj$value)
  if(inherits(obj,"weightedTensor")){
    df$weights = obj$weights
  }
  for(k in 1:obj$rank){
    df[[obj$mode_names[k]]] = obj$Dimnames[[k]][obj$coord[k,]+1]
  }
  return(df)
}

# Transform tensor
transform_tensor = function(obj,fun=I()){
  if(!inherits(obj,"dfTensor")){
    stop("obj must be of class dfTensor",call.=FALSE)
  }
  if(!inherits(obj,"incompleteTensor")){
    test = fun(0)
    if(test != 0){
      stop("transform_tensor: function needs to map zeros to zeros to maintain sparsity",call.=FALSE)
    }
  }
  if(length(fun(c(1:4)))!= 4){
    stop("transform_tensor: function doesn't preserve the length of inputs",call.=FALSE)
  }
  obj$value = fun(obj$value)
  invisible(obj)
}