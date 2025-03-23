
portal_tensor = function(original,df,new_coord_from = NULL,values_from=NULL,weights_from=NULL,remove_dim=-1,SparseAsMissing=FALSE){
  if(!inherits(original,"dfTensor")){
    stop("portal_tensor: original is not a dfTensor")
  }
  if(!inherits(df,"data.frame")){
    stop("portal_tensor: df must be a data.frame",call.=FALSE)
  }
  
  df_colnames = colnames(df)
  nrows = dim(df)[1]
  if(nrows==0){
    stop("portal_tensor: df has zero rows",call. = FALSE)
  }
  
  verify_scalar(SparseAsMissing,"portal_tensor","SparseAsMissing")
  
  # process values column
  if(length(values_from)!=1){
    stop("portal_tensor: Only one column can be used as value column. (length(values_from)!=1)",call.=FALSE)
  }
  if(!is.numeric(values_from)){
    values_from = match(values_from,df_colnames)
  }
  Values = df[,values_from]
  
  # new coordinates
  if(length(new_coord_from)!=1){
    stop("portal_tensor: Can only add 1 dimension at a time (length(new_coord_from)!=1)",call.=FALSE)
  }
  if(!is.numeric(new_coord_from)){
    new_coord_from = match(new_coord_from,df_colnames)
    if(any(is.na(new_coord_from))){
      stop("portal_tensor: The name given in new_coord_from doesn't exist as a column in df",call.=FALSE)
    }
  }

  # process weights column
  weighted=FALSE
  Weights = NULL
  if(!is.null(weights_from)){
    weighted=TRUE
    if(length(weights_from)!=1){
      stop("portal_tensor: Only one column can be used as weights column. (length(weights_from)!=1)",call.=FALSE)
    }
    if(!is.numeric(weights_from)){
      weights_from = match(weights_from,df_colnames)
    }
    Weights= df[,weights_from]
    verify_vector(this$weights,"dfTensor","weights",TRUE)
  }else{
    if(SparseAsMissing){
      Weights = rep(1.0,nrows) 
    }
  }
  
  # get coordinates
  coord_from = match(original$mode_names,df_colnames)
  if(any(is.na(coord_from))){
    naind = which(is.na(coord_from))
    stop(paste("portal_tensor: ",paste(original$mode_names[naind],sep=",",collapse=""),"columns not found in df"),call.=FALSE)
  }
  
  # copy over
  this = as.environment(as.list.environment(original, all.names=TRUE))

  # get original indices
  ogRank = original$rank
  ijk = matrix(0L,ncol=ogRank+1,nrow=nrows)
  for(k in 1:ogRank){
    coord_col = df[,coord_from[k]]
    dimA = this$Dimnames[[k]] 
    ijk[,k] = match(coord_col,dimA)-1 # convert to zero-based
  }
  
  # get new indices
  this$rank = ogRank+1
  coord_col = df[,new_coord_from]
  dimA = sort(unique(coord_col))
  this$Dimnames[[this$rank]] = dimA
  ijk[,this$rank] = match(coord_col,dimA)-1 # convert to zero-based
  this$kdim = c(this$kdim,length(dimA))
  
  # Check for extra coordinates
  isrowna = is.na(rowSums(ijk))
  if(any(isrowna)){
    warning("portal_tensor: there are indices in df that don't exist in original, removing...",call.=FALSE)
    ijk = ijk[!isrowna,]
    Values = Values[!isrowna]
    if(!is.null(Weights)){
      Weights = Weights[!isrowna]
    }
  }
  
  # sort for faster indexing
  #sorted_ind = sort_coordinates(ijk)
  #this$coord=t(ijk[sorted_ind,])
  #this$value=Values[sorted_ind]
  #if(is.null(Weights)){
  #  this$weight = -1
  #}else{
  #  this$weight = Weights[sorted_ind]
  #}

  # Don't sort, only used once
  this$coord=t(ijk)
  this$value=Values
  if(is.null(Weights)){
   this$weight = -1
  }else{
   this$weight = Weights
  }
  # get multiple class
  class_list = c("dfTensor",paste0("Rank",this$rank,"Tensor"))
  if(SparseAsMissing){
    class_list = c(class_list,"incompleteTensor")
  }
  if(weighted){
    class_list = c(class_list,"weightedTensor")
  }
  class(this)=class_list


    # update mode names
  mode_names = c(this$mode_names,df_colnames[new_coord_from])
  rownames(this$coord) = mode_names
  this$mode_names = mode_names
  
  # misc
  if(length(mode_names) != length(unique(mode_names))){
    warning("portal_tensor: mode_names not unique, no attempt to correct this has been made.")
  }
  this$sparsity = 1.0-nrows/prod(this$kdim)
  if(this$sparsity<0){
    stop("portal_tensor: input rows (for the indices columns) must be unique",call.=FALSE)
  }
  
  # remove dimension
  if(any(remove_dim>0)){
    this$mode_names = this$mode_names[-remove_dim]
    this$kdim = this$kdim[-remove_dim]
    this$rank = this$rank-1
    this$coord = this$coord[-remove_dim,]
    this$Dimnames = this$Dimnames[-remove_dim]
  }
  return(this)
  
}
