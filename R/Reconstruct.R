reconstruct = function(obj,tensor,newdf=NULL,Reorder=TRUE,return_as_df=TRUE){
  
  if(inherits(obj,"dfTensor:Factorization")){
    factors= obj$factors
  }else if(inherits(obj,"dfTensor:mean_polish")){
    factors= obj$means
  }else{
    stop("reconstruct:obj must be a dfTensor factorization object",call.=FALSE)     
  }

  # Try to screen for bad entries
  if(!inherits(tensor,"dfTensor")){
    stop("tensor must be of class dfTensor")
  }
  

  lng_obj = length(factors)
  if(lng_obj <(tensor$rank+1)){
    stop("obj doesn't match tensor (has less components then it should)")
  }
  
  nz=-1
  for(k in 1:tensor$rank){
    if(!identical(dim(factors[[k]])[2],tensor$kdim[k])){
      stop(paste0("Dimension 1 of component ",k," doesn't match corresponding of tensor"))
    }
    if(k==1){
      nz = dim(factors[[k]])[1]
    }else if(nz != dim(factors[[k]])[1]){
      stop(paste0("Dimension 2 of component ",k, " doesn't match other components"))
    }
  }
  if(length(factors[[k+1]])!= nz){
    stop("Length of vector component doesn't match dimenson 2 of other components")
  }
  
  # Convert newdf into format needed
  if(is.null(newdf)){  # just use coordinates of current tensor
    Coord = tensor$coord 
  }else if(inherits(newdf,"dfTensor")){  # use coordinates of new tensor
    if(!identical(tensor$kdim,newdf$kdim)){
      stop("Input newdf is a dfTensor, but it doesn't match dimensions of tensor")
    }
    Coord = newdf$coord
  }else if(is.data.frame(newdf)){ # otherwise some conversion is needed
    if(dim(newdf)[2]<tensor$rank){
      stop(paste0("newdf doesn't have enough columns, should have at least ",tensor$rank," columns"))
    }
   # verify_vector(as.matrix(newdf[,1:tensor$rank]),"reconstruct","newdf indices (first columns)")
    # line up columns
    names_df = colnames(newdf)
    if(!is.null(names_df)){
      orderN = match(tensor$mode_names,names_df)
      orderN = orderN[!is.na(orderN)]
      if(length(orderN) != tensor$rank){
        warning("column names of newdf don't line up with obj, attempting to use without names")
      }
    }else if(!Reorder){
      orderN=1:tensor$rank
    }else{
      orderN = tensor$original_column_order
    }
    # line up rows
    ijk_columns = newdf[,orderN]
    Coord = matrix(1L,tensor$rank,dim(newdf)[1])
    for(k in 1:tensor$rank){
      Coord[k,] = match(ijk_columns[,k],tensor$Dimnames[[k]])-1
    }
    
    # remove nas
    nna = colSums(is.na(Coord))
    if(any(nna>0)){
      warning("Some indices don't match tensor")
      Coord = Coord[,nna==0]
    }
  }else{
    stop("Input newdf is not null, a dataframe or a dfTensor")
  }
  
  #return(list(of = factors,cd =Coord,rnk=tensor$rank))
  new_value = .Call("_dfTensor_ReconstructSparseTensor", PACKAGE = 'dfTensor',factors,Coord,tensor$rank)
  return(new_value)
  if(return_as_df){
    nr = length(new_value)
    df_out = data.frame(matrix(0,nr,tensor$rank+1))
    for(k in 1:tensor$rank){
      df_out[,k] = tensor$Dimnames[[k]][Coord[k,]+1]
    }
    df_out[,k+1] = new_value
    colnames(df_out) = c(tensor$mode_names,"value") 
    return(df_out)    
  }else{
    return(new_value)
  }
  
  
}

