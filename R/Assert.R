
verify_scalar = function(input,name1,name2,is_nonnegative=FALSE){
  if(length(input)>1){
    stop(paste0(name1,": ",name2," must be a scalar"),call.=FALSE)
  }
  if(is.infinite(input)){
    stop(paste0(name1,": ",name2," must be a finite"),call.=FALSE)
  }
  if(is.na(input)){
    stop(paste0(name1,": missing ",name2," not allowed"),call.=FALSE)
  }
  if(is_nonnegative){
    if(input<0){
      stop(paste0(name1,": ",name2," must be non-negative"),call.=FALSE)
    }
  }
}

verify_vector = function(input,name1,name2,is_nonnegative=FALSE){
  
  if(any(is.infinite(input))){
    stop(paste0(name1,": ",name2," must be a finite"),call.=FALSE)
  }
  if(any(is.na(input))){
    stop(paste0(name1,": missing values in",name2," not allowed"),call.=FALSE)
  }
  if(is_nonnegative){
    if(any(input<0)){
      stop(paste0(name1,": ",name2," must be non-negative"),call.=FALSE)
    }
  }
}
