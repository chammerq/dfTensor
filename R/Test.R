if(FALSE){
  detach("package:dfTensor", unload = TRUE)
  library.dynam.unload("dfTensor", "C:/Users/chammerquist/Documents/R/R-4.3.1/library/dfTensor")
  
  #devtools::load_all(".")
  # know the right answer test
  x = (1:10)*2*pi/10
  y = (1:20)*2*pi/20
  z = (1:30)*2*pi/30
  w = (1:19)*2*pi/19
  u = (1:24)*2*pi/24
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Dense Tensors
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # size 2 tensor
  ten2 = (sin(x)+1)%o%sin(y)+sin(2*x)%o%sin(2*y)
  dimnames(ten2) = list(paste0("x",1:10),paste0("y",1:20))
  back2 = as.data.frame.table(ten2)
  myten2 = df2Tensor(back2,FALSE,values_from="Freq",coord_from=c("Var1","Var2"))
  out2 = CP_TD(myten2,3,1,lambdas = c(0,.1),initialize = "svd")
  ts.plot(out2$sse);min(out2$sse)
  tmp = reconstruct(out2,myten2,back2)
  
  # size 3 tensor
  ten3 = sin(x)%o%sin(y)%o%sin(z) +sin(2*x)%o%sin(2*y)%o%sin(2*z)+sin(.5*x)%o%(y*0+1)%o%(z*0+1)+(0*x+1)%o%cos(y)%o%(z*0+1)
  dimnames(ten3) = list(paste0("x",1:10),paste0("y",1:20),paste0("z",1:30))
  back3 = as.data.frame.table(ten3)
  myten3 = df2Tensor(back3,TRUE,values_from="Freq",coord_from=c("Var1","Var2","Var3"))
  out3 = CP_TD(myten3,2,10,lambdas = c(0.0,.0),initialize = "svd")
  ts.plot(out3$sse);min(out3$sse)
  ts.plot(t(out3$factors$Var3))

  # size 4 tensor
  ten4 = sin(x)%o%sin(y)%o%sin(z)%o%sin(w)+1*sin(2*x)%o%sin(2*y)%o%sin(2*z)%o%sin(w*2)
  dimnames(ten4) = list(paste0("x",1:10),paste0("y",1:20),paste0("z",1:30),paste0("w",1:19))
  back4 = as.data.frame.table(ten4)
  myten4 = df2Tensor(back4,FALSE,values_from="Freq",coord_from=c("Var1","Var2","Var3","Var4"))
  out4 = CP_TD(myten4,2,4,conv_tol=c(0,0),lambda=c(0,0),initialize = "svd")
  ts.plot(out4$sse);min(out4$sse)
  
  
  # size 5 tensor
  ten5 = sin(x)%o%sin(y)%o%sin(z)%o%cos(w)%o%sin(u) +sin(2.3*x)%o%sin(1.9*y)%o%sin(3.3*z)%o%sin(w*1.1)%o%cos(u*1.41)
  dimnames(ten5) = list(paste0("x",1:10),paste0("y",1:20),paste0("z",1:30),paste0("w",1:19),paste0("u",1:24))
  back5 = as.data.frame.table(ten5)
  myten5 = df2Tensor(back5,FALSE,values_from="Freq",coord_from=c("Var1","Var2","Var3","Var4","Var5"))
  out5 = CP_TD(myten5,2,10,lambdas = c(0,.01),conv_tol =c(0,0),initialize = "random")
  ts.plot(out5$sse);min(out5$sse)
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Sparse Tensors
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  lng2 =length(back2$Freq)
  sint = sample.int(lng2,0.5*lng2)
  spten2 = df2Tensor(back2[sint,],FALSE,values_from="Freq",coord_from=c("Var1","Var2"))
  out2 = CP_TD(spten2,5,100,lambdas = c(0,.1),initialize = "random")
  
  # size 3 tensor
  lng3 =length(back3$Freq)
  sint = sample.int(lng3,.1*lng3)
  spten3 = df2Tensor(back3[sint,],TRUE,values_from="Freq",coord_from=c("Var1","Var2","Var3"))
  out3 = CP_TD(spten3,2,100,lambdas = c(0.0,0.1),initialize="svd")
  ts.plot(out3$sse)
  ts.plot(t(out3$factors$Var3))

  # size 4 tensor
  lng4 =length(back4$Freq)
  sint = sample.int(lng4,0.05*lng4)
  spten4 = df2Tensor(back4[sint,],TRUE,values_from="Freq",coord_from=c("Var1","Var2","Var3","Var4"))
  out4 = CP_TD(spten4,2,10)
  
  
  # size 5 tensor
  lng5 =length(back5$Freq)
  sint = sample.int(lng5,0.1*lng5)
  spten5 = df2Tensor(back5[sint,],TRUE,values_from="Freq",coord_from=c("Var1","Var2","Var3","Var4","Var5"))
  out5 = CP_TD(spten5,2,10,lambdas = c(0.1,.00),conv_tol=c(0,0),initialize = "svd")
  ts.plot(t(out5$factors$Var2))
  ts.plot(out5$sse)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Non-negative tensors
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  nnten2%=%myten2
  nnten2$value=  3*abs(nnten2$value)+1
  out2 = PLSA(nnten2,2,10)
  
  nnten3%=%myten3
  nnten3$value=  3*abs(nnten3$value)+1
  out3 = PLSA(nnten3,2,10)
  
  nnten4%=%myten4
  nnten4$value = 3*abs(nnten4$value)+1
  out4 = PLSA(nnten4,2,10)
  
  nnten5%=%myten5
  nnten5$value = 3*abs(nnten5$value)+1
  out5 = PLSA(nnten5,2,10)
  
  
  
  
  # More Testing
  A = matrix(rnorm(1000),100,10)
  dimnames(A) = list(paste0("x",1:100),paste0("y",1:10))
  back2 = as.data.frame.table(A)
  tmat = df2Tensor(back2,FALSE,values_from="Freq",coord_from=c("Var1","Var2"),Reorder=FALSE)
  out2 = CP_TD(tmat,10,10,ortho_it = 4)
  ts.plot(out2$sse.it)
  
  A = round(exp(matrix(rnorm(300),100,3))%*%exp(matrix(rnorm(30),3,10)))
  dimnames(A) = list(paste0("x",1:100),paste0("y",1:10))
  back2 = as.data.frame.table(A)
  tmat = df2Tensor(back2,FALSE,values_from="Freq",coord_from=c("Var1","Var2"),reorder=TRUE)
  out2 = PLSA(tmat,3,100)
  
  # More testing 
  #devtools::load_all(".")
  n=5
  tx = ((1:n)/n)^2
  ten5 = tx%o%tx%o%tx%o%tx%o%tx
  dimnames(ten5) = list(paste0("x",1:n),paste0("y",1:n),paste0("z",1:n),paste0("w",1:n),paste0("u",1:n))
  
  back5 = as.data.frame.table(ten5)
  myten5 = df2Tensor(back5,FALSE,values_from="Freq",coord_from=c("Var1","Var2","Var3","Var4","Var5"))
  out5 = CP_TD(myten5,1,4)
  plot(out5$fitted.values,c(ten5))
  ts.plot(out5$sse.it)
  
  obj = myten5
  UpdateThese  = rep(1,obj$rank+2) # first fit, update all
  UpdateThese[obj$rank+1]=1
  nlatent=1
  number_it=c(4,0,0)
  factors = .Call('_dfTensor_Initialize',PACKAGE = 'dfTensor',obj$coord,obj$value,0,1,nlatent,obj$kdim,obj$rank) 
  cout = .Call("_dfTensor_sparse_als_cp", PACKAGE = 'dfTensor',
               factors, obj$coord, obj$value,number_it,c(0.001,0.001,0,0,0),UpdateThese,obj$rank) 
  
  plot(cout[[7]],c(obj$value),pch=".")
  ts.plot(cout[[8]])
  ts.plot(t(cout[[5]]))
  
  # Check df2Tensor
  A = array(1:4,dim=c(4,3,5))
  dimnames(A) = list(paste0("x",1:4),paste0("y",1:3),paste0("z",1:5))
  back2 = as.data.frame.table(A)
  tmat = df2Tensor(back2,FALSE,values_from="Freq",coord_from=c("Var1","Var2","Var3"),Reorder=FALSE)
  
  cbind(t(tmat$coord)+1,tmat$value)
  
  # Test qr initialization
  obj=myten3
  factors = .Call('_dfTensor_Initialize',PACKAGE = 'dfTensor',obj$coord,obj$value,0,1,2,obj$kdim,obj$rank) 
  
  
  
  ## Test
  test_walsh_like = function(nr,nc){
    out = matrix(-1,nr,nc)
    for(k in 1:nc){
      K = k-1
      jk=2^K
      for(j in 1:nr){
        J = j-1
        tog = (floor(k*J/nr)%%2)==0
        if(tog){
          out[j,k] = 1
        }
      }
    }
    return(out)
  }
  
  testme = function(nr,nc){
    out = matrix(-1,nr,nc)
    lgb = floor(log2(nr))
    for(k in 1:nc){
      K = k-1
      jk=2^(K%%lgb)
      jk1 = 2^(floor((k)/lgb))
      if(k<=lgb){
        jk1=1
      }
      
      for(j in 1:nr){
        J = j-1
        #tog = (floor(jk*J/nr)%%2)==0
        tog1 = (floor(jk1*J/nr)%%2)==0
        tog2 = (floor(jk*J/nr)%%2)==0
        if(tog1&tog2){
          out[j,k] = 1
        }
      }
    }
    
    return(out)
  };
  
  # Test other set
  tmp = read.csv("test.csv")
  obj = df2Tensor(tmp,TRUE,values_from ="scaled",coord_from =c("frag","ShipToAddressKey") )
  nlatent = 4
  init_type = 1
cout = CP_TD(ten2,4,3,initialize = "svd")
  
  

  # load("subsubset.Rdata")
  # flat_format = data.frame(ss_qty_by_zip$Zip,ss_qty_by_zip$Fragrance,ss_qty_by_zip$PLine,ss_qty_by_zip$QTY)
  # 
  # myten = df2Tensor(flat_format,TRUE)
  # nlatent = 2
  # lambdas  =rep(0.01,13)
  # number_it = c(10,-1,-1)
  # UpdateThese = rep(1,10)
  # obj = myten
  # 
  # 
  # cout = PLSA(myten)
  # dout = CP_TD(myten,wtd_it=10)
  # 
  # ###################################
  # library.dynam.unload("dfTensor", "C:/Users/chammerquist/Documents/R/R-4.1.1/library/dfTensor")
  # # know the right answer test
  # x = (1:100)*2*pi/100
  # y = (1:200)*2*pi/200
  # z = (1:300)*2*pi/300
  # ten = sin(x)%o%sin(y)%o%sin(z)+sin(2*x)%o%sin(2*y)%o%sin(3*z)
  # dimnames(ten) = list(paste0("x",1:100),paste0("y",1:200),paste0("z",1:300))
  # back = as.data.frame.table(ten)
  # myten = df2Tensor(back,FALSE)
  # 
  # nlatent = 2
  # lambdas  =rep(0.01,13)
  # number_it = c(200,-1,-1)
  # UpdateThese = rep(1,10)
  # obj = myten
  # 
  # 
  # 
  # # Get initial factors (currently random is only option)
  # factors = list()
  # for(k in 1:obj$rank){
  #   factors[[k]] = .Call('_dfTensor_randomInitialize', PACKAGE = 'dfTensor', nlatent,obj$kdim[k],2)
  # }
  # factors[[obj$rank+1]] = rep(1.0,nlatent)
  # cout = .Call("_dfTensor_sparse_als_cp_3d", PACKAGE = 'dfTensor',
  #              factors, t(obj$cood), obj$value,number_it,lambdas,UpdateThese)
  # 
  # system.time({cout = .Call("_dfTensor_sparse_als_cp_3d", PACKAGE = 'dfTensor',
  #              factors, t(obj$cood), obj$value,number_it,lambdas,UpdateThese)}) 
  # 
  # # User = 16.83,14.7,15.55 (numit = 200)
  # 
  # 
  # # make sparse
  # smp_back = back[sample.int(6e6,1e5),]
  # obj2 = df2Tensor(smp_back,TRUE)
  # number_it = c(10,-1,10)
  # #system.time({cout = .Call("_dfTensor_sparse_als_wt_cp_3d", PACKAGE = 'dfTensor',
  # #                          factors, t(obj2$cood), obj2$value,obj2$weight,number_it,lambdas,UpdateThese)}) 
  # 
  # cout = .Call("_dfTensor_sparse_als_wt_cp_3d", PACKAGE = 'dfTensor',
  #              factors, t(obj2$cood), obj2$value,obj2$weight,number_it,lambdas,UpdateThese)
  #  
  # 
  # 
  # ###################################
  #  
  # load("subset.Rdata")
  # bigdf = data.frame(qty_by_zip$Zip,qty_by_zip$Fragrance,qty_by_zip$PLine,qty_by_zip$QTY)
  # myTen = as.dfTensor(bigdf)
  # cout = PLSA(myTen)
  # 
  # tst = read.csv("TestTensor.csv")
  # tsm = log(tst[,7])-mean(log(tst[,7]))
  # tsm = 0.2*tsm
  # 
  # yr = tst[,2]
  # mnth = tst[,3]
  # md = tst[,5]
  # wk = tst[,6]
  # hd= tst[,11]
  # 
  # my_df = data.frame(md,yr,mnth,wk,hd,tsm)
  # dyten = df2Tensor(my_df,SparseAsMissing = TRUE)
  # dout = dfTensor::CP_TD(dyten,4,500,.01,0,wtd_it=100)
  # 
  # ny = reconstruct(dout,dyten,return_as_df = FALSE)
  # df_out = reconstruct(dout,dyten)
  # 
  # 1-var(ny-dyten$value)/var(dyten$value)
  # plot(ny,dyten$value)
  # 
  # ind = order(df_out$yr,df_out$mnth,df_out$md)
  # ts.plot(tsm)
  # lines(df_out$value[ind],col="red")
  # 
  # err = tsm-df_out$value[ind]
  # 
  # dout = PLSA(dyten,3,100)
  # 
  # ########################################################
  # # Iterative adjustment for missing weights
  # ########################################################
  # 
  # adjX = function(A,d,P,numit=10,lambda=.1){
  #   D = d
  #   nz = dim(A)[2]
  #   IL = diag(x=lambda,nrow=nz)
  #   LHS = t(A)%*%P%*%A + IL
  #   RHS = t(A)%*%P%*%d
  #   
  #   
  #   xR = solve(LHS,RHS)
  #   x = xR*0
  #   err = matrix(0,numit,1)
  #   L=1
  #   dx=0
  #   RHS0=0
  #   for(k in 1:numit){
  #     RHS = t(A)%*%d+IL%*%x
  #     LHS = t(A)%*%A+IL
  #     dx = solve(LHS,RHS)
  #     if(k==1){
  #       x =   dx 
  #     }else{
  #       x = x-dx
  #     }
  # 
  #     d = P%*%(A%*%x-D)
  #     err[k,1] = sum(abs(x-xR))
  #     #err[k,2] = L #sum((P%*%(A%*%x-D))^2)
  #     
  #   }
  #   return(err)
  #   
  # }
  # 
  # 
  # 
  # out = adjX(Ai,di,Pi,10,1);ts.plot(out)
  # 
  # 
  # 
  # Ai = matrix(rnorm(50000),5000,10)
  # xi = rexp(10,300)+10
  # di = Ai%*%xi+rnorm(5000)*0.01
  # ph = rep(1,5000) #
  # ph[sample.int(5000,4500)]=0
  # Pi= diag(ph)
  # out = adjX(Ai,di,Pi,100,10);ts.plot(out)
}
