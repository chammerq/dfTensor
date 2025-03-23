\name{CP_TD}
\alias{CP_TD}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Canonical Polyadic Tensor Decomposition of a dfTensor}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
CP_TD(obj, nlatent = 2, als_it = 2, lambdas = c(1e-06, 0.001), conv_tol = c(1e-06, 1e-06), initialize = c("random", "svd"), ortho_it = -1, wtd_it = -1, warm_start = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{A dfTensor object}
  \item{nlatent}{size of the latent space to decompose too...}
  \item{als_it}{Number of ALS iterations}
  \item{lambdas}{a 2-vector of lambdas use for regularization. First componet is overall regularization for ALS least squares, second is stepwise,Levenberg Marquardt style.}
  \item{conv_tol}{convergence tolerance (abs_tol,relative_tol)}
  \item{initialize}{Initialization type, svd does sketching on tensor and then QR decomposition, random is random.}
  \item{ortho_it}{Orthogonalize factors for this many iterations}
  \item{wtd_it}{not current used}
  \item{warm_start}{not currently used}
}
\details{
weights are used as is, not squared, i.e. cost(x)  = sum(w*(x-y)^2)
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}

\seealso{ \code{\link{PLSA}},\code{\link{df2Tensor}}}
