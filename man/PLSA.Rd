\name{PLSA}
\alias{PLSA}
\title{PLSA decomposition of a sparse dfTensor}
\description{PLSA decomposition of a sparse dfTensor.}
\usage{
PLSA(obj, nlatent = 2, niter = 2, tolerances = c(1e-05, 1e-05, 1e-06), initialize = c("random", "svd"), warm_start = NULL)
}

\arguments{
  \item{obj}{object of class dfTensor}
  \item{nlatent}{number of latent dimensions}
  \item{niter}{number of iterations}
  \item{tolerances}{3-vector of tolerances, c(absolute_tolerance,relative_tolerance,cutoff )}
  \item{initialize}{How to initialize.}
  \item{warm_start}{Warm start with matrices, not currently an option}
}
\details{
The cutoff toleranance is scaled by the dimension of the factors matrices and is the mininum value that is allowed
in the matrices before rescaling. Essentially it prevents zeros.
}
\value{A list with factors, dimension names and misc}

\seealso{ \code{\link{CP_TD}},\code{\link{df2Tensor}}}
\examples{}


