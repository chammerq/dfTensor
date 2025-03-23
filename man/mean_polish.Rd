\name{mean_polish}
\alias{mean_polish}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Mean Polish on a dfTensor object.
}
\description{
This does the mean polish on a dfTensor( like median polish but with means).
}
\usage{
mean_polish(obj, numit, conv_tol = c(1e-06, 1e-06), stepsize = 1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{dfTensor object}
  \item{numit}{Number of iterations}
  \item{conv_tol}{Convergence tolerance (absolute, relative)}
  \item{stepsize}{stepsize of fitting algorithm, might need to be set to less then 0.5 to allow convergence}
}
\details{
Iteratively finds the fiber means of each tensor mode.
}
\value{
A list with means, convergence info
}

\seealso{
\code{\link{poisson_rates},\link{PLSA}}
}
