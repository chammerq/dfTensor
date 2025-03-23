\name{poisson_rates}
\alias{poisson_rates}

\title{
poisson rates
}
\description{
like mean_polish but with poisson maximum likliehood.
}
\usage{
poisson_rates = (obj,numit,conv_tol=1e-6,threshold=1e-12){
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dims}{dfTensor objects}
  \item{numit}{number of iterations}
  \item{conv_tol}{convergence tolerance of fitting algorithm}
  \item{threshold}{}


\details{
lightly tested, so use with care
}
\value{
A list with poisson rates,means, convergence info
}

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
