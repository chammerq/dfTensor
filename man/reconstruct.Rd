\name{reconstruct}
\alias{reconstruct}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Reconstruct a tensor from factors.
}
\description{
Given a factored tensor object, reconstruct the tensor at indices requested.
}
\usage{
reconstruct(obj, tensor, newdf = NULL, Reorder = TRUE, return_as_df = TRUE)
}
\arguments{
  \item{obj}{A factorization object from \code{\link{PLSA}}, \code{\link{CP_TD}},\code{\link{mean_polish}}}
  \item{tensor}{The original tensor}
  \item{newdf}{The new data frame or dftensor to find the values at.}
  \item{Reorder}{Boolean: Should the columns be reordered to match original order}
  \item{return_as_df}{Boolean: should the result be returned as a data.frame}
}
\details{
This function is unfinished, so use with care.
}
\value{
A vector or data.frame
}


\seealso{
\code{\link{PLSA}}, \code{\link{CP_TD}},\code{\link{mean_polish}}
}
\examples{

}

