\name{df2Tensor}
\alias{df2Tensor}
\title{Data.Frame to Tensor.}
\description{Convert data.frames to a sparse format tensor of class dfTensor.}
\usage{
df2Tensor(df, SparseAsMissing, values_from = NULL, coord_from = NULL, weights_from = NULL, checkUnique = FALSE}

\arguments{
  \item{df}{data.frame to be converted to dftensor}
  \item{SparseAsMissing}{boolean: Should the indices included in the data frame be considered missing or zero.}
  \item{values_from}{Character: Name of column to be the values in the tensor}
  \item{coord_from}{Character: Names of columns that will be the indices of the tensor}
  \item{weights_from}{Character: Names of columns that will be weights of the tensor,optional}
  \item{checkUnique}{boolean, throw an error if coordiantes are not unique}

}
\details{This function sorts the columns by largest dimesion, then by sort the indices for faster computation in methods.}
\value{An object of class dfTensor}


\seealso{ \code{\link{CP_TD}},\code{\link{PLSA}}}
\examples{}

