\name{tensor2df}
\alias{tensor2df}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Convert a tensor back to data.frame.
}
\description{
A function to extract a data.frame from a dftensor object.
}
\usage{tensor2df(obj)}

\arguments{
  \item{obj}{dftensor object}
}
\details{
The resulting data.frame will have rows in different order than the original 
data.frame due the sorting done by \code{\link{df2Tensor}}.
}
\value{
A data frame.
}

\seealso{
\code{\link{df2Tensor}}
}
\examples{

mydf2 = tensor2df(myten)
}
