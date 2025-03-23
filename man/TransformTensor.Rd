\name{transform_tensor}
\alias{transform_tensor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Apply a function to values in a dfTensor}
\description{
Apply functions to values in a dfTensor.
}
\usage{
transform_tensor(obj,fun=I())
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{dfTensor object }
  \item{fun}{Function to apply to values.}
}
\details{
A pretty simple function. Equivalent to obj$values = fun(obj$values). If the
tensor is sparse (i.e. missing values are zero), then function must map zeros to zero. Changes the values in place.
}
\value{
Invisibly returns a dftensor object.
}


\seealso{
\code{\link{df2Tensor}}, ~~~
}
\examples{
logtensor = transform_tensor(mytensor,log1p) 
}
