import breeze.linalg._
import breeze.numerics._

package object nndl {

  type DVec = DenseVector[Double]
  type DMat = DenseMatrix[Double]

  /** sigmoid derivative */
  def sigmoidPrime(z: DVec): DVec = sigmoid(z) :* (-sigmoid(z) + 1.0)

  /** gets a digit and returns a one-hot vector */
  def vectorizeDigit(d: Int): DVec = {
    require(d >= 0 && d <= 9, "need a number between 0 and 9")
    val vec = DenseVector.zeros[Double](10)
    vec(d) = 1
    vec
  }

  /** replaces nan with zero and infinity with finite numbers */
  def nanToNum(v: DVec): DVec = v map { e =>
    if (e.isNaN) 0.0
    else if (e == Double.NegativeInfinity) Double.MinValue
    else if (e == Double.PositiveInfinity) Double.MaxValue
    else e
  }

}
