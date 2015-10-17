import breeze.linalg._
import breeze.numerics._

package object nndl {

  type DVec = DenseVector[Double]
  type DMat = DenseMatrix[Double]

  /** sigmoid derivative */
  def sigmoidPrime(z: DVec): DVec = sigmoid(z) :* (1.0 - sigmoid(z))

  /** gets a digit and returns a one-hot vector */
  def vectorizeDigit(d: Int): DVec = {
    require(d >= 0 && d <= 9, "need a number between 0 and 9")
    val vec = DenseVector.zeros[Double](10)
    vec(d) = 1
    vec
  }

  /** replaces nan with zero and infinity with finite numbers */
  def nanToNum(v: DVec): DVec = v map {
    case Double.NegativeInfinity => Double.MinValue
    case Double.PositiveInfinity => Double.MaxValue
    case x if x.isNaN => 0.0
    case x => x
  }

}
