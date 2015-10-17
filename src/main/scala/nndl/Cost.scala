package nndl

import breeze.linalg._
import breeze.numerics._

trait Cost {
  def apply(a: DVec, y: DVec): Double
  def delta(z: DVec, a: DVec, y: DVec): DVec
}

object QuadraticCost extends Cost {
  def apply(a: DVec, y: DVec): Double = 0.5 * pow(norm(a - y), 2.0)
  def delta(z: DVec, a: DVec, y: DVec): DVec = (a - y) :* sigmoidPrime(z)
}

object CrossEntropyCost extends Cost {
  def apply(a: DVec, y: DVec): Double = sum(nanToNum((-y :* log(a)) - ((1.0 - y) :* log(1.0 - a))))
  def delta(z: DVec, a: DVec, y: DVec): DVec = a - y
}
