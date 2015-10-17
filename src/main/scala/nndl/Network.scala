package nndl

import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Gaussian

class Network(sizes: Array[Int], cost: Cost) {

  val numLayers = sizes.length

  val dist = new Gaussian(0, 1)

  // NOTE input neurons have no biases
  val biases: Array[DVec] = for (s <- sizes.drop(1)) yield DenseVector.rand(s, dist)

  val weights: Array[DMat] =
    for ((in, out) <- sizes zip sizes.drop(1)) yield DenseMatrix.rand(out, in, dist) / sqrt(in)

  def feedForward(input: DVec): DVec = {
    var a = input
    for (i <- 0 until numLayers - 1) {
      val b = biases(i)
      val w = weights(i)
      a = sigmoid(w * a + b)
    }
    a
  }

  def stochasticGradientDescent(
      trainData: Seq[(DVec, DVec)],
      testData: Seq[(DVec, DVec)],
      epochs: Int,
      miniBatchSize: Int,
      eta: Double,
      lambda: Double
  ): Unit = {
    for (j <- 0 until epochs) {
      val miniBatches = Random.shuffle(trainData).grouped(miniBatchSize)
      for (miniBatch <- miniBatches) updateMiniBatch(miniBatch, eta, lambda, trainData.length)
      println(s"Epoch $j training complete")
      println(s"Cost on training: ${totalCost(trainData, lambda)}")
      println(s"Cost on testing: ${totalCost(testData, lambda)}")
      println(s"Accuracy on training: ${accuracy(trainData)} / ${trainData.length}")
      println(s"Accuracy on testing: ${accuracy(testData)} / ${testData.length}")
      println
    }
  }

  def updateMiniBatch(
      miniBatch: Seq[(DVec, DVec)],
      eta: Double,
      lambda: Double,
      n: Int
  ): Unit = {
    val gradBiases = zeroBiases()
    val gradWeights = zeroWeights()
    val size = miniBatch.length
    for ((x, y) <- miniBatch) {
      val (deltaGradBiases, deltaGradWeights) = backPropagation(x, y)
      for (i <- 0 until numLayers - 1) {
        gradBiases(i) += deltaGradBiases(i)
        gradWeights(i) += deltaGradWeights(i)
      }
    }
    for (i <- 0 until numLayers - 1) {
      weights(i) :*= (1 - (eta * (lambda / n)))
      weights(i) -= gradWeights(i) :* (eta / size)
      biases(i) -= gradBiases(i) :* (eta / size)
    }
  }

  def backPropagation(x: DVec, y: DVec): (Array[DVec], Array[DMat]) = {
    val gradBiases = zeroBiases()
    val gradWeights = zeroWeights()
    // forward
    var a = x
    val as = Array.ofDim[DVec](numLayers) // activations per layer
    as(0) = a
    val zs = Array.ofDim[DVec](numLayers - 1) // z vectors per layer
    for (i <- 0 until numLayers - 1) {
      val b = biases(i)
      val w = weights(i)
      val z = w * a + b
      zs(i) = z
      a = sigmoid(z)
      as(i + 1) = a
    }
    // backwards
    var delta = cost.delta(zs.last, as.last, y)
    val sz = numLayers - 1 // used to fake negative indices
    gradBiases(sz - 1) = delta
    gradWeights(sz - 1) = delta * as(as.length - 2).t
    for (l <- 2 until numLayers) {
      val z = zs(sz - l)
      val sp = sigmoidPrime(z)
      delta = (weights(sz - l + 1).t * delta) :* sp
      gradBiases(sz - l) = delta
      gradWeights(sz - l) = delta * as(as.length - l - 1).t
    }
    (gradBiases, gradWeights)
  }

  def accuracy(data: Seq[(DVec, DVec)]): Int = {
    val results = for ((x, y) <- data) yield (argmax(feedForward(x)), argmax(y))
    val correct = results.filter(r => r._1 == r._2)
    correct.length
  }

  def totalCost(data: Seq[(DVec, DVec)], lambda: Double): Double = {
    var c = 0.0
    val n = data.length
    for ((x, y) <- data) {
      val a = feedForward(x)
      c += cost(a, y) / n
    }
    c += 0.5 * (lambda / n) * weights.map(w => pow(norm(w.flatten()), 2)).sum
    c
  }

  def zeroBiases(): Array[DVec] =
    sizes.drop(1).map(DenseVector.zeros[Double](_))

  def zeroWeights(): Array[DMat] =
    sizes.drop(1).zip(sizes).map((DenseMatrix.zeros[Double] _).tupled)

}
