package nndl

object MainApp extends App {

  val trainImagesPath = "mnist/train-images-idx3-ubyte.gz"
  val trainLabelsPath = "mnist/train-labels-idx1-ubyte.gz"
  val testImagesPath = "mnist/t10k-images-idx3-ubyte.gz"
  val testLabelsPath = "mnist/t10k-labels-idx1-ubyte.gz"

  val trainDataset = DataReader.readDataset(trainImagesPath, trainLabelsPath)
  val testDataset = DataReader.readDataset(testImagesPath, testLabelsPath)

  val network = new Network(Array(784, 100, 10), CrossEntropyCost)
  network.stochasticGradientDescent(trainDataset, testDataset, 30, 10, 0.5, 5.0)

}
