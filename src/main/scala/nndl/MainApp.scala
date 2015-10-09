package nndl

object MainApp extends App {

  val trainImagesPath = "data/train-images-idx3-ubyte.gz"
  val trainLabelsPath = "data/train-labels-idx1-ubyte.gz"
  val testImagesPath = "data/t10k-images-idx3-ubyte.gz"
  val testLabelsPath = "data/t10k-labels-idx1-ubyte.gz"

  val trainDataset = DataReader.readDataset(trainImagesPath, trainLabelsPath)
  val testDataset = DataReader.readDataset(testImagesPath, testLabelsPath)

  val network = new Network(Array(784, 100, 10), CrossEntropyCost)
  network.stochasticGradientDescent(trainDataset, testDataset, 30, 10, 0.5, 5.0)

}
