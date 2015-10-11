package nndl

import java.io._
import java.nio.ByteBuffer
import java.util.zip.GZIPInputStream
import breeze.linalg._
import org.apache.commons.io.IOUtils

// http://yann.lecun.com/exdb/mnist/

object DataReader {

  def readDataset(imagesPath: String, labelsPath: String): Seq[(DVec, DVec)] =
    readImagesDVec(imagesPath) zip readLabelsDVec(labelsPath)

  def readLabelsDVec(path: String): Array[DVec] =
    readLabels(path) map vectorizeDigit

  def readImagesDVec(path: String): Array[DVec] = {
    val images = readImages(path)
    val numImages = images.length
    val numRows = images(0).length
    val numCols = images(0)(0).length
    // pixels in the dataset have values from 0 to 255
    // but we need them to be between 0 and 1 for our neural network
    images.map(im => new DenseVector(im.flatten.map(_.toDouble)) / 255.0)
  }

  def readImages(path: String): Array[Array[Array[Int]]] = {
    val bb = readBytes(path)
    require(bb.getInt() == 2051, "wrong magic number")
    val numImages = bb.getInt()
    val numRows = bb.getInt()
    val numCols = bb.getInt()
    val images = Array.ofDim[Int](numImages, numRows, numCols)
    for {
      i <- 0 until numImages
      r <- 0 until numRows
      c <- 0 until numCols
    } {
      // data stored as unsigned bytes
      // but scala bytes are signed
      images(i)(r)(c) = bb.get() & 0xFF
    }
    images
  }

  def readLabels(path: String): Array[Int] = {
    val bb = readBytes(path)
    require(bb.getInt() == 2049, "wrong magic number")
    val numLabels = bb.getInt()
    val labels = Array.ofDim[Int](numLabels)
    for (i <- 0 until numLabels) {
      labels(i) = bb.get()
    }
    labels
  }

  def readBytes(path: String): ByteBuffer = {
    val file = new File(path)
    val in = new GZIPInputStream(new FileInputStream(file))
    val array = IOUtils.toByteArray(in)
    in.close()
    ByteBuffer.wrap(array)
  }

}
