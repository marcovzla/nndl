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

  def readImagesDVec(path: String): Array[DVec] =
    readImages(path) map (im => new DenseVector(im.flatten.map(_ / 255.0)))

  def readImages(path: String): Array[Array[Array[Int]]] = {
    val bb = readBytes(path)
    require(bb.getInt() == 2051, "wrong magic number")
    val numImages = bb.getInt()
    val numRows = bb.getInt()
    val numCols = bb.getInt()
    // data stored as unsigned bytes but scala bytes are signed
    val images = Array.fill[Int](numImages, numRows, numCols)(bb.get() & 0xFF)
    images
  }

  def readLabels(path: String): Array[Int] = {
    val bb = readBytes(path)
    require(bb.getInt() == 2049, "wrong magic number")
    val numLabels = bb.getInt()
    val labels = Array.fill[Int](numLabels)(bb.get())
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
