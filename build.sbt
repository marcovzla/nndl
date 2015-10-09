name := "nndl"

version := "1.0-SNAPSHOT"

scalaVersion := "2.11.6"

scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation")

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "commons-io" % "commons-io" % "2.4"
)
