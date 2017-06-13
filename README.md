# Projection Transformations
Transformations allow us to describe an image with a new set of coordinates.  
This repo demonstrates the ability to transform quadrilaterals into rectangles.

## Running the script
###### Dependencies
  * [NumPy](http://www.numpy.org/)
  * [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

###### Basic usage
`python run.py -s [path to source image] -o [path to output image] -x [width of output image] -y [height of output image]`

###### Example usage
`python run.py -s test_images/test1/source.jpg -o test_images/test1/out.png -x 500 -y 350`

## Results
###### Test 1 source
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test1/source.jpg "Test 1 source")

###### Test 1 selected corners
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test1/out_corners.png "Test 1 corners")

###### Test 1 result with x = 500, y = 350
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test1/out.png "Test 1 result")

###### Test 6 source
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test6/source.jpg "Test 6 source")

###### Test 6 selected corners
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test6/out_corners.png "Test 6 corners")

###### Test 6 result with x = 600, y = 350
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test6/out.png "Test 6 result")

###### Test 7 source
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test7/source.jpg "Test 7 source")

###### Test 7 selected corners
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test7/out_corners.png "Test 7 corners")

###### Test 7 result with x = 700, y = 350
![alt text](https://github.com/g3aishih/projection-transformations/blob/master/test_images/test7/out.png "Test 7 result")
