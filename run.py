# Shihan Ai
# Github: g3aishih

import numpy as np
import cv2 as cv
import argparse, os, sys
from util import *

class Transformation:
    global mouseX, mouseY
    def __init__(self, source, width, height, outputPath):
        self.source = source
        self.sourceCopy = self.source.copy()
        self.outX = width
        self.outY = height
        self.maxX = source.shape[1]
        self.maxY = source.shape[0]
        splitString = outputPath.split('.')
        self.outputPath = str(splitString[0]) + '_corners.'
        for i in range(1, len(splitString)):
            self.outputPath += str(splitString[i])

    def run(self):
        # Open a window for the user to outline four corners
        # Use mouse to left click then press Spacebar to confirm
        # A red dot should appear at the mouse cursor as confirmation
        # Press ESC to quit
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.handle_click)
        corners = []
        while True:
            cv.imshow('image', self.sourceCopy)
            if (len(corners) == 4):
                # Write the four selected points to a file
                writeImage(self.outputPath, self.sourceCopy)

                # Once we have four corners we sort the them to guarantee
                # the following order: [Top_left, Bottom_left, Bottom_right, Top_right]
                corners = self.sortCorners(corners)

                # Calculate the homography matrix
                H = self.calculateHomography(corners)

                # We need to perform inverse warping so we need the inverse
                # of the homography matrix
                H_inv = np.linalg.inv(H)
                return self.inverseWarp(H_inv)

            key = cv.waitKey()
            if key == 27:
                break
            elif key == ord(' '):
                if (len(corners) < 4):
                    cv.circle(self.sourceCopy, (mouseX, mouseY), 3, (0, 0, 255), -1)
                    corners.append((mouseX, mouseY))
        return 0

    def handle_click(self, event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv.EVENT_LBUTTONUP:
            mouseX, mouseY = x, y

    def inverseWarp(self, H):
        # Generate index vectors for efficient computation
        image = np.zeros_like(self.source, dtype=np.uint8)
        rowVec = np.arange(1, self.outY + 1)[:, None]
        colVec = np.arange(1, self.outX + 1)[None, :]
        rowIndices = np.dot(rowVec, np.ones((1, colVec.size), dtype=np.float32))
        colIndices = np.dot(np.ones((rowVec.size, 1), dtype=np.float32), colVec)

        # Transform each pixel
        r, c = self.getCoord(H, rowIndices, colIndices)

        # Use bilinear interpolation to get the color values from the source image
        image = self.bilinear_interpolate(r-1, c-1)

        return image

    def getCoord(self, H, y, x):
        _y = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
        _x = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
        return _y, _x

    def bilinear_interpolate(self, y, x):
        # Method adapted from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.maxX - 1);
        x1 = np.clip(x1, 0, self.maxX - 1);
        y0 = np.clip(y0, 0, self.maxY - 1);
        y1 = np.clip(y1, 0, self.maxY - 1);

        Ia = self.source[y0, x0]
        Ib = self.source[y1, x0]
        Ic = self.source[y0, x1]
        Id = self.source[y1, x1]

        wa = ((x1-x) * (y1-y))[:,:,None]
        wb = ((x1-x) * (y-y0))[:,:,None]
        wc = ((x-x0) * (y1-y))[:,:,None]
        wd = ((x-x0) * (y-y0))[:,:,None]

        image = (wa*Ia + wb*Ib + wc*Ic + wd*Id).astype(np.uint8)
        return image

    def calculateHomography(self, corners):
        # Sets up the system of linear equations and solves for the transformation coefficients
        x1 = corners[0][0]
        x2 = corners[1][0]
        x3 = corners[2][0]
        x4 = corners[3][0]

        y1 = corners[0][1]
        y2 = corners[1][1]
        y3 = corners[2][1]
        y4 = corners[3][1]

        _x1 = 1
        _x2 = 1
        _x3 = self.outX
        _x4 = self.outX

        _y1 = 1
        _y2 = self.outY
        _y3 = self.outY
        _y4 = 1

        A = np.array([
            [x1, y1, 1, 0, 0, 0, -x1*_x1, -y1*_x1],
            [0, 0, 0, x1, y1, 1, -x1*_y1, -y1*_y1],
            [x2, y2, 1, 0, 0, 0, -x2*_x2, -y2*_x2],
            [0, 0, 0, x2, y2, 1, -x2*_y2, -y2*_y2],
            [x3, y3, 1, 0, 0, 0, -x3*_x3, -y3*_x3],
            [0, 0, 0, x3, y3, 1, -x3*_y3, -y3*_y3],
            [x4, y4, 1, 0, 0, 0, -x4*_x4, -y4*_x4],
            [0, 0, 0, x4, y4, 1, -x4*_y4, -y4*_y4]])
        B = np.array([[_x1],[_y1],[_x2],[_y2],[_x3],[_y3],[_x4],[_y4]])
        X = np.linalg.solve(A,B)
        X = np.append(X, [1])
        return X.reshape(3,3)

    def sortCorners(self, corners):
        # Takes in four corners as a list of tuples of the form (x, y) and
        # sorts them in the following order: [Top_left, Bottom_left, Bottom_right, Top_right]
        sortX = sorted(corners, key=lambda x: x[0])
        sortY1 = sorted(sortX[:2], key=lambda y: y[1])
        sortY2 = sorted(sortX[2:], key=lambda y: y[1], reverse=True)
        result = sortY1 + sortY2
        return sortY1 + sortY2

def main(args):
    source = readSource(args.s)
    assert source is not None

    transformation = Transformation(source, args.x, args.y, args.o)
    result = transformation.run()
    debug(result)
    writeImage(args.o, result)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        type=str,
                        help='Path to source image',
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        help='Path to output image',
                        required=True)
    parser.add_argument('-x',
                        type=int,
                        help='Width of output image',
                        required=True)
    parser.add_argument('-y',
                        type=int,
                        help='Height of output image',
                        required=True)
    args = parser.parse_args()

    t1 = t2 = 0
    t1 = cv.getTickCount()
    main(args)
    t2 = cv.getTickCount()
    print('Completed in %g seconds'%((t2-t1)/cv.getTickFrequency()))
