# Shihan Ai
# Github: g3aishih

import numpy as np
import cv2 as cv
import argparse, os, sys

class Transformation:
    global mouseX, mouseY
    def __init__(self, source, width, height):
        self.source = source
        self.sourceCopy = self.source.copy()
        self.outX = width
        self.outY = height
        self.maxX = source.shape[1]
        self.maxY = source.shape[0]

    def run(self):
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.handle_click)
        edges = []
        while True:
            cv.imshow('image', self.sourceCopy)
            if (len(edges) == 4):
                edges = self.sortEdges(edges)
                H = self.calculateHomography(edges)
                H_inv = np.linalg.inv(H)
                return self.inverseWarp(H_inv)

            key = cv.waitKey()
            if key == 27:
                break
            elif key == ord(' '):
                if (len(edges) < 4):
                    cv.circle(self.sourceCopy, (mouseX, mouseY), 3, (0, 0, 255), -1)
                    edges.append((mouseX, mouseY))
        return 0

    def handle_click(self, event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv.EVENT_LBUTTONUP:
            mouseX, mouseY = x, y

    def inverseWarp(self, H):
        image = np.zeros_like(self.source, dtype=np.uint8)
        rowVec = np.arange(1, self.outY + 1)[:, None]
        colVec = np.arange(1, self.outX + 1)[None, :]
        rowIndices = np.dot(rowVec, np.ones((1, colVec.size), dtype=np.float32))
        colIndices = np.dot(np.ones((rowVec.size, 1), dtype=np.float32), colVec)

        r, c = self.getCoord(H, rowIndices, colIndices)
        image = self.bilinear_interpolate(r-1, c-1)

        return image

    def getCoord(self, H, y, x):
        _y = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
        _x = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
        return _y, _x

    def bilinear_interpolate(self, y, x):
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

    def calculateHomography(self, edges):
        x1 = edges[0][0]
        x2 = edges[1][0]
        x3 = edges[2][0]
        x4 = edges[3][0]

        y1 = edges[0][1]
        y2 = edges[1][1]
        y3 = edges[2][1]
        y4 = edges[3][1]

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

    def sortEdges(self, edges):
        sortX = sorted(edges, key=lambda x: x[0])
        sortY1 = sorted(sortX[:2], key=lambda y: y[1])
        sortY2 = sorted(sortX[2:], key=lambda y: y[1], reverse=True)
        result = sortY1 + sortY2
        return sortY1 + sortY2

def debug(image):
    # Display the image
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def readSource(fileName):
    try:
        source = cv.imread(fileName, 1)
    except:
        print("[ERROR] Source must be a color uint8 image")
        return None
    return source

def writeImage(fileName, image):
    try:
        cv.imwrite(fileName, image)
        success = True
    except:
        success = False
    return success

def main(args):
    source = readSource(args.s)
    assert source is not None

    transformation = Transformation(source, args.x, args.y)
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
