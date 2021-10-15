import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


class HomographyMatrixFinder:
    def __init__(self):
        self.dirName = "PictureGroup"
        self.imgNames = []
        self.imgFiles = []
        self.sticher = cv2.Stitcher.create()
        self.panoLeft = None
        self.panoRight = None
        self.pano = None


    def initValues(self):
        for file in glob.glob( self.dirName+"/*.jpg"):
            self.imgNames.append(file)
        self.imgNames.sort()
        for i in self.imgNames:
            self.imgFiles.append(cv2.imread(i))

        print("Init completed!")

    def stitchImgs(self):
        # ret, self.panoLeft = self.sticher.stitch(self.imgFiles[0:3])
        # if ret==cv2.STITCHER_OK:
        #     cv2.imshow("result1",self.panoLeft)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        ret, self.panoRight = self.sticher.stitch( self.imgFiles[2:])
        if ret==cv2.STITCHER_OK:
            cv2.imshow("result1",self.panoRight)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(ret)
        # if ret==cv2.STITCHER_OK:
        #     cv2.imshow("result2",self.panoRight)
        #
        # ret, self.pano = self.sticher.stitch([self.panoLeft,self.panoRight])
        # if ret==cv2.STITCHER_OK:
        #     cv2.imshow("result",self.pano)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     print("Stitching not Successed!")





HM =  HomographyMatrixFinder()
HM.initValues()
HM.stitchImgs()

