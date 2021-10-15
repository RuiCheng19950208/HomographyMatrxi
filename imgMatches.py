import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

class imgMatcher:
    def __init__(self):
        self.dirName = "PictureGroup"
        self.imgNames = []
        self.imgFiles = []
        self.imgNamesProject = []
        self.imgFilesProject = []

    def init_values(self):
        for file in glob.glob( self.dirName+"/*.jpg"):
            self.imgNames.append(file)
        for file in glob.glob( self.dirName+"/project0*.tif"):
            self.imgNamesProject.append(file)
        self.imgNames.sort()
        self.imgNamesProject.sort()
        for i in self.imgNames:
            # self.imgFiles.append(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY) )
            self.imgFiles.append(cv2.imread(i))
        for i in self.imgNamesProject:
            # self.imgFiles.append(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY) )
            self.imgFilesProject.append(cv2.imread(i))
        self.refImg =  cv2.imread(self.dirName+"/project.tif")

    def get_matches(self):
        orb = cv2.ORB_create(50)
        self.kp=[]
        self.des=[]
        for i in range(len(self.imgFiles[:2])):
            self.kp.append(orb.detectAndCompute(self.imgFiles[i],None)[0])
            self.des.append(orb.detectAndCompute(self.imgFiles[i],None)[1])
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        self.matches = matcher.match(self.des[0],self.des[1],None)
        self.matches = sorted(self.matches,key=lambda x:x.distance)[:15]
        result = cv2.drawMatches(self.imgFiles[0],self.kp[0],self.imgFiles[1],self.kp[1],self.matches,None)
        print(len(self.matches))
        # cv2.imshow("result",result)
        # cv2.waitKey(0)

    def find_homography(self):
        self.points=np.zeros((2,len(self.matches),2),dtype=np.float32) # 2 points list for 2 images, len(matches) points per image
        for index, match in enumerate(self.matches):
            self.points[0,index,:] = self.kp[0][match.queryIdx].pt
            self.points[1,index,:] = self.kp[1][match.trainIdx].pt






        # self.HM,self.mask = cv2.findHomography(self.points[0],self.points[1],cv2.RANSAC)
        self.HM, self.mask = cv2.findHomography(self.points[0], self.points[1], cv2.RANSAC)
        height,width,channels = self.imgFiles[3].shape
        # print(height,width,channels)
        self.imgReshape = cv2.warpPerspective(self.imgFiles[1],self.HM,(width,height))
        cv2.imshow("result",self.imgReshape)
        cv2.waitKey(0)

    def find_HM(self):
        # ref_point0,target_point0 = [[53,229],[195,208],[256,307],[115,307]],[[539,265],[666,233],[743,322],[594,329]]

        # ref_point0, target_point0 =[[37,274],[652,253],[802,213],[860,321],[337,396],[30,521]],[[51,261],[540,264],[666,224],[744,323],[303,385],[51,486]]
        # ref_point1,target_point1 = [[652,253],[802,213],[1220,97],[1431,262],[1191,305],[860,321]],[[52,230],[195,209],[544,125],[763,281],[531,308],[256,308]]
        # ref_point2,target_point2 = [[1341,106],[1666,76],[2003,125],[1853,263],[1760,272],[1431,262]],[[108,90],[415,98],[767,137],[605,271],[516,276],[210,253]]
        # ref_point3,target_point3 = [[1341,106],[1666,76],[2003,125],[1853,263],[1760,272],[1431,262]],[[109,87],[406,79],[773,97],[605,242],[515,251],[214,244]]
        # ref_point4,target_point4 = [[1841,139],[2004,126],[2404,95],[2428,330],[2153,317],[1854,264]],[[72,121],[254,122],[654,90],[683,311],[404,304],[92,248]]
        # ref_point5, target_point5 = [[2404,95],[2591,16],[2994,126],[2984,404],[2428,330],[2337,303]],[[113,97],[307,27],[730,119],[717,385],[146,331],[30,305]]
        #
        # ref_point0 +=[[12,499],[319,399],[517,471],[543,486],[192,565],[34,597]]
        # target_point0 +=[[29,464],[304,385],[469,447],[491,460],[215,530],[57,557]]
        # ref_point1 +=[[703,330],[816,323],[1120,310],[765,360],[688,384],[698,445]]
        # target_point1 +=[[115,308],[220,308],[471,308],[176,338],[108,357],[125,413]]
        # ref_point2 +=[[1423,262],[1483,285],[1737,273],[1765,344],[1531,313],[1369,304]]
        # target_point2 +=[[201,253],[262,278],[495,275],[514,339],[308,303],[153,288]]
        #
        # ref_point3 +=[[1423,262],[1603,195],[1737,273],[1765,344],[1531,313],[1369,304]]
        # target_point3 +=[[205,244],[365,187],[493,253],[523,316],[300,289],[151,282]]
        # ref_point4 +=[[1808,194],[1947,193],[2009,194],[2142,251],[2100,292],[2204,421]]
        # target_point4 +=[[32,176],[196,184],[260,188],[392,242],[353,280],[455,398]]
        # ref_point5 +=[[2467,258],[2811,269],[2910,285],[2917,539],[2856,538],[2736,529]]
        # target_point5 +=[[186,255],[533,257],[584,265],[650,513],[588,509],[464,501]]

        # ---------------------------------------------

        # ref_point0 =[[78.522531,112.037616],[76.424055,203.162411],[76.123786,294.839263],[77.682973,385.344286],[164.338344,102.752568],[162.282664,198.234444],[161.635891,294.527714],[162.463546,389.531989],[258.547936,96.88694],[256.799578,195.312155],[255.827476,294.767062],[255.686948,392.859078],[357.568828,95.693787],[356.333804,195.119706],[355.080589,295.676351],[353.840202,394.860607]]
        # target_point0 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]
        # ref_point1 =[[525.342518,140.338259],[526.739358,230.282098],[529.961749,321.339397],[535.023657,411.794879],[610.031618,127.559562],[611.93205,221.897446],[615.284373,317.591148],[620.080524,412.551109],[703.230535,117.446798],[705.843893,214.840753],[709.291993,313.759418],[713.530271,411.820835],[801.512976,111.40418],[804.926517,209.98479],[808.389355,310.111124],[811.815032,409.287448]]
        # target_point1 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]
        # ref_point2 =[[1045.035529,151.356285],[1043.299902,240.880471],[1043.344688,311.82599],[1045.251145,422.486745],[1129.810515,141.726863],[1128.56527,235.488811],[1128.720769,330997130],[1130.346611,426.176161],[1222.856199,134.912674],[1222.418195,231.566185],[1222.764348,330.216262],[1223.930344,428.500163],[1320.71461,132.085387],[1321.289067,229.777217],[1321.87164,329.553891],[1322.447658,428.94295]]
        # target_point2 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]
        #
        # ref_point3 =[[1048.26222,154.505889],[1041.577425,243.680528],[1036.586262,334.44916],[1033.466738,425.101294],[1133.303388,149.832579],[1126.943557,243.069717],[1121.873216,338.328862],[1118.280032,433.535441],[1226.2977,148.3655],[1220.726898,244.311902],[1215.81241,342.642014],[1211.719951,441.00226],[1323.725216,150.94911],[1319.327324,247.772821],[1314.808375,347.173176],[1310.279571,446.695395]]
        # target_point3 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]
        # ref_point4 =[[1520.067071,176.250122],[1518.873512,264.955917],[1519.461446,355.623914],[1521.919018,446.568694],[1604.237322,166.081382],[1603.828023,258.958303],[1604.832342,354.157663],[1607.315265,449.629027],[1696.617752,158.335952],[1697.333795,254.069941],[1698.862382,352.390137],[1701.218702,450.961687],[1793.823742,154.155117],[1795.859591,250.936754],[1797.94725,350.378917],[1800.035124,450.036339]]
        # target_point4 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]
        # ref_point5 =[[2028.517985,146.074359],[2026.090491,235.776906],[2025.435799,326.797759],[2026.644718,417.427171],[2113.481801,137.13766],[2111.458175,231.070551],[2110.823335,326.650751],[2111.661512,421.798527],[2206.692315,131.154710],[2205.400914,227.960737],[2204.877395,326.674206],[2205.177086,424.928475],[2304.664985,129.266132],[2304.334334,227.081858],[2303.992742,326.908743],[2303.648449,426.271320]]
        # target_point5 =[[100,100],[100,200],[100,300],[100,400],[200,100],[200,200],[200,300],[200,400],[300,100],[300,200],[300,300],[300,400],[400,100],[400,200],[400,300],[400,400]]

        #---------------------------------------------

        ref_point0 =[[6.420463,38.808252],[358.760552,-0.530286],[709.057934,48.573150],[0.877583,295.527292],[355.080589,295.676351],[708.783718,304.112772],[12.260712,543.286761],[351.510070,580.291971],[691.740797,551.887717]]
        target_point0 =[[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]
        ref_point1 =[[451.353136,71.528346],[798.219543,16.397339],[1149.444620,45.509348],[454.893413,324.817512],[808.389355,310.111124],[1161.960843,301.081327],[475.259411,573.927938],[818.246608,595.807061],[1157.330357,549.247688]]
        target_point1 = [[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]
        ref_point2 =[[973.747861,80.200225],[1320.160453,38.437659],[1667.028865,76.036089],[968.214767,332.642697],[1321.871640,329.553891],[1675.251882,328.650571],[1675.251882,328.650571],[1323.528619,617.404142],[1666.467635,578.648655]]
        target_point2 = [[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]

        ref_point3 =[[981.059097,79.265228],[1327.916560,58.601630],[1669.197967,112.512356],[961.527216,331.097176],[1314.808375,347.173176],[1667.871366,362.259085],[959.073008,581.600905],[1301.617641,636.783395],[1649.405574,614.871165]]
        target_point3 = [[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]
        ref_point4 =[[1448.974078,106.510234],[1791.880014,61.917856],[2136.941044,91.436475],[1444.330321,356.792437],[1797.947250,350.378917],[2151.201704,342.414019],[1456.728689,609.386935],[1804.017626,640.668262],[2148.458405,593.761111]]
        target_point4 =[[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]
        ref_point5 =[[1957.636540,74.095944],[2304.978699, 35.430054],[2651.614491,76.806850],[1950.294661,327.058488],[2303.992742,326.908743],[2657.371927,329.417747],[1959.943823,577.042675],[2302.985053,614.473284],[2646.147309,579.372929]]
        target_point5 =[[0,0],[400,0],[800,0],[0,300],[400,300],[800,300],[0,600],[400,600],[800,600]]




        ref_points=[ref_point0,ref_point1,ref_point2,ref_point3,ref_point4,ref_point5]
        target_points=[target_point0,target_point1,target_point2,target_point3,target_point4,target_point5]

        # for i in range(6):
        #     for j in range(len(ref_points[0])):
        #         ref_points[i][j][0],ref_points[i][j][1]=ref_points[i][j][1],ref_points[i][j][0]
        #         target_points[i][j][0], target_points[i][j][1] = target_points[i][j][1], target_points[i][j][0]


        # for point in ref_point0:
        #     cv2.circle(self.imgFiles[0], center=tuple(point), radius=5, color=(255, 0, 0), thickness=-1)
        # for point in target_point0:
        #     cv2.circle(self.imgFiles[1], center=tuple(point), radius=5, color=(255, 0, 0), thickness=-1)
        #
        # result_img0 = cv2.cvtColor(self.imgFiles[0], cv2.COLOR_BGR2RGB)
        # result_img1 = cv2.cvtColor(self.imgFiles[1], cv2.COLOR_BGR2RGB)
        # plt.figure(0)
        # plt.imshow(result_img0)
        # plt.figure(1)
        # plt.imshow(result_img1)
        # plt.show()

        height,width,channels = self.imgFiles[1].shape
        ref_height,ref_width,ref_channels = self.refImg.shape
        # print(height,width,channels)

        canvas_height = 685
        canvas_width = 2672
        self.blank_image = np.zeros((canvas_width,canvas_height, ref_channels), np.uint8)
        self.reshapedImgs=[]


        self.HM,self.mask=[],[]

        def myWarpPerspective(img,HM,resolution):
            res = np.zeros((resolution[0],resolution[0], 3), np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    temp_vector = np.matmul(np.array(HM),np.array([i,j,1]).T)
                    i_reshape = temp_vector[0]/temp_vector[2]
                    j_reshape = temp_vector[1] / temp_vector[2]
                    res[int(i_reshape),int(j_reshape)] = img[i,j]
            return res



        for n in range(len(ref_points)):
            HM, mask = cv2.findHomography(np.array(target_points[n]),np.array(ref_points[n]), cv2.RANSAC)
            # HM=np.swapaxes(HM,0,1)
            self.HM.append(HM)
            self.mask.append(mask)

            # self.reshapedImg = cv2.warpPerspective(np.swapaxes(self.imgFiles[n],0,1), HM, (canvas_width,canvas_width))
            self.reshapedImg = myWarpPerspective(np.swapaxes(self.imgFiles[n], 0, 1), HM,(canvas_width, canvas_width))

            # self.reshapedImg = np.swapaxes(self.reshapedImg,0,1)
            self.reshapedImgs.append(self.reshapedImg)
            print(n)
            print(self.reshapedImg.shape)
            for i in range(canvas_width):
                for j in range(canvas_height):
                    if(not np.all((self.reshapedImg[i][j] == 0))):
                        self.blank_image[i][j] = self.reshapedImg[i][j]

        self.blank_image = cv2.cvtColor(self.blank_image, cv2.COLOR_BGR2RGB)
        self.refImg = cv2.cvtColor(self.refImg, cv2.COLOR_BGR2RGB)


        # fig = plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(self.blank_image)
        # plt.subplot(2, 1, 2)
        # plt.imshow(self.refImg)
        # plt.show()


        # # self.imgReshape = cv2.warpAffine(cv2.warpPerspective(self.imgFiles[0],self.HM,(width,height)),T,(width,height))
        # self.imgReshape = cv2.warpPerspective(self.imgFiles[1], self.HM, (2000, 1000))
        # self.imgReshape[:height,:width,:] = self.imgFiles[0]
        # self.imgReshape = cv2.cvtColor(self.imgReshape, cv2.COLOR_BGR2RGB)
        # plt.imshow(self.imgReshape)
        # plt.show()

    def stitch_images(self):
        self.imageStitcher = cv2.Stitcher_create()
        self.error,self.stitcher_output = self.imageStitcher.stitch(self.imgFilesProject)

        # plt.subplot(3, 1, 1)
        # plt.imshow(self.stitcher_output)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(np.swapaxes( self.blank_image,0,1)[75:577,82:2648,:]) #[75:577,82:2648,:] from Hugin
        plt.subplot(2, 1, 2)
        plt.imshow(self.refImg)
        plt.show()


        # for i in range(6):
        #     plt.figure()
        #     plt.subplot(2, 1, 1)
        #     plt.imshow(self.imgFilesProject[i])
        #     plt.subplot(2, 1, 2)
        #     plt.imshow(self.reshapedImgs[i][:,800*i:800*(i+1),:])
        #     plt.show()

        # plt.subplot(6, 1, 1)
        # plt.imshow(self.reshapedImgs[0])
        # plt.subplot(6, 1, 2)
        # plt.imshow(self.reshapedImgs[1])
        # plt.subplot(6, 1, 3)
        # plt.imshow(self.reshapedImgs[2])
        # plt.subplot(6, 1, 4)
        # plt.imshow(self.reshapedImgs[3])
        # plt.subplot(6, 1, 5)
        # plt.imshow(self.reshapedImgs[4])
        # plt.subplot(6, 1, 6)
        # plt.imshow(self.reshapedImgs[5])
        # plt.show()

myMatcher = imgMatcher()
myMatcher.init_values()
# myMatcher.get_matches()
# myMatcher.find_homography()
myMatcher.find_HM()
myMatcher.stitch_images()