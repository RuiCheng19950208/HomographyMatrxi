from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pims
import os
import datetime
import math
import glob
import pdb
from scipy.optimize import fsolve,leastsq
from math import sin,cos


def nothing(x):
    pass

def randrange(n, vmin, vmax):
    '''''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin) * np.random.rand(n) + vmin



video_name_top = '2018-02-08-13-51-00_rat03_top_view.seq'
video_name_left = '2018-02-08-13-51-00_rat03_left_view.seq'
video_name_front = '2018-02-08-13-51-00_rat03_front_view.seq'
video_check_top = '2018-02-08-13-10-14_checker_top_view.seq'
video_check_left = '2018-02-08-13-10-14_checker_left_view.seq'
video_check_front = '2018-02-08-13-10-14_checker_front_view.seq'
# top = pims.open(video_name_top)
# left = pims.open(video_name_left)
# front = pims.open(video_name_front)
# topc = pims.open(video_check_top)
# leftc = pims.open(video_check_left)
# frontc = pims.open(video_check_front)
# video_name = video_name_front
# video_obj1 = topc
# video_obj2 = leftc
# video_obj3 = frontc

# At first run these code to make folders, otherwise don't make it !!!
# os.makedirs('topcheck')
# os.makedirs('leftcheck')
# os.makedirs('frontcheck')
# os.makedirs('top')
# os.makedirs('left')
# os.makedirs('front')


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints1 = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.
objpoints2 = [] # 3d point in real world space
imgpoints2 = [] # 2d points in image plane.
objpoints3 = [] # 3d point in real world space
imgpoints3 = [] # 2d points in image plane.
objpoints01 = [] # 3d point in real world space
imgpoints01 = [] # 2d points in image plane.
objpoints02 = [] # 3d point in real world space
imgpoints02 = [] # 2d points in image plane.
objpoints03 = [] # 3d point in real world space
imgpoints03 = [] # 2d points in image plane.


imagestop = glob.glob('top/*.jpg')
imagesfront = glob.glob('front/*.jpg')
imagesleft = glob.glob('left/*.jpg')


imagestopc = glob.glob('topcheck/*.jpg')
imagesfrontc = glob.glob('frontcheck/*.jpg')
imagesleftc = glob.glob('leftcheck/*.jpg')

firstcamera=imagesfrontc+[]
secondcamera=imagestopc+[]
thirdcamera=imagesleftc+[]

print(imagesleftc)



#Run these code to save your frames as pictures

# for frame_idx in range(len(video_obj1) - 1):
#     # cv2.imwrite(str(frame_idx)+'.jpg', cv2.cvtColor(video_obj1[frame_idx], cv2.COLOR_GRAY2BGR))
#     cv2.imencode('.jpg', cv2.cvtColor(video_obj1[frame_idx], cv2.COLOR_GRAY2BGR))[1].tofile('top/'+str(frame_idx)+'.jpg')
#
# for frame_idx in range(len(video_obj1) - 1):
#     # cv2.imwrite(str(frame_idx)+'.jpg', cv2.cvtColor(video_obj1[frame_idx], cv2.COLOR_GRAY2BGR))
#     cv2.imencode('.jpg', cv2.cvtColor(video_obj2[frame_idx], cv2.COLOR_GRAY2BGR))[1].tofile('left/'+str(frame_idx)+'.jpg')
#
# for frame_idx in range(len(video_obj1) - 1):
#     # cv2.imwrite(str(frame_idx)+'.jpg', cv2.cvtColor(video_obj1[frame_idx], cv2.COLOR_GRAY2BGR))
#     cv2.imencode('.jpg', cv2.cvtColor(video_obj3[frame_idx], cv2.COLOR_GRAY2BGR))[1].tofile('front/'+str(frame_idx)+'.jpg')









# #Run these code to find the picture with clear corner

#
# i=499
# for fname in imagestop[500:]:
#     i=i+1
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         # objpoints.append(objp)
#         # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         # imgpoints.append(corners2)
#         # # Draw and display the corners
#         # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         cv2.imencode('.jpg', img)[1].tofile('C:/Users/chengrui/PycharmProjects/untitled8/topcheck/'+ str(i) + '.jpg')
# i=499
# for fname in imagesfront[500:]:
#     i=i+1
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         # objpoints.append(objp)
#         # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         # imgpoints.append(corners2)
#         # # Draw and display the corners
#         # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         cv2.imencode('.jpg', img)[1].tofile('C:/Users/chengrui/PycharmProjects/untitled8/frontcheck/'+ str(i) + '.jpg')
# i=499
# for fname in imagesleft[500:]:
#     i=i+1
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         # objpoints.append(objp)
#         # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         # imgpoints.append(corners2)
#         # Draw and display the corners
#         # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         cv2.imencode('.jpg', img)[1].tofile('C:/Users/chengrui/PycharmProjects/untitled8/leftcheck/'+ str(i) + '.jpg')


k=6 #How many frames we want to use?

print(k)



i=1
for fname1 in firstcamera:    #the name(imagesfrontc,imagesleftc,imagestopc) determines which one is Camera1 and Which one is Camera2
    for fname2 in secondcamera:
        if str(fname1[-7:]) == str(fname2[-7:]):
            if i<=k:
                i=i+1
                print(fname1[-7:])
                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret1, corners1 = cv2.findChessboardCorners(gray1, (7, 6), None)
                ret2, corners2 = cv2.findChessboardCorners(gray2, (7, 6), None)
                # If found, add object points, image points (after refining them)
                if ret1 == True:
                    objpoints1.append(objp)
                    objpoints2.append(objp)
                    corners12 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners22 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                    imgpoints1.append(corners12)
                    imgpoints2.append(corners22)
                    # Draw and display the corners
                    img1 = cv2.drawChessboardCorners(img1, (7, 6), corners12, ret1)
                    img2 = cv2.drawChessboardCorners(img2, (7, 6), corners22, ret2)
                    cv2.imshow('img1', img1)
                    cv2.imshow('img2', img2)
                    cv2.waitKey(0)
cv2.destroyAllWindows()
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray1.shape[::-1],None,None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray2.shape[::-1],None,None)




i=1
for fname1 in firstcamera:    #the name(imagesfrontc,imagesleftc,imagestopc) determines which one is Camera1 and Which one is Camera2
    for fname3 in thirdcamera:
        if str(fname1[-7:]) == str(fname3[-7:]):
            if i<=k:
                i=i+1
                print(fname1[-7:])
                img01 = cv2.imread(fname1)
                img3 = cv2.imread(fname3)
                gray01 = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret01, corners01 = cv2.findChessboardCorners(gray01, (7, 6), None)
                ret3, corners3 = cv2.findChessboardCorners(gray3, (7, 6), None)
                # If found, add object points, image points (after refining them)
                if ret01 == True:
                    objpoints01.append(objp)
                    objpoints3.append(objp)
                    corners012 = cv2.cornerSubPix(gray01, corners01, (11, 11), (-1, -1), criteria)
                    corners32 = cv2.cornerSubPix(gray3, corners3, (11, 11), (-1, -1), criteria)
                    imgpoints01.append(corners012)
                    imgpoints3.append(corners32)
                    # Draw and display the corners
ret01, mtx01, dist01, rvecs01, tvecs01 = cv2.calibrateCamera(objpoints01, imgpoints01, gray01.shape[::-1],None,None)
ret3, mtx3, dist3, rvecs3, tvecs3 = cv2.calibrateCamera(objpoints3, imgpoints3, gray3.shape[::-1],None,None)



i=1
for fname2 in secondcamera:    #the name(imagesfrontc,imagesleftc,imagestopc) determines which one is Camera1 and Which one is Camera2
    for fname3 in thirdcamera:
        if str(fname2[-7:]) == str(fname3[-7:]):
            if i<=k:
                i=i+1
                print(fname2[-7:])
                img02 = cv2.imread(fname2)
                img03 = cv2.imread(fname3)
                gray02 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)
                gray03 = cv2.cvtColor(img03, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret02, corners02 = cv2.findChessboardCorners(gray02, (7, 6), None)
                ret03, corners03 = cv2.findChessboardCorners(gray03, (7, 6), None)
                # If found, add object points, image points (after refining them)
                if ret02 == True:
                    objpoints02.append(objp)
                    objpoints03.append(objp)
                    corners022 = cv2.cornerSubPix(gray02, corners02, (11, 11), (-1, -1), criteria)
                    corners032 = cv2.cornerSubPix(gray03, corners03, (11, 11), (-1, -1), criteria)
                    imgpoints02.append(corners022)
                    imgpoints03.append(corners032)
                    # Draw and display the corners
ret02, mtx02, dist02, rvecs02, tvecs02 = cv2.calibrateCamera(objpoints02, imgpoints02, gray02.shape[::-1],None,None)
ret03, mtx03, dist03, rvecs03, tvecs03 = cv2.calibrateCamera(objpoints03, imgpoints03, gray03.shape[::-1],None,None)




print(mtx1)





rotation1= np.zeros([3,3])
rotation2= np.zeros([3,3])
rotation3= np.zeros([3,3])
rotation01= np.zeros([3,3])

rotation02= np.zeros([3,3])
rotation03= np.zeros([3,3])

#Transpose rotation vector into rotation matrix
cv2.Rodrigues(rvecs1[0], rotation1)
cv2.Rodrigues(rvecs2[0], rotation2)
cv2.Rodrigues(rvecs3[0], rotation3)
cv2.Rodrigues(rvecs01[0], rotation01)
cv2.Rodrigues(rvecs02[0], rotation02)
cv2.Rodrigues(rvecs03[0], rotation03)



##Draw Points in Camera1
# Trans1 = np.transpose(objpoints1[0])+0
# TransR1=np.dot(rotation1,Trans1)
# for j in range(0, imgpoints1[0].shape[0]):
#        TransR1[0,j] = TransR1[0,j] +tvecs1[0][0,0]
#        TransR1[1, j] = TransR1[1, j] + tvecs1[0][1, 0]
#        TransR1[2, j] = TransR1[2, j] + tvecs1[0][2, 0]

# goal=np.transpose(TransR1)

#Draw Points in Camera2

# Trans2 = np.transpose(objpoints2[0])+0
#
# TransR2=np.dot(rotation2,Trans2)
#
#
#
# for j in range(0, imgpoints2[0].shape[0]):
#        TransR2[0,j] = TransR2[0,j] +   tvecs2[0][0,0]+0
#        TransR2[1, j] = TransR2[1, j] + tvecs2[0][1, 0]
#        TransR2[2, j] = TransR2[2, j] + tvecs2[0][2, 0]

# goal=np.transpose(TransR2)




#Use 3DPoints in Camera1 to reconstruct the points in Camera2 after calculate 3D point in Camera 1

sup=[0,0,0,1]

RT1=np.hstack((rotation1,tvecs1[0]))
RT1=np.vstack((RT1,sup))

RT2=np.hstack((rotation2,tvecs2[0]))
RT2=np.vstack((RT2,sup))

RT3=np.hstack((rotation3,tvecs3[0]))
RT3=np.vstack((RT3,sup))

RT01=np.hstack((rotation01,tvecs01[0]))
RT01=np.vstack((RT01,sup))

RT02=np.hstack((rotation02,tvecs02[0]))
RT02=np.vstack((RT02,sup))

RT03=np.hstack((rotation03,tvecs03[0]))
RT03=np.vstack((RT03,sup))

# Find rotation and translation matrix from Camera1 to Camera2
RT1to2=np.dot(RT2,np.linalg.inv(RT1))+0
R1to2=RT1to2[:3,:3]+0
T1to2=RT1to2[:3,3]+0


# Find rotation and translation matrix from Camera2 to Camera1
RT2to1=np.dot(RT1,np.linalg.inv(RT2))+0
R2to1=RT2to1[:3,:3]
T2to1=RT2to1[:3,3]


# Find rotation and translation matrix from Camera1 to Camera3
RT1to3=np.dot(RT3,np.linalg.inv(RT01))+0
R1to3=RT1to3[:3,:3]+0
T1to3=RT1to3[:3,3]+0

# Find rotation and translation matrix from Camera3 to Camera1
RT3to1=np.dot(RT01,np.linalg.inv(RT3))+0
R3to1=RT3to1[:3,:3]+0
T3to1=RT3to1[:3,3]+0

# Find rotation and translation matrix from Camera2 to Camera3

RT2to3=np.dot(RT03,np.linalg.inv(RT02))+0
R2to3=RT2to3[:3,:3]+0
T2to3=RT2to3[:3,3]+0

# Find rotation and translation matrix from Camera3 to Camera2
RT3to2=np.dot(RT02,np.linalg.inv(RT03))+0
R3to2=RT3to2[:3,:3]+0
T3to2=RT3to2[:3,3]+0




Camera3PositionInCamera2=np.array([[T3to2[0]],[T3to2[1]], [T3to2[2]]])
Camera2PositionInCamera3=np.array([[T2to3[0]],[T2to3[1]], [T2to3[2]]])
Camera1PositionInCamera3=np.array([[T1to3[0]],[T1to3[1]], [T1to3[2]]])
Camera3PositionInCamera1=np.array([[T3to1[0]],[T3to1[1]], [T3to1[2]]])
Camera2PositionInCamera1=np.array([[T2to1[0]],[T2to1[1]], [T2to1[2]]])
Camera1PositioninCamera2=np.array([[T1to2[0]],[T1to2[1]],[T1to2[2]]])
# goal1to2=np.transpose(goal)
# goal1to2=np.dot(R1to2,goal1to2)
# for j in range(0, imgpoints1[0].shape[0]):
#        goal1to2[0,j] =goal1to2[0, j]+  T1to2[0]
#        goal1to2[1, j] = goal1to2[1, j] + T1to2[1]
#        goal1to2[2, j] = goal1to2[2, j] + T1to2[2]
# goal=np.transpose(goal1to2)



for i in range(0,k):
    imgpoints1[i] = imgpoints1[i].reshape((-1, 2))
    imgpoints2[i] = imgpoints2[i].reshape((-1, 2))

print(imgpoints1[0])
print(mtx1)
print(mtx2)



#define the goal
goal=np.zeros([42,3])




#Use pixel to solve 3Dpoint.
for i in range(0,imgpoints1[0].shape[0]):
    def f(x):
        X = float(x[0])
        Y = float(x[1])
        Z = float(x[2])
        X2=RT1to2[0,0]*X+RT1to2[0,1]*Y+RT1to2[0,2]*Z+RT1to2[0,3]
        Y2=RT1to2[1,0]*X+RT1to2[1,1]*Y+RT1to2[1,2]*Z+RT1to2[1,3]
        Z2=RT1to2[2,0]*X+RT1to2[2,1]*Y+RT1to2[2,2]*Z+RT1to2[2,3]
        return [
            mtx1[0,0]*X+Z*mtx1[0,2]- Z*imgpoints1[0][i, 0],
            mtx1[1, 1] * Y +Z* mtx1[1, 2] - Z*imgpoints1[0][i, 1],
            mtx2[0, 0] * X2  +Z2* mtx2[0, 2] - Z2*imgpoints2[0][i, 0],
            # mtx2[1, 1] * Y2  + Z2*mtx2[1, 2] -Z2* imgpoints2[0][i, 1],
        ]
    x0 = [10,10, 100]
    result = fsolve(f, x0)




    goal[i,:]=[result[0],result[1],result[2]]
    print('X=' + str(result[0]) + ' Y=' + str(result[1]) + ' Z=' + str(result[2]) )






#Then we should reconstruct the points in Camera2

goal1to2=np.transpose(goal)
goal1to2=np.dot(R1to2,goal1to2)
for j in range(0, imgpoints1[0].shape[0]):
       goal1to2[0,j] =goal1to2[0, j]+  T1to2[0]
       goal1to2[1, j] = goal1to2[1, j] + T1to2[1]
       goal1to2[2, j] = goal1to2[2, j] + T1to2[2]

goal=np.transpose(goal1to2)



#Then we should reconstruct the points in Camera3

goal1to3=np.transpose(goal)
goal1to3=np.dot(R1to3,goal1to3)
for j in range(0, imgpoints1[0].shape[0]):
       goal1to3[0,j] =goal1to3[0, j]+  T1to3[0]
       goal1to3[1, j] = goal1to3[1, j] + T1to3[1]
       goal1to3[2, j] = goal1to3[2, j] + T1to3[2]

# goal=np.transpose(goal1to3)














#Use the coded below to draw plot


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for  i in range(0,imgpoints1[0].shape[0]):

    ax.scatter(goal[i,0], goal[i,1], goal[i,2], c='b')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

cv2.namedWindow('image')
cv2.createTrackbar('rotatey','image',0,400,nothing)
cv2.createTrackbar('rotatez','image',0,400,nothing)
plt.savefig("plot.png")

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    rotatey = cv2.getTrackbarPos('rotatey','image')
    rotatez = cv2.getTrackbarPos('rotatez','image')
    ax.view_init(elev=rotatey, azim=rotatez)
    plt.savefig("plot.png")
    img = cv2.imread('plot.png')  #import gray picture
    cv2.imshow('image',img )
cv2.destroyAllWindows()