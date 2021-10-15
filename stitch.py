#!/usr/bin/env python

import cv2
import sys, traceback
import glob
import os
import numpy as np

def get_video_filenames(folder_name):
    vid_folder = folder_name + "/processed_videos/"
    video_filenames = glob.glob(vid_folder + "*.mp4")
    return sorted(video_filenames)

def create_video_map(video_filenames):
    video_map = {}
    for video_filename in video_filenames:
        prefix = video_filename.split("/")[-1].replace(".mp4","")
        video_map[prefix] = cv2.VideoCapture(video_filename)

    return video_map

def stitch_videos(video_map, output_path, debug=False, skip=0 ):

    idx = 0

    directory = os.path.dirname(output_path)
    

    skipped = 0
    while(video_map["1.1"].isOpened()):
        frame_map = {}
        if skipped < skip:
            for prefix in sorted(video_map.keys()):
                ret, frame = video_map[prefix].read()
            skipped += 1
            continue

        no_frames = False

        for prefix in sorted(video_map.keys()):

            ret, frame = video_map[prefix].read()
            
            if ret == True:
                # print("got video:", prefix)
                frame_map[prefix] = frame
                # cv2.imshow(prefix,frame)
            else:
                no_frames = True

        idx += 1

        # if idx % 50 != 0:
        #     continue

        if no_frames:
            print("missing frames @", idx)
            break

        

        left_pano, right_pano, pano = stitch_frame(frame_map, output_path, idx)
        
        if pano is not None:
            print("frame_size:", pano.shape)
            # resized_pano = cv2.resize(pano, ( 1300, 320), interpolation = cv2.INTER_AREA)
            # cv2.imshow("pano", resized_pano)
            # cv2.waitKey(25)

            if not os.path.exists(directory + "/../imgs/panos/"+str(idx)):
                os.makedirs(directory + "/../imgs/panos/"+str(idx))
                cv2.imwrite(directory + "/../imgs/panos/"+str(idx)+"/left_pano.png", left_pano)
                cv2.imwrite(directory + "/../imgs/panos/"+str(idx)+"/right_pano.png", right_pano)
                cv2.imwrite(directory + "/../imgs/panos/"+str(idx)+"/pano.png", pano)


        if debug:
            # cv2.waitKey(10000)
            break 


def save_images(frame_map, output_path, idx):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory + "/../imgs/raw/"+str(idx)):
        os.makedirs(directory + "/../imgs/raw/"+str(idx))

    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/1.1.png", frame_map["1.1"])
    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/1.2.png", frame_map["1.2"])
    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/1.4.png", frame_map["1.4"])
    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/2.1.png", frame_map["2.1"])
    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/2.2.png", frame_map["2.2"])
    cv2.imwrite(directory + "/../imgs/raw/"+str(idx) + "/2.4.png", frame_map["2.4"])

    # sys.exit(0)


def stitch_frame(frame_map, output_path, idx):
    # imgs = []
    # for prefix in frame_map:
    #     imgs.append(frame_map[prefix])
    # for prefix in sorted(frame_map.keys()):
        # cv2.imshow(prefix, frame_map[prefix])

    pano = None
    try:
        left_imgs = [
            frame_map["1.1"], 
            frame_map["1.2"], 
            frame_map["1.4"]
            
        ]
        right_imgs = [
            frame_map["2.2"],
            frame_map["2.1"],
            frame_map["2.4"]
        ]

        stitcher = cv2.createStitcher(False)
        # status, M = stitcher.estimateTransform(imgs,np.array([]))
        save_images(frame_map, output_path, idx)
        print("Computing left pano")
        status, left_pano = stitcher.stitch(left_imgs)
        if left_pano is None:
            print("Recalculating left pano, because of None")
            status, left_pano = stitcher.stitch(left_imgs)
            print("status:", status)
        print("Computing right pano")
        status, right_pano = stitcher.stitch(right_imgs)
        if right_pano is None:
            print("Recalculating right pano, because of None")
            status, right_pano = stitcher.stitch(right_imgs)
            print("status:", status)
        print("Computing full pano", left_pano.shape, right_pano.shape)
        status, pano = stitcher.stitch([left_pano, right_pano])
        
        # cv2.imshow("right_pano", pano)
        # cv2.waitKey(10000)
        
    except Exception as e: 
        print("error:",e)
        traceback.print_exc(file=sys.stdout)
        return None, None, None


    resp = None

    if status == cv2.STITCHER_OK:
        resp = pano
        # cv2.imshow("pano", pano)
        # cv2.waitKey(25)

    return left_pano, right_pano, resp

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    
    folder_name = sys.argv[1]

    video_filenames = get_video_filenames(folder_name)

    print ( "video_filenames:", video_filenames )

    video_map = create_video_map(video_filenames)

    print ( "video map:", video_map )

    video_file = folder_name + "/panoramic_videos/pano.mp4"
    ensure_dir(video_file)
    stitched_video = stitch_videos(video_map, video_file, False, 100)

