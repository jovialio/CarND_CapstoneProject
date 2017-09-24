"""
Read from rosbag and dump to folder
"""

import argparse
import os
import shutil
import rosbag
from cv_bridge import CvBridge
import cv2
import tqdm

def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbagpath", help="rosbag file")
    parser.add_argument("imgdirpath", help="output path to folder rosbag images")

    args = parser.parse_args()
    rosbagpath = args.rosbagpath
    imgdirpath = args.imgdirpath

    # delete folder if present and import fresh rosbag
    shutil.rmtree(imgdirpath, ignore_errors=True)
    os.makedirs(imgdirpath)

    bridge = CvBridge()

    with rosbag.Bag(rosbagpath) as bag:

        for index, (topic, msg, t) in (tqdm.tqdm(enumerate(bag.read_messages(topics="/image_raw")))):

            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            path = os.path.join(imgdirpath, "%09i.png" % index)
            cv2.imwrite(path, img)

if __name__ == "__main__":
    main()