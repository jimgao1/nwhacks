# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
from sys import platform
import argparse
import numpy as np

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx*dx + dy*dy) ** 0.5

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../bin/python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
        import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000192.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"
    params['hand'] = 1
    # params["net_resolution"] = "160x80"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    cap = cv2.VideoCapture(0)

    draw_pose_ids = False
    draw_hand_ids = False
    draw_pointer = True

    finger_pos = dict()

    canvas = np.zeros((480, 640, 3))
    canvas.fill(255)

    while True:
        # Process Image
        ret, imageToProcess = cap.read()

        if not ret: print("Failed to capture image")
        print(imageToProcess.shape)
        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess

        start_time = int(round(time.time() * 1000))
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        end_time = int(round(time.time() * 1000))

        print("frame time: %d ms" % (end_time - start_time))
        model_img = datum.cvOutputData

        # for point in datum.poseKeypoints
        if draw_pose_ids and datum.poseKeypoints is not None:
            for person_id, person in enumerate(datum.poseKeypoints):
                for point_id, point in enumerate(person):
                    if point[0] == 0.0 and point[1] == 0.0: continue
                    model_img = \
                        cv2.putText(model_img, "%d" % point_id,
                                    (point[0], point[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        if datum.handKeypoints is not None:
            hand_colors = [(0, 255, 255), (255, 0, 255)]
            for hand_id, hands in enumerate(datum.handKeypoints):
                if hands is None: continue
                num_people = hands.shape[0]
                for person_id in range(num_people):
                    for point_id in range(21):
                        point = hands[person_id][point_id]
                        if draw_hand_ids:
                            model_img = cv2.putText(model_img, "%c%d" % ('L' if hand_id == 0 else 'R', point_id),
                                                    (point[0], point[1]), cv2.FONT_HERSHEY_DUPLEX,
                                                    0.5, hand_colors[hand_id])

                        if hand_id == 1 and 5 <= point_id <= 8:
                            finger_pos[point_id] = (point[0], point[1])

                        clicked = False
                        if 5 in finger_pos and 7 in finger_pos and 8 in finger_pos:
                            d1 = dist(finger_pos[5], finger_pos[8])
                            d2 = dist(finger_pos[7], finger_pos[8])
                            print(d1, d2, d2/d1)

                            clicked = d2/d1 < 0.3

                        # make right-index a special point
                        if draw_pointer and point_id == 8:
                            model_img = cv2.circle(model_img, (point[0], point[1]), 20,
                                                   (0, 0, 255) if clicked else (255, 255, 255),
                                                   4)

                        if hand_id == 1 and point_id == 8 and clicked:
                            model_img = cv2.circle(model_img, (point[0], point[1]), 2, (255, 255, 0), 2)
                            imageToProcess = cv2.circle(imageToProcess, (point[0], point[1]), 2, (255, 255, 0), 2)
                            canvas = cv2.circle(canvas, (point[0], point[1]), 2, (255, 255, 0), 2)

        imageToProcess = np.where(canvas < 255.0, imageToProcess, canvas)
        # imageToProcess = (imageToProcess + canvas) // 2

        cv2.imshow("frame", imageToProcess)
        cv2.imshow("output", model_img)
        cv2.imshow("canvas", canvas)
        cv2.waitKey(1)
except Exception as e:
    raise e
