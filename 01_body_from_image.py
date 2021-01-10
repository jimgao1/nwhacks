# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
from sys import platform
import argparse
import numpy as np


draw_color = [(0, 255, 0), (69,69,69)]


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
    # cap = cv2.VideoCapture("http://192.168.2.14:8080/playlist.m3u")
    # cap = cv2.VideoCapture("rtsp://192.168.2.14:5554/out.h264")

    draw_pose_ids = False
    draw_hand_ids = False
    draw_pointer = True

    finger_pos = dict()

    cancer = [np.zeros((480, 640, 1)), np.zeros((480, 640, 1))]
    cancer[0].fill(0)
    cancer[1].fill(0)

    previous_point = dict()

    last_frame_time = 0
    ratio = 0

    while True:
        # Process Image
        ret, imageToProcess = cap.read()

        if not ret: print("Failed to capture image")

        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess

        start_time = int(round(time.time() * 1000))
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        end_time = int(round(time.time() * 1000))

        model_img = datum.cvOutputData

        cv2.putText(model_img, "%.2f fps" % (1000 / (start_time-last_frame_time)),
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0))
        cv2.putText(model_img, "prop = %.3f" % ratio,
                    (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))

        last_frame_time = start_time

        # for point in datum.poseKeypoints
        if draw_pose_ids and datum.poseKeypoints is not None:
            for person_id, person in enumerate(datum.poseKeypoints):
                for point_id, point in enumerate(person):
                    if point[0] == 0.0 and point[1] == 0.0: continue
                    model_img = \
                        cv2.putText(model_img, "%d" % point_id,
                                    (point[0], point[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        if datum.handKeypoints is not None:
            hand_colors = [(0, 255, 255), (0, 128, 128)]
            for hand_id, hands in enumerate(datum.handKeypoints):
                if hands is None: continue

                num_people = hands.shape[0]
                for person_id in range(num_people):
                    for point_id in range(21):
                        point = hands[person_id][point_id]
                        if point[0] == 0 and point[1] == 0: continue

                        if draw_hand_ids:
                            model_img = cv2.putText(model_img, "%c%d" % ('L' if hand_id == 0 else 'R', point_id),
                                                    (point[0], point[1]), cv2.FONT_HERSHEY_DUPLEX,
                                                    0.5, hand_colors[hand_id])

                        if hand_id == 1:
                            finger_pos[point_id] = (point[0], point[1])

                        clicked = False
                        # if 5 in finger_pos and 7 in finger_pos and 8 in finger_pos:
                        #     d1 = dist(finger_pos[5], finger_pos[8])
                        #     d2 = dist(finger_pos[7], finger_pos[8])
                        #
                        #     clicked = d2/d1 < 0.6

                        if all([i in finger_pos for i in [0, 4, 8, 12, 16, 20]]):
                            other_fingers = sum([dist(finger_pos[0], finger_pos[i]) for i in [4, 12, 16, 20]]) / 4.0
                            index_finger = dist(finger_pos[0], finger_pos[8])

                            ratio = (index_finger - other_fingers) / other_fingers
                            clicked = ratio > 0.7


                        # make right-index a special point
                        if draw_pointer and point_id == 8:
                            model_img = cv2.circle(model_img, (point[0], point[1]), 20,
                                                   (0, 0, 255) if clicked else (255, 255, 255),
                                                   4)

                        if point_id == 8:
                            if clicked:
                                model_img = cv2.circle(model_img, (point[0], point[1]), 2, hand_colors[hand_id], 2)
                                imageToProcess = cv2.circle(imageToProcess, (point[0], point[1]), 2, hand_colors[hand_id], 2)
                                cancer[hand_id] = cv2.circle(cancer[hand_id], (point[0], point[1]), 2, (255,), 2)

                                if hand_id in previous_point:
                                    cancer[hand_id] = cv2.line(cancer[hand_id], previous_point[hand_id], (point[0], point[1]), (255, ), 2)

                            previous_point[hand_id] = (point[0], point[1])


                            # imageToProcess = np.where(canvas < 255.0, imageToProcess, canvas)
        # imageToProcess = (imageToProcess + canvas) // 2

        for i in [0, 1]: cancer[i] = np.clip(cancer[i], 0, 255).astype('uint8')
        combined_cancer = np.clip(sum([draw_color[i] * cancer[i] for i in [0, 1]]), 0, 255).astype('uint8')

        combined = imageToProcess - cv2.bitwise_and(imageToProcess, imageToProcess, mask=cv2.bitwise_or(cancer[0], cancer[1]))
        combined = combined + combined_cancer
        combined = np.clip(combined, 0, 255).astype('uint8')


        print("combined shape", combined.shape)

        cv2.imshow("frame", cv2.flip(combined, 1))
        cv2.imshow("output", model_img)
        cv2.imshow("canvas", cv2.flip(combined_cancer, 1))
        key = cv2.waitKey(1)

        # key binds to erase canvas
        if key == ord('l') or key == ord('c'): cancer[0].fill(0)
        if key == ord('r') or key == ord('c'): cancer[1].fill(0)
except Exception as e:
    raise e
