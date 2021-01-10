# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
from sys import platform
import argparse
import numpy as np
import collections

class FuckBuffer:
    def __init__(self, queue_size=10, gamma=0.5):
        self.count = 0
        self.queue_size = queue_size
        self.gamma = gamma
        self.sum = 0.0
        self.wsum = 0.0
        self.c = collections.deque()

    def append(self, e):
        e = float(e)

        self.c.append(e)
        self.count += 1
        self.sum += e
        self.wsum = self.gamma * self.wsum + e

        if self.count > self.queue_size:
            val = self.c.popleft()
            self.sum -= val
            self.wsum -= val * self.gamma**(self.count+1)
            self.count -= 1

    def pop(self):
        self.count -= 1
        self.sum -= self.c.popleft()

    def avg(self):
        return self.sum / self.count

    def wavg(self):
        # if self.count:
        #     return self.wsum ** (1/self.count)
        total_weight = (1 - self.gamma ** (self.count + 1)) / (1 - self.gamma)
        return self.wsum / total_weight

    def clear(self):
        self.count = 0
        self.wsum = self.sum = 0.0
        self.c.clear()

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
    params['hand_net_resolution'] = '288x288'
    params['render_pose'] = 1
    params['hand_render'] = 1
    # params['hand_net_resolution'] = '192x192'
    # params["net_resolution"] = "640x480"
    # params["hand_detector"] = 3
    # params["body"] = 0

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
    # cap = cv2.VideoCapture("C:/Users/Jim/Desktop/test.mp4")
    # cap = cv2.VideoCapture("http://192.168.2.14:8080/playlist.m3u")
    # cap = cv2.VideoCapture("rtsp://192.168.2.14:5554/out.h264")

    draw_pose_ids = True
    draw_hand_ids = False
    draw_pointer = True

    finger_pos = dict()

    # dimension of video stream
    height, width = 480, 640

    # dimension of canvas
    canvas_view_corner = np.array((height//2, width//2), dtype=np.int32)
    cancer = [np.zeros((height*2, width*2, 1)), np.zeros((height*2, width*2, 1))]

    previous_point = dict()

    cur_clicked = False
    last_frame_time = 0
    ratio = 0
    fprop = []

    cur_dragging = False
    last_drag_pos = None

    click_debouncer = FuckBuffer(queue_size=10)
    drag_debouncer = FuckBuffer(queue_size=20)
    motion_x = [FuckBuffer(queue_size=10, gamma=0.05), FuckBuffer(queue_size=10, gamma=0.05)]
    motion_y = [FuckBuffer(queue_size=10, gamma=0.05), FuckBuffer(queue_size=10, gamma=0.05)]

    arm_x = FuckBuffer(queue_size=5, gamma=0.05)
    arm_y = FuckBuffer(queue_size=5, gamma=0.05)

    while True:
        # Process Image
        ret, imageToProcess = cap.read()
        if not ret: print("Failed to capture image")

        print(imageToProcess.shape)

        imageToProcess = cv2.resize(imageToProcess, (640, 480))

        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess

        start_time = int(round(time.time() * 1000))
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        end_time = int(round(time.time() * 1000))

        model_img = datum.cvOutputData

        cv2.putText(model_img, "%.2f fps" % (1000 / (start_time-last_frame_time)),
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0))

        if cur_clicked:
            reason = "clicked"
        elif not all([i in finger_pos for i in [0, 4, 8, 12, 16, 20]]):
            reason = "insufficient"
        else:
            reason = "released"

        if click_debouncer.count:
            reason += "(%d)" % int(click_debouncer.avg())

        click_debouncer.append(cur_clicked)

        cv2.putText(model_img, "prop = %.3f (%s)" % (ratio, reason),
                    (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255))

        # cv2.putText(model_img, "fprop = " + str(["%.1f" % x for x in fprop]),
        #             (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255))

        if len(fprop):
            extend_count = sum([ fprop[i] > 1.9 for i in [1, 2, 3]])
            drag_debouncer.append(extend_count)
            cv2.putText(model_img, "%s (%d)" % ("dragging" if drag_debouncer.wavg() > 1.5 else "not dragging", extend_count),
                        (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255))

            dragging = drag_debouncer.wavg() > 1.5

            cur_pos = np.array((int(arm_x.wavg()), int(arm_y.wavg())))

            if cur_dragging and dragging:
                pos_diff = cur_pos - last_drag_pos
                # canvas_view_corner += pos_diff

                canvas_view_corner[0] -= pos_diff[0]
                canvas_view_corner[1] -= pos_diff[1]

            last_drag_pos = cur_pos
            cur_dragging = dragging

        last_frame_time = start_time

        # for point in datum.poseKeypoints
        if draw_pose_ids and datum.poseKeypoints is not None:
            for person_id, person in enumerate(datum.poseKeypoints):
                for point_id, point in enumerate(person):
                    if point[0] == 0.0 and point[1] == 0.0: continue

                    if point_id == 4:
                        arm_x.append(point[0])
                        arm_y.append(point[1])

                    model_img = cv2.putText(model_img, "%d" % point_id,
                                    (point[0], point[1]),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

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

                        cur_clicked = False
                        # if 5 in finger_pos and 7 in finger_pos and 8 in finger_pos:
                        #     d1 = dist(finger_pos[5], finger_pos[8])
                        #     d2 = dist(finger_pos[7], finger_pos[8])
                        #
                        #     clicked = d2/d1 < 0.6

                        # click
                        if all([i in finger_pos for i in [0, 4, 8, 12, 16, 20]]):
                            other_fingers = sum([dist(finger_pos[0], finger_pos[i]) for i in [4, 12, 16, 20]]) / 4.0
                            index_finger = dist(finger_pos[0], finger_pos[8])

                            ratio = (index_finger - other_fingers) / other_fingers
                            cur_clicked = ratio > 0.7

                            if all([i in finger_pos for i in [0, 1, 5, 9, 13, 17]]):
                                fprop = [ dist(finger_pos[i+3], finger_pos[0]) / dist(finger_pos[i], finger_pos[0])
                                          for i in [1, 5, 9, 13, 17] ]


                        # make right-index a special point
                        if draw_pointer and point_id == 8:
                            model_img = cv2.circle(model_img, (point[0], point[1]), 20,
                                                   (0, 0, 255) if cur_clicked else (255, 255, 255),
                                                   4)
                            # draw smoothed point
                            model_img = cv2.circle(model_img, (int(motion_x[hand_id].wavg()), int(motion_y[hand_id].wavg())), 20,
                                                   (0, 0, 127) if cur_clicked else (127, 127, 127),
                                                   3)

                        if point_id == 8:
                            motion_x[hand_id].append(point[0])
                            motion_y[hand_id].append(point[1])
                            smoothed = (int(motion_x[hand_id].wavg()), int(motion_y[hand_id].wavg()))

                            if not cur_dragging and (cur_clicked or click_debouncer.avg() >= 1.0):
                                model_img = cv2.circle(model_img, (point[0], point[1]), 2, hand_colors[hand_id], 2)
                                # imageToProcess = cv2.circle(imageToProcess, (point[0], point[1]), 2, hand_colors[hand_id], 2)

                                cancer[hand_id] = cv2.circle(cancer[hand_id],
                                                             tuple(np.array((point[0], point[1]), dtype=np.int32) + canvas_view_corner), 2, (255,), 2)

                                if hand_id in previous_point:
                                    cancer[hand_id] = cv2.line(cancer[hand_id], tuple(previous_point[hand_id] + canvas_view_corner),
                                        tuple(smoothed + canvas_view_corner), (255, ), 2)

                            # clear motion smoothing buffer on release
                            # if click_debouncer.avg() < 2.0:
                            #     motion_x.clear()
                            #     motion_y.clear()

                            previous_point[hand_id] = smoothed


                            # imageToProcess = np.where(canvas < 255.0, imageToProcess, canvas)
        # imageToProcess = (imageToProcess + canvas) // 2


        cancer_disp = [
            np.clip(cancer[i][canvas_view_corner[1]:canvas_view_corner[1] + height, canvas_view_corner[0]:canvas_view_corner[0] + width, :], 0, 255).astype('uint8')
            for i in [0, 1]
        ]

        print(cancer_disp[0].shape, cancer_disp[1].shape)

        combined_cancer = np.clip(sum([draw_color[i] * cancer_disp[i] for i in [0, 1]]), 0, 255).astype('uint8')

        combined = imageToProcess - cv2.bitwise_and(imageToProcess, imageToProcess, mask=cv2.bitwise_or(cancer_disp[0], cancer_disp[1]))
        combined = combined + combined_cancer
        combined = np.clip(combined, 0, 255).astype('uint8')

        cv2.imshow("frame", cv2.flip(combined, 1))
        cv2.imshow("output", model_img)
        cv2.imshow("canvas", cv2.flip(combined_cancer, 1))
        # cv2.imshow("c0", cancer[0])
        # cv2.imshow("c1", cv2.rectangle(255 - cancer[1], tuple(canvas_view_corner),
        #                                tuple(canvas_view_corner + np.array((640, 480))),
        #                                (0, 127, 0)))
        key = cv2.waitKey(1)

        # key binds to erase canvas
        if key == ord('l') or key == ord('c'): cancer[0].fill(0)
        if key == ord('r') or key == ord('c'): cancer[1].fill(0)

        # translating the canvas view
        if key == ord('j'): canvas_view_corner[0] += 10
        if key == ord('k'): canvas_view_corner[0] -= 10
        if key == ord('h'): canvas_view_corner[1] -= 10
        if key == ord('l'): canvas_view_corner[1] += 10
except Exception as e:
    raise e
