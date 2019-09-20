from collections import deque
from imutils.video import VideoStream

import numpy as np
import argparse
import cv2
import imutils
import time

ball_color_lower1 = (0, 155, 100)
ball_color_upper1 = (5, 255, 255)

ball_color_lower2 = (175, 155, 100)
ball_color_upper2 = (180, 255, 255)

goal_color_lower = (25, 0, 180)
goal_color_upper = (45, 100, 255)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())


if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])


time.sleep(2.0)

frame = vs.read()
frame = frame[1] if args.get("video", False) else frame

frame = imutils.resize(frame, width=600)
blank = np.zeros(frame.shape[0:2])

blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

goal_mask = cv2.inRange(hsv, goal_color_lower, goal_color_upper)
goal_mask = cv2.erode(goal_mask, None, iterations=2)
goal_mask = cv2.dilate(goal_mask, None, iterations=2)

initial_goal_contours, hierarchy = cv2.findContours(goal_mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

for contour in initial_goal_contours:
    if not cv2.contourArea(contour) > 1000:
        initial_goal_contours.remove(contour)

goal_0_image = cv2.drawContours(blank.copy(), initial_goal_contours, 0, 1, -1)
goal_1_image = cv2.drawContours(blank.copy(), initial_goal_contours, 1, 1, -1)

scores = np.zeros(len(initial_goal_contours))

points = deque(maxlen=args["buffer"])


def generate_ball_mask(image):
    ball_mask1 = cv2.inRange(image, ball_color_lower1, ball_color_upper1)
    ball_mask2 = cv2.inRange(image, ball_color_lower2, ball_color_upper2)

    ball_mask = ball_mask1 + ball_mask2
    ball_mask = cv2.erode(ball_mask, None, iterations=2)
    ball_mask = cv2.dilate(ball_mask, None, iterations=2)

    return ball_mask


def generate_ball_contours(mask):
    ball_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    ball_contours = imutils.grab_contours(ball_contours)

    return ball_contours


while True:
    frame = vs.read()

    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    ball_mask = generate_ball_mask(hsv)
    ball_contours = generate_ball_contours(ball_mask.copy())

    ball_center = None

    ball_image = cv2.drawContours(blank.copy(), ball_contours, 0, 1, -1)

    goal_0_collision = np.logical_and(ball_image, goal_0_image).any()
    goal_1_collision = np.logical_and(ball_image, goal_1_image).any()

    cv2.putText(frame, str(scores), (50,50), 0, 2, (0, 255, 255), 4)

    if goal_0_collision:
        cv2.putText(frame, 'GOAL 0!', (100,100), 0, 3, (0, 255, 0), 4)
        scores[0] += 1

    if goal_1_collision:
        cv2.putText(frame, 'GOAL 1!', (200,100), 0, 3, (255, 255, 0), 4)
        scores[1] += 1

    if len(ball_contours) > 0:
        c = max(ball_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)

    if len(initial_goal_contours) > 0:
        cv2.drawContours(frame, initial_goal_contours, -1, (255, 0, 255), -1)

    points.appendleft(ball_center)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
