from collections import deque
from imutils.video import VideoStream

import numpy as np
import argparse
import cv2
import imutils
import time

BALL_COLOR_LOWER1 = (0, 155, 100)
BALL_COLOR_UPPER1 = (5, 255, 255)

BALL_COLOR_LOWER2 = (175, 155, 100)
BALL_COLOR_UPPER2 = (180, 255, 255)

GOAL_COLOR_LOWER = (25, 0, 180)
GOAL_COLOR_UPPER = (45, 100, 255)

BUFFER_SIZE = 64

vs = VideoStream(src=0).start()


def read_frame():
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    return frame


def generate_ball_mask(image):
    ball_mask1 = cv2.inRange(image, BALL_COLOR_LOWER1, BALL_COLOR_UPPER1)
    ball_mask2 = cv2.inRange(image, BALL_COLOR_LOWER2, BALL_COLOR_UPPER2)

    ball_mask = ball_mask1 + ball_mask2
    ball_mask = cv2.erode(ball_mask, None, iterations=2)
    ball_mask = cv2.dilate(ball_mask, None, iterations=2)

    return ball_mask


def generate_ball_contours(mask):
    ball_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    ball_contours = imutils.grab_contours(ball_contours)

    return ball_contours


def generate_blurred_hsv(image):
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return hsv


def generate_goal_mask(image):
    goal_mask = cv2.inRange(image, GOAL_COLOR_LOWER, GOAL_COLOR_UPPER)
    goal_mask = cv2.erode(goal_mask, None, iterations=2)
    goal_mask = cv2.dilate(goal_mask, None, iterations=2)

    return goal_mask.copy()


def generate_goal_contours(mask):
    goal_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
    if len(goal_contours) > 0:
        for contour in goal_contours:
            if not cv2.contourArea(contour) > 1000:
                goal_contours.remove(contour)

    return goal_contours


def generate_goal_image(goal_contours, index):
    if len(goal_contours) >= index:
        image = cv2.drawContours(blank_frame.copy(), goal_contours, index, 1, -1)

    return image


time.sleep(2.0)

frame = read_frame()
hsv = generate_blurred_hsv(frame)
blank_frame = np.zeros(frame.shape[0:2])
scores = np.zeros(2)

goal_mask = generate_goal_mask(hsv)
goal_contours = generate_goal_contours(goal_mask)

goal_0_image = generate_goal_image(goal_contours, 0)
goal_1_image = generate_goal_image(goal_contours, 0)

points = deque(maxlen=BUFFER_SIZE)


while True:
    frame = read_frame()
    hsv = generate_blurred_hsv(frame)
    ball_mask = generate_ball_mask(hsv)
    ball_contours = generate_ball_contours(ball_mask.copy())

    ball_center = None

    ball_image = cv2.drawContours(blank_frame.copy(), ball_contours, 0, 1, -1)

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

    if len(goal_contours) > 0:
        cv2.drawContours(frame, goal_contours, -1, (255, 0, 255), -1)

    points.appendleft(ball_center)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
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
