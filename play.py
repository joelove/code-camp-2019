from collections import deque
from imutils.video import VideoStream

import numpy as np
import cv2
import imutils
import time
import json

import face_utility


BALL_COLOR_LOWER1 = (0, 155, 100)
BALL_COLOR_UPPER1 = (5, 255, 255)

BALL_COLOR_LOWER2 = (175, 155, 100)
BALL_COLOR_UPPER2 = (180, 255, 255)

BALL_CIRCLE_COLOR = (0, 255, 255)

GOAL_COLOR_LOWER = (25, 0, 180)
GOAL_COLOR_UPPER = (45, 100, 255)

GOAL_FILL_COLOR = (255, 0, 255)

PLAYER_NAME_COLOR = (50, 255, 100)
START_TEXT_COLOR = (0, 255, 0)
SCORE_TEXT_COLOR = (255, 255, 255)
SCORE_RECT_COLOR = (0, 0, 0)

BUFFER_SIZE = 64

MAX_SCORE = 5
GOAL_MIN_AREA = 700

def init_players():
    players = []
    while not len(players) == 2:
        frame = read_frame()
        draw_start_text(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        players = find_players()

    print(f'found {len(players)} players')
    return players


def init_dummy_players():
    print('using dummy players')
    return [
        { "name": "Foo" },
        { "name": "Bar" }
    ]


def create_player(face):
    identifier, distance = face
    name = identifier if (distance < 0.6) else 'Unknown'

    return { "name": name }


def find_players():
    faces = face_utility.identify_faces(frame)[:2]
    players = list(map(create_player, faces))

    return players


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

    return ball_mask.copy()


def detect_ball_contours(mask):
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


def detect_goal_contours(image):
    mask = generate_goal_mask(image)
    goal_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goal_contours = [contour for contour in goal_contours if cv2.contourArea(contour) > GOAL_MIN_AREA]
    print(f'found {len(goal_contours)} goals')

    return goal_contours


def get_goal_image(goal_contours, index):
    if len(goal_contours) > index:
        image = cv2.drawContours(blank_frame.copy(), goal_contours, index, 1, -1)
        return image


def draw_ball_circle(ball_contours):
    ball_center = None

    if len(ball_contours) > 0:
        c = max(ball_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), BALL_CIRCLE_COLOR, 2)
            cv2.circle(frame, ball_center, 5, BALL_CIRCLE_COLOR, -1)

    return ball_center


def draw_goals(frame, goal_contours):
    if len(goal_contours) > 0:
        cv2.drawContours(frame, goal_contours, -1, GOAL_FILL_COLOR, -1)


def draw_ball_tracking_points(frame, ball_tracking_points):
    for i in range(1, len(ball_tracking_points)):
        if ball_tracking_points[i - 1] is None or ball_tracking_points[i] is None:
            continue

        thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
        cv2.line(frame, ball_tracking_points[i - 1], ball_tracking_points[i], (0, 0, 255), thickness)


def draw_scores(frame, scores, players):
    if player_one_won(scores) or player_two_won(scores):
        player_index = 0 if player_one_won(scores) else 1
        player_text = f'{players[player_index]["name"]} won!'
        cv2.putText(frame, player_text, (130, 150), 0, 2, SCORE_TEXT_COLOR, 4)
    else:
        cv2.rectangle(frame, (180,0), (420, 60), SCORE_RECT_COLOR, -1)
        cv2.putText(frame, f'{int(scores[0])} - {int(scores[1])}', (200, 50), 0, 2, SCORE_TEXT_COLOR, 4)


def draw_player_names(frame, players):
    if len(players) > 0:
        cv2.putText(frame, players[0]["name"], (10, 30), 0, 1, PLAYER_NAME_COLOR, 4)
    if len(players) > 1:
        cv2.putText(frame, players[1]["name"], (500, 30), 0, 1, PLAYER_NAME_COLOR, 4)


def draw_start_text(frame):
     cv2.putText(frame, "Looking for players...", (10, 30), 0, 1, START_TEXT_COLOR, 4)


def player_one_won(scores):
    return scores[0] >= MAX_SCORE


def player_two_won(scores):
    return scores[1] >= MAX_SCORE


face_utility.create_faces_file()
vs = VideoStream(src=0).start()

time.sleep(2.0)

frame = read_frame()
hsv = generate_blurred_hsv(frame)
blank_frame = np.zeros(frame.shape[0:2])
ball_tracking_points = deque(maxlen=BUFFER_SIZE)
scores = np.zeros(2)

players = init_players()

goal_contours = detect_goal_contours(hsv)
goal_0_image = get_goal_image(goal_contours, 0)
goal_1_image = get_goal_image(goal_contours, 1)

is_in_goal = 0
frame_count = 1

while True:
    frame = read_frame()
    hsv = generate_blurred_hsv(frame)

    draw_player_names(frame, players)
    draw_goals(frame, goal_contours)
    draw_scores(frame, scores, players)

    ball_mask = generate_ball_mask(hsv)
    ball_contours = detect_ball_contours(ball_mask)
    ball_image = cv2.drawContours(blank_frame.copy(), ball_contours, 0, 1, -1)

    ball_center = draw_ball_circle(ball_contours)
    ball_tracking_points.appendleft(ball_center)
    draw_ball_tracking_points(frame, ball_tracking_points)

    goal_0_collision = np.logical_and(ball_image, goal_0_image).any()
    goal_1_collision = np.logical_and(ball_image, goal_1_image).any()

    if goal_0_collision and is_in_goal == 0:
        scores[0] += 1
        is_in_goal = 1
        print(f'detected goal for player {players[0]["name"]}')

    if goal_1_collision and is_in_goal == 0:
        scores[1] += 1
        is_in_goal = 1
        print(f'detected goal for player {players[1]["name"]}')

    if not goal_0_collision and not goal_1_collision:
        is_in_goal = 0

    cv2.imshow("Frame", frame)

    if (frame_count % 500 == 0):
        goal_contours = detect_goal_contours(hsv)
        goal_0_image = get_goal_image(goal_contours, 0)
        goal_1_image = get_goal_image(goal_contours, 1)
        frame_count = 1
    else:
        frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()

cv2.destroyAllWindows()
