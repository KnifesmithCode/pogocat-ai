import cv2
import numpy as np
import re as regex
import pytesseract

death = cv2.imread('./images/death.jpg')
playing_dd = cv2.imread('./images/playing_double_digits.jpg')
playing = cv2.imread('./images/playing.jpg')
start = cv2.imread('./images/start.jpg')

images = [death, playing_dd, playing, start]

img = start

# region Development helpers


def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)


def show_clickable_image():
    cv2.imshow('playing', img)
    cv2.setMouseCallback('playing', click_event)
# endregion

# region Score selection
height, width, _ = img.shape

score_roi = [[int((240 / 2732) * width), int((50 / 2048) * height)],
             [int((800 / 2732) * width), int((130 / 2048) * height)]]

score_image = img[score_roi[0][1]:score_roi[1]
                  [1], score_roi[0][0]:score_roi[1][0]]


def clean_score_image(score_img):
    score_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(score_img)
    score_img = s
    (thresh, score_img) = cv2.threshold(
        score_img, 30, 255, cv2.THRESH_BINARY_INV)
    return score_img


def read_score(cleaned_score_img):
    matches = regex.findall('([0-9.]+)m.*x([0-9])', pytesseract.image_to_string(
        cleaned_score_img, config=r'--oem 3 --psm 6'))
    if len(matches) > 0:
        return {'score': float(matches[0][0]), 'multiplier': int(matches[0][1])}
    else:
        return {'score': None, 'multiplier': None}


score_image = clean_score_image(score_image)
print(read_score(score_image))
# endregion

# region Ground isolation

# bgr.b: high amounts of contrast between sky and ground, but rocks may be an issue
#        sky is high b, ground is low b
# hsv.v: decent contrast, but not as high as others
#        sky is high v, ground is low v
# lab.b: decent contrast, but rocks may once again be an issue
#        sky is low b, gorund is high b
# hsv.s: high contrast between clouds and ground, but not great between ground and sky, and rocks are an issue
#        sky is low s, ground is high s


def isolate_ground(bgr):
    thresh = cv2.inRange(bgr, (50, 100, 120), (200, 255, 200))
    return cv2.erode(cv2.dilate(cv2.erode(thresh, None, iterations=2), None, iterations=5), None, iterations=3)


def find_ground_contour(ground):
    contours, heirarchy = cv2.findContours(
        ground, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    return contour


lined_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
contoured_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

i = 1
while (i * 100) < img.shape[1]:
    cv2.line(lined_image, (i * 100, 0), (i * 100, img.shape[0]), 255, 3)
    i = i + 1
cv2.drawContours(contoured_image, find_ground_contour(
    isolate_ground(img)), -1, 255, 3)

intersections = cv2.bitwise_and(lined_image, contoured_image)

ground_points = {}

(contours, heirarchy) = cv2.findContours(
    intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for c in contours:
    m = cv2.moments(c)
    try:
        x = round(((m["m10"] / m["m00"]) / 10)) * 10
        y = round(m["m01"] / m["m00"])
        if abs(img.shape[0] - y) > 10:
            if x in ground_points:
                # if y is lesser then we want it
                if y > ground_points[x]:
                    continue
            ground_points[x] = y
    except ZeroDivisionError:
        continue


def draw_ground_points(img):
    for x, y in ground_points.items():
        cv2.circle(img, (x, y), 5, (0, 0, 255), 3)

# miiiight not work in case of brown stuff :)
# endregion

cv2.waitKey()
cv2.destroyAllWindows()
