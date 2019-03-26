import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ball_tracking import cameraSettings
from ball_tracking import findMarker


def dist2(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def fuse(points, d):
    ret = []
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            taken[i] = True
            count = 1
            point = points[i].copy()
            for j in range(i + 1, n):
                dist = dist2(points[i], points[j])
                if not taken[j] and dist < d:
                    taken[j] = True
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
            point[0] /= count
            point[1] /= count
            ret.append([point[0], point[1]])
    return ret


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.det(A) != 0:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return True, [x0, y0]
    return False, [0, 0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    res, point = intersection(line1, line2)
                    if (res):
                        intersections.append(point)

    return intersections


def mouseCallback(event, x, y, flags, params):
    global trackedPoints, currentImage, windowSize

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(currentImage, cv2.COLOR_BGR2HSV)
        if 0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]:
            for trackedPoint in trackedPoints:
                if trackedPoint["color"] is None:
                    trackedPoint["color"] = list(hsv[y, x])
                    break;


fuseRadius = 150
rows, cols = 4, 8
windowSize = (1280, 720)
unit = 1.0

calib = np.load('data/calib.npz')
height, width = calib['shape']
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(calib['mtx'], calib['dist'], (width, height), 1, (width, height))
mapx, mapy = cv2.initUndistortRectifyMap(calib['mtx'], calib['dist'], None, newCameraMtx, (width, height), cv2.CV_16SC2)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


CamToCoord = None
scaling = None
if (Path("data/recordedMatrix.npz").exists()):
    recordedPositions = np.load('data/recordedMatrix.npz')
    CamToCoord = recordedPositions['CamToCoord']
    scaling = recordedPositions['scaling']

cameraSettings(camera)

if CamToCoord is None:

    while True:
        ret, img = camera.read()
        if ret:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)

            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            segmented = segment_by_angle_kmeans(lines)
            isections = segmented_intersections(segmented)
            isections = fuse(isections, fuseRadius)

            isections = [pt for pt in isections if
                         (pt[1] < (roi[1] + roi[3] - 10)) and
                         (pt[0] < (roi[0] + roi[2] - 10)) and
                         (pt[1] > (roi[1] + 10)) and
                         (pt[0] > (roi[0] + 10))]
            isections = [pt[1] for pt in sorted([(pt[1], pt) for pt in isections])]
            for row in range(rows):
                begin, end = row * cols, min((row + 1) * cols, len(isections))
                isections[begin:end] = [pt[1] for pt in sorted([(pt[0], pt) for pt in isections[begin:end]])]

            count = 1;
            for isection in isections:
                cv2.circle(img, (int(isection[0]), int(isection[1])), 6, (200, 255, 200), -1)
                cv2.circle(img, (int(isection[0]), int(isection[1])), fuseRadius, (255, 0, 0), 1)
                cv2.putText(img, str(count), (int(isection[0]), int(isection[1])), cv2.FONT_HERSHEY_SIMPLEX, 7,
                            (0, 0, 0))
                count += 1

            img = cv2.resize(img, windowSize, interpolation=cv2.INTER_AREA)
            cdst = cv2.resize(cdst, windowSize, interpolation=cv2.INTER_AREA)
            cv2.imshow("Source", img)
            cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

            key = cv2.waitKey() & 0xFF
            if key == 27:
                break

    cv2.destroyAllWindows()

    scaling = float(height) / float(rows)
    objectPoints = [[y, x] for x in range(rows) for y in range(cols)]
    objectPoints = np.multiply(objectPoints, scaling)
    isections = np.float32(isections)
    isections = np.array(isections)
    isections = np.float32(isections)
    objectPoints = np.float32(objectPoints)
    objectPoints = np.array(objectPoints)
    objectPoints = np.float32(objectPoints)

    CamToCoord, status = cv2.findHomography(isections, objectPoints, method=0)

    np.savez('data/recordedMatrix.npz', CamToCoord=CamToCoord, scaling=scaling)

trackedPoints = [{"color": None, "position": None, "radius": 0},
                 {"color": None, "position": None, "radius": 0}]

windowName = "Calibrated"
cv2.imshow(windowName, np.zeros([100, 100, 3], dtype=np.uint8))

cv2.setMouseCallback(windowName, mouseCallback)

while True:
    ret, img = camera.read()
    if ret:
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        im_dst = cv2.warpPerspective(img, CamToCoord, (img.shape[1], img.shape[0]))

        for trackedPoint in trackedPoints:
            if trackedPoint["color"] is not None:
                res, trackedPoint["position"], trackedPoint["radius"] = findMarker(im_dst, trackedPoint["color"])
                if (res):
                    cv2.circle(im_dst, tuple(trackedPoint["position"]), 10, (0, 0, 255), -1)

        if (trackedPoints[0]["position"] is not None and trackedPoints[1]["position"] is not None):
            distance = dist2(trackedPoints[0]["position"], trackedPoints[1]["position"]) / scaling * unit;
        else:
            distance = None;

        currentImage = cv2.resize(im_dst[0:int((rows - 1) * scaling), 0:int((cols - 1) * scaling)],
                                  (int((cols - 1) * 100), int((rows - 1) * 100)), interpolation=cv2.INTER_AREA)

        if distance is not None:
            cv2.putText(currentImage, "Distance: " + str(distance), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow(windowName, currentImage)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
