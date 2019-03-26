import configparser
import json
import math

import cv2
import numpy as np
from numpy.linalg import norm

from utils import ball_tracking as tracking

windowName = "HCI-KDD"
config = configparser.ConfigParser()
config.read("config.ini")
checkerboardWidth = int(config["checkerboard"]["width"])
checkerboardHeight = int(config["checkerboard"]["height"])
checkerboardStep = float(config["checkerboard"]["step"])
ASPECT_RATIO = ((checkerboardWidth - 1) * checkerboardStep, (checkerboardHeight - 1) * checkerboardStep)

camera = cv2.VideoCapture(int(config["camera"]["id"]))
camera.set(cv2.CAP_PROP_SETTINGS, 1)
# camera.set(cv2.CAP_PROP_FPS, config["algorithm"]["fps"])
# camera.set(cv2.CAP_PROP_AUTOFOCUS, 0) # doesn't work

cameraMatrix = None
newCameraMatrix = None
CamToCoord = None
distortionCoeffs = None
calibratedCheckerboard = False

magnification_pos = None
currentImage = []

magnifications = {}
slideMarkers = [
    {"color": [0, 0, 0], "position": [0, 0], "radius": 0},  # Mobile
    {"color": [0, 0, 0], "position": [0, 0], "radius": 0},  # Fixed 1
    {"color": [0, 0, 0], "position": [0, 0], "radius": 0}  # Fixed 2
]
origin = [0, 0]
referenceDistance = 1.0
correctionAngle = 0

correctedPoint = [0, 0]

try:
    if int(config["calib_checkerboard"]["use_file"]):
        with open("calibration.json", "r") as calibrationFile:
            calib = json.load(calibrationFile)
            cameraMatrix = np.array(calib["cameraMatrix"])
            newCameraMatrix = np.array(calib["newCameraMatrix"])
            CamToCoord = np.array(calib["transformationMatrix"])
            distortionCoeffs = np.array(calib["distortionCoeffs"])
            calibratedCheckerboard = True
except IOError as e:
    print("No configuration file found.")

mousePosition = [0, 0]
mousePositionColor = [0, 0, 0]
pixelColorAvgQueue = []
currentScene = int(config["calib_checkerboard"]["use_file"]) + int(config["general"]["skip_magnifications"])


def nextScene():
    global currentScene
    currentScene += 1


def calibrateCheckerboard():
    global calibratedCheckerboard, cameraMatrix, newCameraMatrix, CamToCoord, distortionCoeffs
    if not calibratedCheckerboard:
        frames = [camera.read()[1] for _ in range(10)]

        objPoints = []  # 3d point in real world space
        imgPoints = []  # 2d points in image plane
        imgSize = frames[0].shape[:2]

        # (0, 0, 0), (0, 0, 1), ..., (width-1, height-1, 1), (width-1, height-1, 2)
        coords3d = np.zeros((checkerboardWidth * checkerboardHeight, 3), np.float32)
        coords3d[:, :2] = np.mgrid[0:checkerboardWidth, 0:checkerboardHeight].T.reshape(-1, 2)

        for frame in frames:
            found, corners = findCheckerboard(frame)
            if found:
                objPoints.append(coords3d)
                imgPoints.append(corners)

        if len(objPoints) == 0:
            return False
        RMSReprojectionError, cameraMatrix, distortionCoeffs, rotationVectors, translationVectors = cv2.calibrateCamera(
            objPoints,
            imgPoints,
            imgSize[::-1],
            None, None)
        newCameraMatrix, ROI = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeffs, imgSize, 1, imgSize)

        if int(config["calib_checkerboard"]["correct_fisheye"]):
            correctedImgPoints, jacobian = cv2.projectPoints(objPoints[-1], rotationVectors[-1], translationVectors[-1],
                                                             newCameraMatrix, distortionCoeffs)
            correctedROI = np.float32([correctedImgPoints[0][0],
                                       correctedImgPoints[checkerboardWidth * checkerboardHeight - checkerboardWidth][
                                           0],
                                       correctedImgPoints[checkerboardWidth - 1][0],
                                       correctedImgPoints[checkerboardWidth * checkerboardHeight - 1][0]])
        else:
            correctedROI = np.float32([imgPoints[0][0][0],
                                       imgPoints[0][checkerboardWidth * checkerboardHeight - checkerboardWidth][0],
                                       imgPoints[0][checkerboardWidth - 1][0],
                                       imgPoints[0][checkerboardWidth * checkerboardHeight - 1][0]])

        baseROI = np.float32([[0, 0], [ASPECT_RATIO[1], 0], [0, ASPECT_RATIO[0]], [ASPECT_RATIO[1], ASPECT_RATIO[0]]])
        transformationMatrix = cv2.getPerspectiveTransform(correctedROI, baseROI)
        if RMSReprojectionError < 1:
            calibratedCheckerboard = True
            with open("calibration.json", "w") as saveFile:
                json.dump({
                    "cameraMatrix": cameraMatrix.tolist(),
                    "newCameraMatrix": newCameraMatrix.tolist(),
                    "transformationMatrix": transformationMatrix.tolist(),
                    "distortionCoeffs": distortionCoeffs.tolist()
                }, saveFile)
            nextScene()
    else:
        nextScene()


def findCheckerboard(image):
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(config["calib_checkerboard"]["criteria_iterations"]),
        float(config["calib_checkerboard"]["criteria_epsilon"]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    found, corners = cv2.findChessboardCorners(gray, (checkerboardWidth, checkerboardHeight), None)

    # If found, add object points, image points (after refining them)
    if found:
        searchWindow = int(config["calib_checkerboard"]["search_window"])
        # Refine corners positions
        preciseCorners = cv2.cornerSubPix(gray, corners, (searchWindow, searchWindow), (-1, -1),
                                          criteria)
        return found, preciseCorners
    return False, None


def showCheckerboard(image):
    found, corners = findCheckerboard(image)
    if found:
        # Draw and display the corners
        return cv2.drawChessboardCorners(image, (checkerboardWidth, checkerboardHeight), corners, found)
    return image


def calibrateMagnification():
    global magnifications, magnification_pos
    magnification = int(input("What is the value of the active magnification? "))
    if magnification > 0:
        if (magnification_pos is None):
            magnification_pos = mousePosition
        else:
            magnification_pos = [int(float(magnification_pos[0] + mousePosition[0]) / 2.0),
                                 int(float(magnification_pos[1] + mousePosition[1]) / 2.0)]
        magnifications[magnification] = mousePositionColor
    else:
        nextScene()


def calibrateSlideMobile():
    global slideMarkers
    slideMarkers[0]["color"] = mousePositionColor
    nextScene()


def calibrateSlideFixed1():
    global slideMarkers
    slideMarkers[1]["color"] = mousePositionColor
    slideMarkers[1]["position"] = mousePosition
    nextScene()


def getCircleMask(img, xc, yc, r):
    # size of the image
    H, W, _ = img.shape
    # x and y coordinates per every pixel of the image
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    # squared distance from the center of the circle
    d2 = (x - xc) ** 2 + (y - yc) ** 2
    # mask is True inside of the circle
    mask = d2 < r ** 2
    return mask


def calibrateSlideFixed2():
    global slideMarkers, currentScene, origin
    slideMarkers[2]["color"] = mousePositionColor
    slideMarkers[2]["position"] = mousePosition
    if updateReferences():
        origin = slideMarkers[0]["position"]
        nextScene()
    else:
        currentScene -= 2


def updateReferences():
    global slideMarkers, referenceDistance
    for i in range(3):
        markerFound, markerCenter, markerX, markerY, markerRadius = tracking.findMarker(currentImage,
                                                                                        slideMarkers[i]["color"],
                                                                                        None if (i == 0) else
                                                                                        slideMarkers[i]["position"],
                                                                                        None if (i == 0) else 35)
        if markerFound:
            slideMarkers[i]["position"] = markerCenter
            slideMarkers[i]["radius"] = markerRadius
        else:
            return False

    points = np.array([slideMarkers[1]["position"], slideMarkers[2]["position"]], dtype="float32")
    points = np.array([points])

    correctedPoints = cv2.perspectiveTransform(points, CamToCoord)
    ptA = list(correctedPoints[0][0])
    ptB = list(correctedPoints[0][1])
    distance = math.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)
    if distance > 0:
        referenceDistance = float(config["general"]["fixed_distance"]) / distance
        return True
    return False


def angle_between(a, b):
    arccosInput = np.dot(a, b) / norm(a) / norm(b)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    return math.acos(arccosInput)


def angle_toOrigin(target):
    v1 = (0, 1)
    v2 = (target[0] - origin[0], target[1] - origin[1])
    return angle_between(v1, v2);


def calibrateAngle():
    global correctionAngle
    markerFound, markerCenter, markerX, markerY, markerRadius = findMarker(currentImage, slideMarkers[0]["color"])
    if markerFound:
        correctionAngle = angle_toOrigin(markerCenter)
        nextScene()


def convertPoint(point):
    moved = (point[0] - origin[0], point[1] - origin[1])
    rotated = rotate(moved, correctionAngle)
    scaled = [
        rotated[0] * referenceDistance, rotated[1] * referenceDistance
    ]
    return scaled


def rotateInOrign(o, p, a):
    ox, oy = o
    px, py = p

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy


def rotate(p, a):
    px, py = p

    qx = math.cos(a) * (px) - math.sin(a) * (py)
    qy = math.sin(a) * (px) + math.cos(a) * (py)
    return qx, qy


def trackMarker(image):
    global correctedPoint
    if updateReferences():
        trackedPoint = convertPoint(slideMarkers[0]["position"])
    return showStats(image)


def showStats(image):
    global slideMarkers

    foundMagnification = -1
    for magnification, color in magnifications.items():
        found, center, x, y, radius = findMarker(image, color, magnification_pos, 35)
        if found:
            cv2.circle(image, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            foundMagnification = magnification

    cv2.putText(image, "Magnification: " + str(foundMagnification), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(image, "Reference distance: " + str(referenceDistance), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0))
    cv2.putText(image, "Correction angle: " + str(math.degrees(correctionAngle)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0))
    cv2.putText(image, "Origin: " + str(origin), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(image, "Position: " + str(slideMarkers[0]["position"]), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0))
    cv2.putText(image, "Corrected position: " + str(correctedPoint), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    for i in range(3):
        cv2.circle(image, tuple(slideMarkers[i]["position"]), int(slideMarkers[i]["radius"]), (255, 255, 255), 1)
        cv2.circle(image, tuple(slideMarkers[i]["position"]), 3, (0, 0, 0), -1)

    cv2.circle(image, tuple(origin), 2, (0, 0, 255), -1)
    zeroOne = rotateInOrign(origin, [origin[0] + 0, origin[1] + 50], correctionAngle)
    oneZero = rotateInOrign(origin, [origin[0] + 50, origin[1] + 0], correctionAngle)
    cv2.line(image, tuple(origin), (int(zeroOne[0]), int(zeroOne[1])), (0, 0, 255))
    cv2.line(image, tuple(origin), (int(oneZero[0]), int(oneZero[1])), (0, 255, 0))
    return image


def lastScene():
    a = 1


scenario = [
    {
        "text": "Place the checkerboard on the same table the microscope will be.\n"
                "When the board is detected, click to calibrate.",
        "image": showCheckerboard,
        "input": calibrateCheckerboard
    },
    {
        "text": "Set the microscope on the lowest magnification.\n"
                "Click on the tracking marker of the magnification visible on camera.\n"
                "Switch to the next magnification.\n"
                "Repeat until all magnifications have been calibrated.\n"
                "Then type -1 when asked for the value.",
        "image": showStats,
        "input": calibrateMagnification
    },
    {
        "text": "Click on the mobile tracking markers of the slide.",
        "image": showStats,
        "input": calibrateSlideMobile
    },
    {
        "text": "Click on the first fixed tracking marker of the slide.",
        "image": showStats,
        "input": calibrateSlideFixed1
    },
    {
        "text": "Click on the second fixed tracking marker of the slide.",
        "image": showStats,
        "input": calibrateSlideFixed2
    },
    {
        "text": "Move the mobile marker only on the x axis at least 1cm.",
        "image": showStats,
        "input": calibrateAngle
    },
    {
        "text": "Calibration complete.\n"
                "Click anywhere to exit.",
        "image": trackMarker,
        "input": lastScene
    },
]


def mouseCallback(event, x, y, flags, params):
    global scenario, currentScene, mousePosition, mousePositionColor, pixelColorAvgQueue

    if event == cv2.EVENT_MOUSEMOVE:
        mousePosition = [x, y]
        pixelColorAvgQueue = []

    if event == cv2.EVENT_LBUTTONDOWN:
        scenario[currentScene]["input"]()


while True:
    ret, img = camera.read()
    if ret:
        currentImage = img

        if calibratedCheckerboard:
            a = 1  # img = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, newCameraMatrix) # This breaks the distance calculations
            # img = cv2.warpPerspective(img, transformationMatrix, ASPECT_RATIO)

        # Calculate the pixel color where the mouse cursor is
        # Average out the color over some frames
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if 0 <= mousePosition[0] < hsv.shape[1] and 0 <= mousePosition[1] < hsv.shape[0]:
            pixelColor = list(hsv[mousePosition[1], mousePosition[0]])
            if len(pixelColorAvgQueue) >= int(config["general"]["average_color"]):
                mousePositionColor = [
                    int(sum([h for [h, s, v] in pixelColorAvgQueue]) / len(pixelColorAvgQueue)),
                    int(sum([s for [h, s, v] in pixelColorAvgQueue]) / len(pixelColorAvgQueue)),
                    int(sum([v for [h, s, v] in pixelColorAvgQueue]) / len(pixelColorAvgQueue))
                ]
                pixelColorAvgQueue[0] = pixelColor
            else:
                mousePositionColor = pixelColor
                pixelColorAvgQueue.insert(0, pixelColor)

        cv2.setMouseCallback(windowName, mouseCallback)
        if scenario[currentScene]["image"]:
            img = scenario[currentScene]["image"](img)

        # Write instructions
        instructions = scenario[currentScene]["text"].split("\n")
        fontSize = 13
        borderSize = int(fontSize * 1.35 * (len(instructions)))
        img = cv2.copyMakeBorder(img, 0, borderSize, 0, 0, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
        for i in range(len(instructions)):
            cv2.putText(img, instructions[i], (2, img.shape[0] - borderSize + int((i + 0.7) * (fontSize * 1.35))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontSize / 30,
                        (0, 0, 0))

        cv2.imshow(windowName, img)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()
