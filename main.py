import configparser
import mousetracking
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from utils import ball_tracking as tracking
from utils import calibration

infoBackground = {"text": "", "pos_start": (20, 60), "pos_end": (300, 200), "color": (255, 255, 255), "alpha": 0.8}

infoCameraOrigin = {"text": "Origin Camera: ", "text_pos": (40, 80), "color": (0, 0, 0)}
infoCamera = {"text": "Tracker position: ", "text_pos": (40, 100), "color": (0, 0, 255)}
infoCameraOnGrid = {"text": "Tracker position on coordinate system: ", "text_pos": (40, 120), "color": (0, 0, 255)}

infoMouseOrigin = {"text": "Origin Mouse: ", "text_pos": (40, 140), "color": (0, 0, 0)}
infoMouse = {"text": "Mouse position: ", "text_pos": (40, 160), "color": (255, 0, 0)}
infoMouseOnGrid = {"text": "Mouse position on coordinate system: ", "text_pos": (40, 180), "color": (255, 0, 0)}

windowName = "Laser-Camera-Based-Measurement"
config = configparser.ConfigParser()
config.read("config.ini")

trackedCamPoint = {"color": [0, 0, 0], "position": None, "radius": 0}
trackedMousePoint = None
trackedMousePoint2 = None

windowSize = (1280, 720)

mouseDPI = 16400.0
mousePosition = [0, 0]

currentScene = 0
currentImage = []

recordedPositions = None
recordedCameraPos = []
recordedMousePos = []

CamToCoord = None

markerFound = False

origin = None

if not Path("data/calib.npz").exists():
    print("Please first calibrate with camera_calib.py!")
    exit(-1)

calib = np.load('data/calib.npz')
cam_height, cam_width = calib['shape']

# Calculate scaling from camera image size to window size
scaling = (float(windowSize[0]) / float(cam_width), float(windowSize[1]) / float(cam_height))

# Load recorded Position if they were saved
if Path("data/recordedPositions.npz").exists():
    recordedPositions = np.load('data/recordedPositions.npz')
    recordedCameraPos = recordedPositions['recordedCameraPos']
    recordedMousePos = recordedPositions['recordedMousePos']


def nextScene():
    global currentScene
    currentScene += 1


def calcError():
    errorSum = 0
    for index in range(recordedCameraPos.shape[0]):
        camPoint = recordedCameraPos[index]
        mousePoint = recordedMousePos[index]
        correctedCamPoint = CamToCoord.Transform(camPoint)
        errorSum += np.sqrt((mousePoint[0] - correctedCamPoint[0]) ** 2 + (mousePoint[1] - correctedCamPoint[1]) ** 2)
    errorSum /= recordedMousePos.shape[0]

    return errorSum


class MouseTracking(threading.Thread):
    def run(self):
        print("Started Mousetracking Thread")
        mousetracking.run()
        print("Stopped Mousetracking Thread")


def SacleToScreen(point):
    global scaling
    return int(float(point[0]) * scaling[0]), int(float(point[1]) * scaling[1])


def drawRectangleOpacity(image, pt1, pt2, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), 1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def drawPositions(image):
    global trackedCamPoint, CamToCoord, trackedMousePoint, origin

    if origin is not None:
        cv2.circle(image, SacleToScreen(origin), 5, infoCameraOrigin["color"], -1)

    if trackedCamPoint["position"] is not None:
        cv2.circle(image, SacleToScreen(trackedCamPoint["position"]), 5, infoCamera["color"], -1)

    if CamToCoord is not None:
        cv2.circle(image, tuple(CamToCoord.Transform_inv([0, 0])), 5, infoMouseOrigin["color"], -1)
        cv2.circle(image, SacleToScreen(CamToCoord.Transform_inv(trackedMousePoint)), 5,
                   infoMouseOnGrid["color"], -1)

        # Draw Grid from -50 to 50 mm
        rsize = 50
        for linePos in range(-rsize, rsize, 10):
            lineStart = CamToCoord.Transform_inv([-rsize, linePos])
            lineEnd = CamToCoord.Transform_inv([rsize, linePos])
            cv2.line(image, SacleToScreen(lineStart), SacleToScreen(lineEnd), (0, 0, 255))

            lineStart = CamToCoord.Transform_inv([linePos, -rsize])
            lineEnd = CamToCoord.Transform_inv([linePos, rsize])
            cv2.line(image, SacleToScreen(lineStart), SacleToScreen(lineEnd), (0, 0, 255))

        # Draw Recorded Mouse and Camera points to show the error after learning
        # for i in range(recordedMousePos.shape[0]):
        #     rmp = recordedMousePos[i]
        #     crmp = CamToCoord.Transform_inv(rmp)
        #     cv2.circle(image, (int(crmp[0]), int(crmp[1])), 3, (0, 0, 255), -1)
        #     rcp = recordedCameraPos[i]
        #     cv2.circle(image, (int(rcp[0]), int(rcp[1])), 3, (0, 255, 0), -1)
    else:
        # Draw Recorded CameraPoints to show a new point was recorded
        for rcp in recordedCameraPos:
            cv2.circle(image, SacleToScreen(rcp), 3, (0, 255, 0), -1)


def drawInfo(image):
    global trackedCamPoint, CamToCoord, trackedMousePoint

    image = drawRectangleOpacity(image, infoBackground["pos_start"], infoBackground["pos_end"],
                                 infoBackground["color"], infoBackground["alpha"])

    cv2.putText(image, infoMouse["text"] + str(trackedMousePoint), infoMouse["text_pos"], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                infoMouse["color"])

    if origin is not None:
        cv2.putText(image, infoCameraOrigin["text"] + str(origin), infoCameraOrigin["text_pos"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    infoCameraOrigin["color"])

    if trackedCamPoint["position"] is not None:
        cv2.putText(image, infoCamera["text"] + str(trackedCamPoint["position"]), infoCamera["text_pos"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    infoCamera["color"])

    if CamToCoord is not None:
        correctedMousePoint = CamToCoord.Transform_inv(trackedMousePoint);

        cv2.putText(image, infoMouseOrigin["text"] + str(CamToCoord.Transform_inv([0, 0])), infoMouseOrigin["text_pos"],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, infoMouseOrigin["color"])
        cv2.putText(image, infoMouseOnGrid["text"] + str(correctedMousePoint), infoMouseOnGrid["text_pos"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    infoMouseOnGrid["color"])

        if (trackedCamPoint["position"] is not None):
            cv2.putText(image, infoCameraOnGrid["text"] + str(
                CamToCoord.Transform(trackedCamPoint["position"])), infoCameraOnGrid["text_pos"],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, infoCameraOnGrid["color"])

    # Write instructions
    instructions = scenario[currentScene]["text"].split("\n")
    borderSize = int(13 * 1.35 * (len(instructions)))
    image = cv2.copyMakeBorder(image, 0, borderSize, 0, 0, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))

    for i in range(len(instructions)):
        cv2.putText(img, instructions[i], (2, img.shape[0] - borderSize + int((i + 0.7) * (13 * 1.35))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    13 / 30,
                    (0, 0, 0))
    return image


def trackMarker(image):
    global origin, trackedCamPoint, markerFound
    markerFound, markerCenter, markerRadius = tracking.findMarker(image,
                                                                  trackedCamPoint["color"],
                                                                  trackedCamPoint["position"],
                                                                  int(float(img.shape[0]) / 20.0))
    if markerFound:
        trackedCamPoint["position"] = markerCenter
        trackedCamPoint["radius"] = markerRadius
        if origin is None and markerFound:
            origin = trackedCamPoint["position"]


def recordPositions(image):
    global trackedMousePoint, trackedMousePoint2, trackedCamPoint, recordedMousePos, recordedCameraPos, recordedPositions
    if trackedMousePoint not in recordedMousePos and markerFound and recordedPositions is None \
            and trackedMousePoint == trackedMousePoint2:
        recordedMousePos.append(trackedMousePoint)
        recordedCameraPos.append([float(trackedCamPoint["position"][0]), float(trackedCamPoint["position"][1])])


def setTrackingColor():
    global trackedCamPoint, currentImage, mousePosition
    hsv = cv2.cvtColor(currentImage, cv2.COLOR_BGR2HSV)
    if 0 <= mousePosition[0] < hsv.shape[1] and 0 <= mousePosition[1] < hsv.shape[0]:
        trackedCamPoint["color"] = list(hsv[mousePosition[1], mousePosition[0]])
        print(trackedCamPoint["color"])
        mousetracking.resetToZero()
        nextScene()


def calibrateTransformationMatrix():
    global CamToCoord, recordedCameraPos, recordedMousePos
    recordedCameraPos = np.array(recordedCameraPos)
    recordedMousePos = np.array(recordedMousePos)

    # Saving recorded Mouse and Camera points
    # np.savez('data/recordedPositions.npz', recordedCameraPos=recordedCameraPos, recordedMousePos=recordedMousePos)

    CamToCoord = calibration.FitHomography(recordedCameraPos, recordedMousePos)
    print(calcError())

    nextScene()


def mouseCallback(event, x, y, flags, params):
    global scenario, currentScene, mousePosition, scaling
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePosition = [int(float(x) / scaling[0]), int(float(y) / scaling[1])]
        if scenario[currentScene]["click"]:
            scenario[currentScene]["click"]()


scenario = [
    {
        "text": "Click on the mobile tracking markers of the slide.",
        "work": None,
        "click": setTrackingColor,
        "draw": [drawPositions, drawInfo],
    },
    {
        "text": "Move the mouse until there are enough calibration points",
        "work": [trackMarker, recordPositions],
        "click": calibrateTransformationMatrix,
        "draw": [drawPositions, drawInfo],
    },
    {
        "text": "Calibration complete.\n"
                "Click anywhere to exit.",
        "work": [trackMarker],
        "draw": [drawPositions, drawInfo],
    },
]

# Init camera with the same size as the calibration
camera = cv2.VideoCapture(int(config["camera"]["id"]), cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# Calculate matrix for Camera distortion
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(calib['mtx'], calib['dist'], (cam_width, cam_height), 1,
                                                  (cam_width, cam_height))
mapx, mapy = cv2.initUndistortRectifyMap(calib['mtx'], calib['dist'], None, newCameraMtx, (cam_width, cam_height),
                                         cv2.CV_16SC2)

# Open empty window for following commands
cv2.imshow(windowName, np.zeros([windowSize[1], windowSize[0], 3], dtype=np.uint8))

# The C++ Program needs to have a window first
mouseTracking = MouseTracking()
mouseTracking.start()

# Mouse Callback needs to have a window to be set
cv2.setMouseCallback(windowName, mouseCallback)

# Camera settings cannot be opened before the window otherwise this would be the active one
camera.set(cv2.CAP_PROP_SETTINGS, 1)

counter = 0
fps_start = time.time()
while True:
    # To make sure both camera and mouse represent the same point
    # we have to check if the point itself was not moving
    #we can check this by taking two mousepoints
    trackedMousePoint = [float(mousetracking.getX()) * 2.54 * 10.0 / mouseDPI,
                         float(mousetracking.getY()) * 2.54 * 10.0 / mouseDPI]
    ret, img = camera.read()

    trackedMousePoint2 = [float(mousetracking.getX()) * 2.54 * 10.0 / mouseDPI,
                          float(mousetracking.getY()) * 2.54 * 10.0 / mouseDPI]

    if ret:
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        currentImage = img

        if scenario[currentScene]["work"]:
            for work in scenario[currentScene]["work"]:
                work(img)

        # Resize image for displaying on the screen
        img = cv2.resize(img, windowSize, interpolation=cv2.INTER_AREA)

        if scenario[currentScene]["draw"]:
            for work in scenario[currentScene]["draw"]:
                work(img)

        cv2.imshow(windowName, img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    counter += 1
    if (time.time() - fps_start) > 1:
        print("FPS: ", counter / (time.time() - fps_start))
        counter = 0
        fps_start = time.time()

camera.release()
cv2.destroyAllWindows()

mousetracking.stop()
mouseTracking.join()
