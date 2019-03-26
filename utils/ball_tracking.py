import cv2


def cameraSettings(camera, windowSize=(1280, 720)):
    camera.set(cv2.CAP_PROP_SETTINGS, 1)
    while True:

        ret, img = camera.read()
        if (ret):
            img = cv2.resize(img, windowSize, interpolation=cv2.INTER_AREA)
            cv2.imshow("Calibration", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


def colorRangeHSV(color):
    # H is between 0 and 180.
    h1, h2 = min(max(int(color[0]) - 10, 0), 180), min(max(int(color[0]) + 10, 0), 180)
    # S is between 0 and 255.
    s1, s2 = min(max(int(color[1]) - 50, 1), 253), min(max(int(color[1]) + 50, 1), 253)
    # V is between 0 and 255.
    v1, v2 = min(max(int(color[2]) - 50, 1), 253), min(max(int(color[2]) + 50, 1), 253)
    color1 = (h1, s1, v1)
    color2 = (h2, s2, v2)
    return min(color1, color2), max(color1, color2)


def findMarker(image, color, crop_pos=None, crop_radius=None):
    colorLower, colorUpper = colorRangeHSV(color)

    rectX = 0
    rectY = 0
    if ((crop_pos is not None) and (crop_radius is not None)):
        rectY = max(crop_pos[1] - crop_radius, 0)
        rectX = max(crop_pos[0] - crop_radius, 0)
        image = image[rectY:min((rectY + 2 * crop_radius), image.shape[0]),
                rectX:min((rectX + 2 * crop_radius), image.shape[1])]

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # cv2.imshow("Mask", mask)
    # cv2.waitKey()

    countours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(countours) > 0:
        # Find the largest contour in the mask and use it to compute the minimum enclosing circle and centroid
        maxContour = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(maxContour)

        # Only proceed if the radius meets a minimum size
        if radius > 3:
            M = cv2.moments(maxContour)
            center = (int(M["m10"] / M["m00"] + rectX), int(M["m01"] / M["m00"] + rectY))
            return True, center, radius

    return False, None, None
