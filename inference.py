import numpy as np
import cv2


class Static_detection:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier('cascade.xml')

    def detect(self, img, draw_box=False):
        loc = []
        res = self.classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(28, 28))
        if draw_box:
            for (x, y, w, h) in res:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                loc.append([x, y, w, h])
            loc = np.array(loc)
            cv2.imshow('detection', img)
            cv2.waitKey(0)
            print('locations are: ', loc)
            return loc
        else:
            for (x, y, w, h) in res:
                loc.append([x, y, w, h])
            loc = np.array(loc)
            return loc


if __name__ == '__main__':
    eagle_eye = Static_detection()

    frame = cv2.imread('test.jpg', 1)
    # frame = cv2.resize(frame, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_AREA)
    eagle_eye.detect(frame, draw_box=True)
