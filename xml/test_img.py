import numpy as np
import cv2


class Static_detection:
    def __init__(self):
        self.my_Cascade = cv2.CascadeClassifier('cascade.xml')
        self.img = None

    def detect(self, draw_box=False):
        if self.img is not None:
            result = []
            aim = self.my_Cascade.detectMultiScale(self.img,
                                                   scaleFactor=1.1,
                                                   minNeighbors=8,
                                                   minSize=(28, 28))
            if draw_box:
                for (x, y, w, h) in aim:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    result.append([x, y, w, h])
                result = np.array(result)
                cv2.imshow('detection', self.img)
                cv2.waitKey(0)
                print('locations are: ', result)
                return result
            else:
                for (x, y, w, h) in aim:
                    result.append([x, y, w, h])
                result = np.array(result)
                return result


if __name__ == '__main__':
    eagle_eye = Static_detection()
    eagle_eye.img = cv2.imread('test.jpg', 1)
    eagle_eye.img = cv2.resize(eagle_eye.img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_AREA)
    eagle_eye.detect(draw_box=True)
