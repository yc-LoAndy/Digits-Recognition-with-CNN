import re
import os
import cv2 as cv
import numpy as np

BLACK = (0,0,0)
WHITE = (255,255,255)
OUTSIZE = 28

class ImgProcessor:
    def __init__(self, imgpath, crop: tuple=None):
        img = cv.imread(imgpath)
        if crop is not None:
            x,y,w,h = crop
            img = img[y:y+h, x:x+w]
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.img = img
        # Remove all old digits inside the dir the same as impath
        res = [
            f for f in os.listdir(os.path.dirname(imgpath)) if re.search(r"digit_[\d]+.jpg$", f)
        ]
        for f in res:
            os.remove(os.path.dirname(imgpath)+'/'+f)

    def convert(self, outdir: str, outsize=OUTSIZE):
        ''' Convert the digits in the images to outsize*outsize small images '''
        if not outdir.endswith("/"):
            outdir = outdir + "/"

        self.img = cv.blur(self.img, (1,1))
        canny = cv.Canny(self.img, 50, 100)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
        dilate = cv.dilate(canny, kernel)
        erode = cv.erode(dilate, kernel)
        full_contours, full_hierarchy = self.__give_clean_contours(erode)
        canvas = self.__fill_digits(full_contours, full_hierarchy)

        # Draw rectangles
        c = 0
        for i, contour in enumerate(full_contours):
            if full_hierarchy[0][i][3] == -1:
                x,y,w,h = cv.boundingRect(contour)
                # trim the rectangle to contain the whole digit
                x = x - 5 if x >= 5 else 0
                y = y - 5 if y >= 5 else 0
                w = w + 5*2 if w <= (canvas.shape[1]-5-x) else (canvas.shape[1]-x)
                h = h + 5*2 if h <= (canvas.shape[0]-5-y) else (canvas.shape[0]-y)
                # cv.rectangle(canvas, (x,y), (x+w,y+h), WHITE, 5)

                # Build a new image for each digit that complies with the input format of the model
                newh, neww = self.__new_h_w(h, w, outsize)
                digit = np.zeros((newh, neww))
                digit = cv.resize(canvas[y:y+h, x:x+w], (neww, newh), dst=digit)

                card = np.zeros((outsize, outsize))
                start_x = (outsize//2) - (neww//2)
                start_y = (outsize//2) - (newh//2)
                card[start_y:start_y+newh, start_x:start_x+neww] = digit
                card = cv.dilate(card, kernel)

                cv.imwrite(f"{outdir}digit_{c}.jpg", card)
                c += 1
        cv.imwrite("/home/yclo/pyproj/practice/scripts/pic/digits/contor.jpg", canvas)

    def __give_clean_contours(self, edges):
        full_contours, full_hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        full_contours_clean = []
        for i in range(len(full_contours)):
            if full_contours[i].shape[0] > 20:
                full_contours_clean.append(full_contours[i])
        return full_contours_clean, full_hierarchy

    def __fill_digits(self, contours, hierarchy):
        ''' Fill the digits in white. Contours are expected to be drawn with RETR_TREE. '''
        canvas = cv.drawContours(np.zeros(self.img.shape), contours, -1, WHITE, 1)
        canvas = canvas.astype(np.uint8)
        hierarchy = np.array(hierarchy).squeeze()
        pts = []
        for i, hier in enumerate(hierarchy):
            if hier[3] == -1:  # this is the external contour
                cont = contours[i].squeeze()
                pt = cont[0]
                pt = (pt[0], pt[1]+3)
                pts.append(pt)
                # cv.circle(canvas, pt, 3, WHITE, 1)

        h, w = canvas.shape[:2]
        m = np.zeros((h+2, w+2), np.uint8)
        for i in range(len(pts)):
            cv.floodFill(canvas, m, pts[i], WHITE, 0, 0)
        return canvas

    def __new_h_w(self, h, w, outsize):
        smaller = h if h < w else w
        larger = w if smaller == h else h
        ratio = smaller / larger

        new_larger = outsize - 3
        new_smaller = int(new_larger * ratio)
        if larger == h:
            return (new_larger, new_smaller)
        else:
            return (new_smaller, new_larger)
