__author__ = 'shengjia'


import numpy as np
from skimage import segmentation



class Image:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.new_image = self.image.copy()
        self.background_mask = np.zeros(self.new_image.shape[0:2], dtype=np.int)
        self.segment_label = None
        self.background = []

    def apply_lut(self, klut, rlut, glut, blut):
        klut = np.array(klut)
        rlut = np.array(rlut)
        glut = np.array(glut)
        blut = np.array(blut)
        self.new_image[:, :, 0] = rlut[klut[self.image[:, :, 0]]]
        self.new_image[:, :, 1] = glut[klut[self.image[:, :, 1]]]
        self.new_image[:, :, 2] = blut[klut[self.image[:, :, 2]]]

    def plot(self):
        self.ax.cla()
        if self.segment_label is None:
            self.ax.imshow(self.new_image)
        else:
            boundary_image = segmentation.mark_boundaries(self.new_image, self.segment_label)
            self.ax.imshow(boundary_image)
            self.ax.imshow(self.background_mask.astype(float), alpha=0.3)

        self.ax.figure.canvas.draw()

    # Return the histogram of a image channel (r, g, b, all(k))
    def histogram(self, channel):
        histolut = []
        for i in range(0, 256):
            histolut.append(i / 8)
        histolut = np.array(histolut)
        if channel == 'r':
            histoimg = histolut[self.new_image[:, :, 0]]
        elif channel == 'g':
            histoimg = histolut[self.new_image[:, :, 1]]
        elif channel == 'b':
            histoimg = histolut[self.new_image[:, :, 2]]
        else:
            histoimg = histolut[self.new_image]

        bins, edge = np.histogram(histoimg, 32)
        bins = bins * 256 / max(bins)
        return bins

    def processed_image(self):
        return self.new_image

    def segmentation(self, n_seg=100):
        self.segment_label = segmentation.slic(self.new_image, n_segments=int(round(n_seg)), compactness=20)
        print(np.min(self.segment_label), np.max(self.segment_label))
        self.background = []
        self.background_mask = np.zeros(self.new_image.shape[0:2], dtype=np.int)
        return self.segment_label

    def mark_as_background(self, x, y):
        label = self.segment_label[y, x]
        if label in self.background:
            self.background.remove(label)
            self.fill(x, y, 0)
        else:
            self.background.append(self.segment_label[y, x])
            self.fill(x, y, 1)
        self.plot()

    def fill(self, x, y, val=1):
        stack = [[x, y]]
        while len(stack) != 0:
            elem = stack.pop()
            x = elem[0]
            y = elem[1]
            cur_label = self.segment_label[y, x]
            if x > 0 and self.background_mask[y, x-1] != val and self.segment_label[y, x-1] == cur_label:
                stack.append([x-1, y])
                self.background_mask[y, x-1] = val
            if y > 0 and self.background_mask[y-1, x] != val and self.segment_label[y-1, x] == cur_label:
                stack.append([x, y-1])
                self.background_mask[y-1, x] = val
            if x < self.image.shape[1] - 1 and self.background_mask[y, x+1] != val and self.segment_label[y, x+1] == cur_label:
                stack.append([x+1, y])
                self.background_mask[y, x+1] = val
            if y < self.image.shape[0] - 1 and self.background_mask[y+1, x] != val and self.segment_label[y+1, x] == cur_label:
                stack.append([x, y+1])
                self.background_mask[y+1, x] = val

