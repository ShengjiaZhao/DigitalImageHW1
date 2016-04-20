__author__ = 'shengjia'


import numpy as np
from skimage import segmentation
from scipy import misc



class Image:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.new_image = self.image.copy()
        self.background_mask = np.ones(self.new_image.shape[0:2], dtype=np.int)
        self.segment_label = None
        self.background = []
        self.background_remove = False
        self.stitch_images = []

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
            mask = 1 - self.background_mask
            content_on_canvas = np.zeros(list(mask.shape) + [4], np.uint8)    # Prepare the canvas
            for stitch_image in self.stitch_images:
                content = stitch_image['content']
                x = stitch_image['x']
                y = stitch_image['y']
                size = stitch_image['size']
                print(x, y, size)
                resized = misc.imresize(content, size)                      # Resize the image

                # Compute the intersecting length between canvas and content
                x_copy_width = mask.shape[1] - stitch_image['x']
                if resized.shape[1] < x_copy_width:
                    x_copy_width = resized.shape[1]
                y_copy_width = mask.shape[0] - stitch_image['y']
                if resized.shape[0] < y_copy_width:
                    y_copy_width = resized.shape[0]
                print(x_copy_width, y_copy_width)
                # Copy content onto canvas and apply mask
                new_content = np.zeros(list(mask.shape) + [4], np.uint8)
                new_content[y:y+y_copy_width, x:x+x_copy_width, :] = resized[0:y_copy_width, 0:x_copy_width, :]
                content_on_canvas = self.merge(content_on_canvas, new_content)
            content_on_canvas[:, :, 3] = np.multiply(mask, content_on_canvas[:, :, 3])
            self.ax.imshow(content_on_canvas)
        self.ax.figure.canvas.draw()

    def render(self):
        if self.segment_label is None:
            return self.new_image
        else:
            mask = 1 - self.background_mask
            content_on_canvas = np.zeros(list(mask.shape) + [4], np.uint8)    # Prepare the canvas
            for stitch_image in self.stitch_images:
                content = stitch_image['content']
                x = stitch_image['x']
                y = stitch_image['y']
                size = stitch_image['size']
                print(x, y, size)
                resized = misc.imresize(content, size)                      # Resize the image

                # Compute the intersecting length between canvas and content
                x_copy_width = mask.shape[1] - stitch_image['x']
                if resized.shape[1] < x_copy_width:
                    x_copy_width = resized.shape[1]
                y_copy_width = mask.shape[0] - stitch_image['y']
                if resized.shape[0] < y_copy_width:
                    y_copy_width = resized.shape[0]
                print(x_copy_width, y_copy_width)
                # Copy content onto canvas and apply mask
                new_content = np.zeros(list(mask.shape) + [4], np.uint8)
                new_content[y:y+y_copy_width, x:x+x_copy_width, :] = resized[0:y_copy_width, 0:x_copy_width, :]
                content_on_canvas = self.merge(content_on_canvas, new_content)
            content_on_canvas[:, :, 3] = np.multiply(mask, content_on_canvas[:, :, 3])
            return self.merge(content_on_canvas, np.concatenate([self.new_image,
                                                                 np.ones(list(self.background_mask.shape) + [1]) * 255],
                                                                axis=2))

    @staticmethod
    def merge(image1, image2):
        alpha1 = image1[:, :, 3].astype(float)
        alpha2 = image2[:, :, 3].astype(float)
        weight1 = alpha1 / 255
        weight2 = np.minimum(alpha2 / 255, 1 - weight1)
        result = np.ndarray(image1.shape, np.float)
        result[:, :, 0] = np.multiply(image1[:, :, 0], weight1) + np.multiply(image2[:, :, 0], weight2)
        result[:, :, 1] = np.multiply(image1[:, :, 1], weight1) + np.multiply(image2[:, :, 1], weight2)
        result[:, :, 2] = np.multiply(image1[:, :, 2], weight1) + np.multiply(image2[:, :, 2], weight2)
        result[:, :, 3] = np.clip(alpha1 + alpha2, 0, 255)
        return result.astype(np.uint8)

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
        self.background_mask = np.ones(self.new_image.shape[0:2], dtype=np.int)
        return self.segment_label

    def mark_as_background(self, x, y):
        label = self.segment_label[y, x]
        if label in self.background:
            self.background.remove(label)
            self.fill(x, y, 1)
        else:
            self.background.append(self.segment_label[y, x])
            self.fill(x, y, 0)
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

    def save(self, fname):
        if self.background_remove:
            shape = list(self.image.shape)
            shape[2] = 4
            image_with_alpha = np.ndarray(shape)
            image_with_alpha[:, :, 0:3] = self.new_image
            image_with_alpha[:, :, 3] = self.background_mask * 255
            misc.imsave(fname, image_with_alpha)
            print(image_with_alpha[0, 0, 3])
        else:
            misc.imsave(fname, self.render())

    def add_stitch(self, x, y, content):
        self.stitch_images.append({'x': x, 'y': y, 'size': 1.0, 'content': content})
        self.plot()
        return len(self.stitch_images) - 1

    def update_stitch(self, id, new_x, new_y, new_size):
        if new_x is not None:
            self.stitch_images[id]['x'] = new_x
        if new_y is not None:
            self.stitch_images[id]['y'] = new_y
        if new_size is not None:
            self.stitch_images[id]['size'] = new_size
        self.plot()

from matplotlib import pyplot as plt
if __name__ == '__main__':
    image1 = misc.imread('stitch1.png')
    image1 = image1[0:1024, 0:1024, :]
    print(image1.shape)
    image2 = misc.imread('stitch2.png')
    image2 = image2[0:1024, 0:1024, :]
    print(image2.shape)

    plt.imshow(Image.merge(image2, image1))
    plt.show()