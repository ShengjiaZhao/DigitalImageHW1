__author__ = 'shengjia'


import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc, ndimage
from scipy.optimize import minimize, rosen, rosen_der
from math import cos, sin, tan, sqrt
import time
from skimage import transform as sktransform

class Matcher:
    def __init__(self):
        self.img1 = misc.imread('part1.jpeg')          # queryImage
        self.img2 = misc.imread('part2.jpeg')          # trainImage
        # self.img1 = misc.imresize(self.img1, 0.4)
        # self.img2 = misc.imresize(self.img2, 0.4)
        cimg1 = cv2.Canny(self.img1, 10, 100)
        cimg2 = cv2.Canny(self.img2, 10, 100)


        # Initiate SIFT detector
        orb = cv2.ORB(nfeatures=8000)

        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = orb.detectAndCompute(cimg1, None)
        self.kp2, self.des2 = orb.detectAndCompute(cimg2, None)
        print(len(self.kp1))

        # create BFMatcher object
        # FLANN parameters
        # index_params = dict(algorithm=1, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary

        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # self.matches = flann.knnMatch(self.des1, self.des2, k=2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors and Sort them in the order of their distance.
        self.matches = bf.match(self.des1, self.des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        print(len(self.matches))

        self.optim_count = 0

    def optim_function(self, param_list):
        affine_matrix = np.array(param_list[:-1])
        affine_matrix.resize([2, 3])
        affine_matrix = np.concatenate([affine_matrix, np.array([[0.0, 0.0, 1.0]])], 0)
        intersection = param_list[-1]
        print(intersection, affine_matrix)
        new_pt_list = []
        for kp in self.kp2:
            pt = self.transform(affine_matrix, kp.pt)
            new_pt_list.append(pt)

        distances = []
        for match in self.matches:
            # print(float(self.kp1[match.queryIdx].pt[0]) / self.img1.shape[1], float(self.kp2[match.queryIdx].pt[0]) / self.img2.shape[1])
            if float(self.kp1[match.queryIdx].pt[0]) / self.img1.shape[1] > 1 - intersection and \
                    float(self.kp2[match.trainIdx].pt[0]) / self.img2.shape[1] < intersection:
                distances.append(float(abs(self.kp1[match.queryIdx].pt[0] - new_pt_list[match.trainIdx][0]) +
                                     abs(self.kp1[match.queryIdx].pt[1] - new_pt_list[match.trainIdx][1])))
        distances = sorted(distances)
        distance_sum = sum(distances) / len(distances)
        det_val = np.linalg.det(affine_matrix[0:2, 0:2])
        if det_val < 1.0:
            distance_sum += 1.0 / det_val - 1
        else:
            distance_sum += 2 ** (det_val - 1)
        # distance_sum += 20 * np.sum(np.abs(affine_matrix[0:2, 0:2] - np.array([[1, 0], [0, 1]])))
        print(distance_sum)

        self.optim_count += 1
        if self.optim_count % 20 == 0:
            self.image_transform(affine_matrix, intersection)
        return distance_sum

    def transform(self, affine_matrix, pt):
        pt = np.array(list(pt) + [1.0], np.float)
        return np.dot(affine_matrix, pt)[0:2]

    def image_transform(self, affine_matrix, intersection):
        canvas_shape = list(self.img1.shape)
        canvas_shape[1] *= 2
        canvas_shape2 = list(self.img2.shape)
        canvas_shape2[1] *= 2
        if canvas_shape[0] < canvas_shape2[0]:
            canvas_shape[0] = canvas_shape2[0]
        if canvas_shape[1] < canvas_shape2[1]:
            canvas_shape[1] = canvas_shape2[1]
        canvas = np.zeros(canvas_shape, np.uint8)
        canvas[:self.img2.shape[0], :self.img2.shape[1], :] = self.img2[:, :, :] / 2
        canvas = sktransform.warp(canvas, np.linalg.inv(affine_matrix), preserve_range=True)
        canvas[:self.img1.shape[0], :self.img1.shape[1], :] += self.img1[:, :, :] / 2
        ax = plt.gca()
        ax.cla()
        ax.imshow(canvas.astype(np.uint8))
        print_count = 0
        for match in self.matches:
            if float(self.kp1[match.queryIdx].pt[0]) / self.img1.shape[1] > 1 - intersection and \
                    float(self.kp2[match.trainIdx].pt[0]) / self.img2.shape[1] < intersection:
                # print(match.distance)
                pt1 = self.kp1[match.queryIdx].pt
                pt2 = self.transform(affine_matrix, self.kp2[match.trainIdx].pt)
                circle = plt.Circle(pt1, 10, color='y', fill=False)
                ax.add_artist(circle)
                circle = plt.Circle(pt2, 10, color='g', fill=False)
                ax.add_artist(circle)
                arrow = plt.Arrow(pt1[0], pt1[1], pt2[0]-pt1[0], pt2[1]-pt1[1], width=1, color='y')
                ax.add_artist(arrow)
                print_count += 1
                #if print_count > 50:
                 #   break
            # print(match.distance, self.des1[match.queryIdx], self.des2[match.trainIdx])
        plt.draw()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.show()
        time.sleep(0.2)

    def render(self, param_list):
        affine_matrix = np.array(param_list[:-1])
        affine_matrix.resize([2, 3])
        affine_matrix = np.concatenate([affine_matrix, np.array([[0.0, 0.0, 1.0]])], 0)

        canvas_shape = list(self.img1.shape)
        canvas_shape[1] *= 2
        canvas_shape2 = list(self.img2.shape)
        canvas_shape2[1] *= 2
        if canvas_shape[0] < canvas_shape2[0]:
            canvas_shape[0] = canvas_shape2[0]
        if canvas_shape[1] < canvas_shape2[1]:
            canvas_shape[1] = canvas_shape2[1]
        canvas = np.zeros(canvas_shape, np.uint8)
        canvas[:self.img2.shape[0], :self.img2.shape[1], :] = self.img2[:, :, :]
        canvas = sktransform.warp(canvas, np.linalg.inv(affine_matrix), preserve_range=True)
        left_interp = int(round(self.img1.shape[1] * 0.8))
        right_interp = self.img1.shape[1]
        canvas[:self.img1.shape[0], :left_interp, :] = self.img1[:, :left_interp, :]
        for column in range(left_interp, right_interp):
            ratio = float(column - left_interp) / (right_interp - left_interp)
            canvas[:self.img1.shape[0], column, :] = canvas[:self.img1.shape[0], column, :] * ratio + \
                                                     self.img1[:, column, :] * (1-ratio)
        return canvas.astype(np.uint8)

    def stitch(self):
        result = minimize(self.optim_function, [1.0, 0.0, 1500, 0.0, 1.0, 0.0, 0.3], method='nelder-mead', tol=1e-6)
        return self.render(result['x'])

if __name__ == '__main__':
    matcher = Matcher()

    matcher.optim_function([1, 0, 1050, 0, 1, 0, 0.3])

    result = matcher.stitch()
    misc.imsave('result.png', result)
    plt.cla()
    plt.imshow(result)
    print("Finished")
    plt.ioff()
    plt.show()
    # matcher.image_transform(np.array([[cos(0.5), -sin(0.5), 1000], [sin(0.5), cos(0.5), 10], [0, 0, 1.0]]))