__author__ = 'shengjia'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from image import Image
import numpy as np
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
# from skimage import feature
from scipy import interpolate
from merge import LUTBuilder
import tkFileDialog as fd
import random


class UIHandler(object):
    def __init__(self):
        fig = plt.figure()
        self.display_ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])
        self.image = None
        axopen = plt.axes([0.2, 0.05, 0.1, 0.075])
        bopen = Button(axopen, 'Open')
        bopen.on_clicked(self.open)
        axsave = plt.axes([0.31, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(self.save)
        axlut = plt.axes([0.42, 0.05, 0.1, 0.075])
        blut = Button(axlut, 'LUT')
        blut.on_clicked(self.lut)
        axseg = plt.axes([0.53, 0.05, 0.1, 0.075])
        bseg = Button(axseg, 'Seg')
        bseg.on_clicked(self.seg)
        axsegadj = plt.axes([0.74, 0.05, 0.2, 0.03])
        self.ssegadj = Slider(axsegadj, 'Size', 1, 15, valinit=0)
        self.cid = self.display_ax.figure.canvas.mpl_connect('button_press_event', self)
        plt.show()

    # This will not return and start the event handling process
    def start(self):
        plt.show()

    def open(self, event):
        fname = fd.askopenfilename(initialdir='.')
        opened_image = misc.imread(fname)
        opened_image = misc.imresize(opened_image, 1024 / float(opened_image.shape[1]))
        print(opened_image.shape)
        self.image = Image(self.display_ax, opened_image)
        self.image.plot()

    def save(self, event):
        fname = fd.asksaveasfilename(initialdir='.')
        misc.imsave(fname, self.image.processed_image())

    def lut(self, event):
        lut_fig = plt.figure()
        ax_r = lut_fig.add_subplot(221)
        ax_r.set_title('R')
        ax_g = lut_fig.add_subplot(222)
        ax_g.set_title('G')
        ax_b = lut_fig.add_subplot(223)
        ax_b.set_title('B')
        ax_k = lut_fig.add_subplot(224)
        ax_k.set_title('all')

        def update_callback():
            self.image.apply_lut(lut_k.lut, lut_r.lut, lut_g.lut, lut_b.lut)
            lut_k.histogram = self.image.histogram('k')
            lut_r.histogram = self.image.histogram('r')
            lut_g.histogram = self.image.histogram('g')
            lut_b.histogram = self.image.histogram('b')
            lut_k.plot()
            lut_r.plot()
            lut_g.plot()
            lut_b.plot()
            self.image.plot()

        lut_r = LUTBuilder(ax_r, update_callback, self.image, 'r')
        lut_g = LUTBuilder(ax_g, update_callback, self.image, 'g')
        lut_b = LUTBuilder(ax_b, update_callback, self.image, 'b')
        lut_k = LUTBuilder(ax_k, update_callback, self.image, 'k')

        plt.show()

    def seg(self, event):
        self.image.segmentation(2 ** self.ssegadj.val)
        self.image.plot()

    def __call__(self, event):
        if event.inaxes != self.display_ax.axes:
            return
        x_val = int(round(event.xdata))
        y_val = int(round(event.ydata))
        self.image.mark_as_background(x_val, y_val)

if __name__ == '__main__':
    ui = UIHandler()
    ui.start()

