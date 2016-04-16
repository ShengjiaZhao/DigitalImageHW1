__author__ = 'shengjia'

import numpy as np
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
# from skimage import feature
from scipy import interpolate


class LUTBuilder:
    def __init__(self, ax, callback, display, channel='r'):
        self.ax = ax
        self.channel = channel
        self.callback = callback
        self.display = display
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
        self.control_points = {0: 0, 255: 255}
        self.lut = []
        for i in range(0, 256):
            self.lut.append(i)
        self.histogram = self.display.histogram(channel)
        self.plot()

    def __call__(self, event):
        if event.inaxes != self.ax.axes:
            return
        self.ax.cla()

        # Find the rounded control point and snap if there is a neighbor
        x_val = int(round(event.xdata))
        y_val = int(round(event.ydata))
        for x_snap in self.control_points:
            if abs(x_snap - x_val) < 5:
                x_val = x_snap
        if event.button == 3 and x_val != 0 and x_val != 255 and x_val in self.control_points:
            del self.control_points[x_val]
        elif event.button != 1:
            return
        else:
            self.control_points[x_val] = y_val

        # Sort the control points and interpolate
        sorted_x = sorted(self.control_points.keys())
        sorted_y = []
        for item in sorted_x:
            sorted_y.append(self.control_points[item])
        interp_func = interpolate.interp1d(sorted_x, sorted_y)
        for i in range(0, 256):
            self.lut[i] = int(round(interp_func(i)))
            if self.lut[i] < 0:
                self.lut[i] = 0
            if self.lut[i] > 255:
                self.lut[i] = 255
        self.callback()

    # Plot the current LUT and histogram
    def plot(self):
        self.ax.cla()
        self.ax.plot(range(0, 256), self.lut, c=self.channel)
        for control_point in self.control_points:
            self.ax.axvline(control_point, c='y')
        self.ax.bar(np.array(range(0, 32)) * 8, self.histogram, 1.5)
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.figure.canvas.draw()







