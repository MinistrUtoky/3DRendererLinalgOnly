import math

import matplotlib.pyplot as plt
import numpy as np
import re
from ctypes import windll

class UtahTeapotVisualizer:
    def __init__(self):
        print("Teapot evaluator initiated")

    def teapot_readlines(self):
        print("Reading teapot file")
        f = open("Data\\teapot.obj", 'r')
        return f.readlines()
    def teapot_tuple(self):
        print("Retrieving teapot values")
        figures, vertices = [], []
        f = self.teapot_readlines()
        if f is not None:
            for line in f:
                vs = re.findall(r'v\s-?\d+\.\d+\s-?\d+\.\d+\s-?\d+\.\d+', line)
                fs = re.findall(r'f\s\d+\s\d+\s\d+', line)
                if (len(vs)>0):
                    vertices.append(vs[0].split())
                if (len(fs)>0):
                    figures.append(fs[0].split())
        return (vertices, figures)


    def visualize_teapot(self, teapot_tuple):
        print("Starting teapot visualization")
        screensize = [windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)]
        size_coef = int(self.size_coef(screensize, teapot_tuple[0]))

        img = np.zeros((int(max(screensize[0]/3, screensize[1]/3)),
                        int(max(screensize[0]/3, screensize[1]/3)), 3), dtype=np.float32)
        center = [img.shape[0]/2, img.shape[1]/2]
        print(img)
        print("Building lines using Bresenham algorithm")
        for figure in teapot_tuple[1]:
            point1 = teapot_tuple[0][int(figure[1])-1][1:]
            point2 = teapot_tuple[0][int(figure[2])-1][1:]
            point3 = teapot_tuple[0][int(figure[3])-1][1:]
            img = self.bresenham(img, point1, point2, size_coef)
            img = self.bresenham(img, point2, point3, size_coef)
            img = self.bresenham(img, point3, point1, size_coef)

        mn, mx = self.min_max_X_coord(teapot_tuple[0])
        img = self.hor_gradient_img(img, img.shape[1] - int(mx*size_coef + center[1]), img.shape[1] - int(mn*size_coef + center[0]))


        self.show_img(img)


    def bresenham(self, img, point1, point2, size_coef):
        center = np.array([img.shape[0]/2, img.shape[1]/2], dtype=np.float32)
        x0, y0 = float(point1[1]), float(point1[0])#img.shape[0]-int(float(point1[1])*size_coef + center[0]), -int(float(point1[0])*size_coef + center[1])
        x1, y1 = float(point2[1]), float(point2[0])#img.shape[0]-int(float(point2[1])*size_coef + center[0]), -int(float(point2[0])*size_coef + center[1])

        scale_to_canvas_size = np.array([[size_coef, 0,         0],
                                         [0,         size_coef, 0],
                                         [0,         0,         1]])
        rotate_to_be_horizontal = np.array([[np.cos(math.pi), -np.sin(math.pi), 0],
                                            [np.sin(math.pi), np.cos(math.pi),  0],
                                            [0,               0,                1]])
        shift_to_center = -np.array([[1, 0, center[0]],
                                     [0, 1, center[1]],
                                     [0, 0,        1]])
        p1 = [x0, y0]
        p2 = [x1, y1]
        v = np.array([p1, p2], dtype=np.float32).T

        v_project_coordinates = np.concatenate([v, np.ones((1, v.shape[1]))], axis=0)

        v_project_coordinates = scale_to_canvas_size @ v_project_coordinates
        v_project_coordinates = rotate_to_be_horizontal @ v_project_coordinates
        v_project_coordinates = shift_to_center @ v_project_coordinates

        v = v_project_coordinates[:-1] / v_project_coordinates[-1]

        v = v.T
        x0, y0 = int(v[0][0]), int(v[0][1])
        x1, y1 = int(v[1][0]), int(v[1][1])

        dx = abs(x1 - x0)
        if (x0 < x1):
            sx = 1
        else:
            sx = -1
        dy = -abs(y1 - y0)
        if (y0 < y1):
            sy = 1
        else:
            sy = -1

        error = dx + dy

        while True:
            img[x0, y0, 0] = 1
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * error
            if e2 >= dy:
                if x0 == x1:
                    break
                error = error + dy
                x0 = x0 + sx
            if e2 <= dx:
                if y0 == y1:
                    break
                error = error + dx
                y0 = y0 + sy
        return img

    def hor_gradient_img(self, img, mn, mx):
        print("Making gradient")
        for x in range(img.shape[1]):
            for y in range(mn, mx):
                if (img[x, y, 0] == 1):
                    img[x, y, 1] = (y-mn)/(mx-mn)
        return img

    def show_img(self, img):
        print("Showing teapot")
        plt.figure()
        plt.imshow(img, aspect='equal', origin='upper')
        plt.show()
        plt.imsave("Data\\sample_image.png", img)

    def size_coef(self, screensize, points):
        print("Retrieving coefficient for teapot resizing to 1/3 of the screen")
        max_coord = -1e+100
        min_coord = 1e+100
        for point in points:
            if (max(float(point[1]), float(point[2])) > max_coord):
                max_coord = max(float(point[1]), float(point[2]))
            if (min(float(point[1]), float(point[2])) < min_coord):
                min_coord = min(float(point[1]), float(point[2]))
        return screensize[0]/3/math.ceil(max_coord-min_coord)

    def min_max_X_coord(self, points):
        print("retrieving min and max X values for gradient")
        max_coord = -1e+100
        min_coord = 1e+100
        for point in points:
            if (float(point[1]) > max_coord):
                max_coord = float(point[1])
            if (float(point[1]) < min_coord):
                min_coord = float(point[1])
        return min_coord, max_coord

teapotVis = UtahTeapotVisualizer()
teapotVis.visualize_teapot(teapotVis.teapot_tuple())