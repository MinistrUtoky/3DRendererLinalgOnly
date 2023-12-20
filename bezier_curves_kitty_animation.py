
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
plt.rcParams['animation.ffmpeg_path'] = 'bin\\ffmpeg.exe'

class AnimationSplinin:
    def projective(self, x):
        x = np.concatenate([x, np.ones((1, x.shape[1]))], axis=0)
        return x

    def cartesian(self, x):
        return x[:-1] / x[-1]

    def shift_matrix(self, shift_vector):
        return np.array([[1, 0, shift_vector[0]],
                         [0, 1, shift_vector[1]],
                         [0, 0, 1]])

    def rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])

    def scale_matrix(self, size_coef):
        return np.array([[size_coef, 0, 0],
                         [0, size_coef, 0],
                         [0, 0, 1]])

    def scale_matrix_xy(self, x_scale, y_scale):
        return np.array([[x_scale, 0, 0],
                         [0, y_scale, 0],
                         [0, 0, 1]])

    def create_empty_image(self, height, width, background_color):
        img = np.zeros((height, width, 4), np.uint8)
        img[:, :, :3] = background_color
        img[:, :, 3] = 255
        return img

    def set_color(self, img, x, y, color):
        img[x, y, :3] = color

    def draw_line(self, img, point1, point2, color):
        x0, y0 = int(point1[0]), int(point1[1])
        x1, y1 = int(point2[0]), int(point2[1])

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
            self.set_color(img, int(x0), int(y0), color)
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

    def show_image(self, img):
        img = np.flipud(img)
        plt.figure()
        plt.imshow(img)
        plt.show()

    def create_bezier_curve_2(self, key_vertices_number, P0, P1, P2):
        pts = []
        key_vertices_number -= 1
        t = 0
        while t <= 1:
            x = P0[0]*(1-t)**2 + 2*P1[0]*(1-t)*t + P2[0]*t**2
            y = P0[1]*(1-t)**2 + 2*P1[1]*(1-t)*t + P2[1]*t**2
            pts.append([x, y])
            t += 1/key_vertices_number
        return pts

    def create_bezier_curve_3(self, key_vertices_number, P0, P1, P2, P3):
        pts = []
        key_vertices_number -= 1
        t = 0
        while t <= 1:
            x = P0[0]*(1-t)**3 + 3*P1[0]*t*(1-t)**2 + 3*P2[0]*(1-t)*t**2 + P3[0]*t**3
            y = P0[1]*(1-t)**3 + 3*P1[1]*t*(1-t)**2 + 3*P2[1]*(1-t)*t**2 + P3[1]*t**3
            pts.append([x, y])
            t += 1/key_vertices_number
        return pts

    def ultimate_bezier_curve(self, key_vertices_number, Ps):
        pts = []
        key_vertices_number -= 1
        Ps = np.array(Ps, dtype=np.float32)
        n = len(Ps) - 1
        t = 0
        while t <= 1:
            x, y = 0, 0
            for k in range(len(Ps)):
                x += np.math.factorial(n)/np.math.factorial(n-k)/np.math.factorial(k) * t**k * (1-t)**(n-k) * Ps[k][0]
                y += np.math.factorial(n)/np.math.factorial(n-k)/np.math.factorial(k) * t**k * (1-t)**(n-k) * Ps[k][1]
            pts.append([x, y])
            t += 1/key_vertices_number
        return np.array(pts)

    def flood_only_white(self, img, point, flood_color):
        start_cell_color = np.array([255,255,255])
        img[point[0]][point[1]][:3] = flood_color
        points = [point]
        max_depth = 50000
        while len(points) > 0 and max_depth > 0:
            p = points[0]
            if p[0] > 0 and p[0] < img.shape[0] - 1 and p[1] > 0 and p[1] < img.shape[1] - 1:
                if np.array_equal(img[p[0]+1][p[1]][:3], start_cell_color):
                    points.append([p[0]+1, p[1]])
                    img[p[0]+1][p[1]][:3] = flood_color
                if np.array_equal(img[p[0]][p[1]+1][:3], start_cell_color):
                    points.append([p[0], p[1]+1])
                    img[p[0]][p[1]+1][:3] = flood_color
                if np.array_equal(img[p[0] - 1][p[1]][:3], start_cell_color):
                    points.append([p[0] - 1, p[1]])
                    img[p[0]-1][p[1]][:3] = flood_color
                if np.array_equal(img[p[0]][p[1] - 1][:3], start_cell_color):
                    points.append([p[0], p[1] - 1])
                    img[p[0]][p[1]-1][:3] = flood_color
            points.remove(p)
            max_depth -= 1
        return img

    def flood(self, img, point, flood_color):
        start_cell_color = img[point[0]][point[1]][:3].copy()
        img[point[0]][point[1]][:3] = flood_color
        points = [point]
        max_depth = 50000
        while len(points) > 0 and max_depth > 0:
            p = points[0]
            if p[0] > 0 and p[0] < img.shape[0] - 1 and p[1] > 0 and p[1] < img.shape[1] - 1:
                if np.array_equal(img[p[0]+1][p[1]][:3], start_cell_color):
                    points.append([p[0]+1, p[1]])
                    img[p[0]+1][p[1]][:3] = flood_color
                if np.array_equal(img[p[0]][p[1]+1][:3], start_cell_color):
                    points.append([p[0], p[1]+1])
                    img[p[0]][p[1]+1][:3] = flood_color
                if np.array_equal(img[p[0] - 1][p[1]][:3], start_cell_color):
                    points.append([p[0] - 1, p[1]])
                    img[p[0]-1][p[1]][:3] = flood_color
                if np.array_equal(img[p[0]][p[1] - 1][:3], start_cell_color):
                    points.append([p[0], p[1] - 1])
                    img[p[0]][p[1]-1][:3] = flood_color
            points.remove(p)
            max_depth-=1
        return img

    def midpoint(self, p1, p2):
        return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

    def continous_bezier_curve(self, control_points, step):
        pts, control_pts = [], []
        for i in range(1, len(control_points)-1, 2):
            control_pts.append(self.midpoint(control_points[i - 1], control_points[i]))
            control_pts.append(control_points[i])
            control_pts.append(control_points[i+1])
            if i+2 < len(control_points) - 1:
                control_pts.append(self.midpoint(control_points[i+1], control_points[i+2]))
        for i in range(0, len(control_pts)-2, 4):
            a0 = control_pts[i]
            a1 = control_pts[i+1]
            a2 = control_pts[i+2]
            ps = [a0, a1, a2]
            if i + 3 <= len(control_pts) - 1:
                ps.append(control_pts[i+3])
            for p in self.ultimate_bezier_curve(1/step, ps):
                pts.append(p)
        return pts

    def draw_zigzag(self, img, points, color):
        prevXY = [points[0][0], points[0][1]]
        self.set_color(img, int(points[0][0]), int(points[0][1]), color)
        for i in range(1, len(points)):
            self.set_color(img, int(points[i][0]), int(points[i][1]), color)
            self.draw_line(img, [prevXY[0], prevXY[1]], [points[i][0], points[i][1]], color)
            prevXY = [points[i][0], points[i][1]]
        return img

    def create_tail(self):
        tail1 = np.array([np.array([-40, 0], dtype=np.float32),
                    np.array([0, 50], dtype=np.float32),
                    np.array([80, 20], dtype=np.float32),
                    np.array([80, 70], dtype=np.float32)])
        tail2 = np.array([np.array([-40, -5], dtype=np.float32),
                    np.array([0, 50], dtype=np.float32),
                    np.array([80, 20], dtype=np.float32),
                    np.array([80, 70], dtype=np.float32)])
        tail3 = np.array([np.array([26, 40], dtype=np.float32),
                    np.array([40, 55], dtype=np.float32),
                    np.array([55, 40], dtype=np.float32)])
        tail4 = np.array([np.array([41, 19], dtype=np.float32),
                    np.array([50, 20], dtype=np.float32),
                    np.array([55, 25], dtype=np.float32),
                    np.array([55, 40], dtype=np.float32)])
        tail1 = self.ultimate_bezier_curve(10, tail1)
        tail2 = self.ultimate_bezier_curve(10, tail2)
        tail3 = self.ultimate_bezier_curve(10, tail3)
        tail4 = self.ultimate_bezier_curve(10, tail4)
        tail1 = self.cartesian(self.rotation_matrix(np.math.pi / 10)
                               @ self.scale_matrix(1.3)
                               @ self.projective(tail1.T)).T
        tail2 = self.cartesian(self.shift_matrix(np.array([18, -12]))
                               @ self.rotation_matrix(np.math.pi / 10)
                               @ self.scale_matrix(1.3)
                               @ self.projective(tail2.T)).T
        tail3 = self.cartesian(self.shift_matrix(np.array([60, 53]))
                               @ self.rotation_matrix(np.math.pi / 10)
                               @ self.projective(tail3.T)).T
        tail4 = self.cartesian(self.shift_matrix(np.array([62, 55]))
                               @ self.rotation_matrix(np.math.pi * 18 / 128)
                               @ self.scale_matrix(0.99)
                               @ self.projective(tail4.T)).T
        tail = [tail1, tail2, tail3, tail4]
        the_tail = []
        for subtail in tail:
            the_tail.append(self.cartesian(self.shift_matrix(np.array([30, -2])) @ self.projective(subtail.T)).T)
        return the_tail

    def create_full_back_leg(self):
        thigh = np.array([np.array([-75, 15], dtype=np.float32),
                        np.array([-35, 25], dtype=np.float32),
                        np.array([-5, -30], dtype=np.float32),
                        np.array([100, 5], dtype=np.float32)])
        calf = np.array([np.array([-25, -25], dtype=np.float32),
                        np.array([-20, -25], dtype=np.float32),
                        np.array([15, -10], dtype=np.float32),
                        np.array([15, 20], dtype=np.float32)])
        hip = np.array([np.array([-40, -50], dtype=np.float32),
                        np.array([0, -25], dtype=np.float32),
                        np.array([-20, 15], dtype=np.float32),
                        np.array([25, 30], dtype=np.float32)])
        shin = np.array([np.array([-50, -30], dtype=np.float32),
                        np.array([-50, -15], dtype=np.float32),
                        np.array([35, -15], dtype=np.float32),
                        np.array([35, 50], dtype=np.float32)])
        paw = np.array([np.array([0, 21], dtype=np.float32),
                        np.array([0, 5], dtype=np.float32),
                        np.array([18, 18], dtype=np.float32),
                        np.array([5, 0], dtype=np.float32),
                        np.array([21, 0], dtype=np.float32)])
        thigh = self.ultimate_bezier_curve(10, thigh)
        calf = self.ultimate_bezier_curve(10, calf)
        hip = self.ultimate_bezier_curve(10, hip)
        shin = self.ultimate_bezier_curve(10, shin)
        paw = self.ultimate_bezier_curve(10, paw)

        thigh = self.cartesian(self.shift_matrix(np.array([55, -30]))
                               @ self.rotation_matrix(np.math.pi / 2 + np.math.pi / 11)
                               @ self.projective(thigh.T)).T
        calf = self.cartesian(self.shift_matrix(np.array([48, -127])) @ self.scale_matrix(1.4)
                              @ self.rotation_matrix(np.math.pi / 12)
                              @ self.projective(calf.T)).T
        hip = self.cartesian(self.shift_matrix(np.array([29, -58])) @ self.scale_matrix(0.7)
                             @ self.rotation_matrix(np.math.pi * 13 / 48)
                             @ self.projective(hip.T)).T
        shin = self.cartesian(self.shift_matrix(np.array([24, -130])) @ self.scale_matrix(0.7)
                              @ self.rotation_matrix(np.math.pi / 8)
                              @ self.projective(shin.T)).T
        paw = self.cartesian(self.shift_matrix(np.array([11, -179]))
                              @ self.rotation_matrix(np.math.pi * 18 / 100)
                              @ self.projective(paw.T)).T
        return [thigh, calf, hip, shin, paw]

    def create_full_front_leg(self):
        thigh = np.array([np.array([-90, -10], dtype=np.float32),
                        np.array([-25, -5], dtype=np.float32),
                        np.array([10, 15], dtype=np.float32),
                        np.array([142, 70], dtype=np.float32)])
        hip = np.array([np.array([-100, -10], dtype=np.float32),
                          np.array([20, 0], dtype=np.float32),
                          np.array([40, 40], dtype=np.float32),
                          np.array([-0, 35], dtype=np.float32)])
        calf = np.array([np.array([-33, -43], dtype=np.float32),
                        np.array([-5, -45], dtype=np.float32),
                        np.array([5, 5], dtype=np.float32),
                        np.array([18, 10] , dtype=np.float32)])
        shin = np.array([np.array([-50, -50], dtype=np.float32),
                        np.array([-50, -15], dtype=np.float32),
                        np.array([35, -15], dtype=np.float32),
                        np.array([35, 35], dtype=np.float32)])
        paw = np.array([np.array([0, 25], dtype=np.float32),
                        np.array([0, 5], dtype=np.float32),
                        np.array([15, 15], dtype=np.float32),
                        np.array([5, 0], dtype=np.float32),
                        np.array([30, -8], dtype=np.float32)])
        thigh = self.ultimate_bezier_curve(10, thigh)
        hip = self.ultimate_bezier_curve(10, hip)
        calf = self.ultimate_bezier_curve(10, calf)
        shin = self.ultimate_bezier_curve(10, shin)
        paw = self.ultimate_bezier_curve(10, paw)

        thigh = self.cartesian(self.shift_matrix(np.array([73, 108])) @ self.scale_matrix(0.7)
                              @ self.rotation_matrix(np.math.pi / 3)
                              @ self.projective(thigh.T)).T
        hip = self.cartesian(self.shift_matrix(np.array([45, 128])) @ self.scale_matrix(0.65)
                               @ self.rotation_matrix(np.math.pi / 3)
                               @ self.projective(hip.T)).T
        calf = self.cartesian(self.shift_matrix(np.array([35, 42]))
                              @ self.rotation_matrix(np.math.pi / 24)
                              @ self.projective(calf.T)).T
        shin = self.cartesian(self.shift_matrix(np.array([3, 48])) @ self.scale_matrix(0.7)
                              @ self.rotation_matrix(np.math.pi / 8)
                              @ self.projective(shin.T)).T
        paw = self.cartesian(self.shift_matrix(np.array([-5, -13]))
                              @ self.rotation_matrix(np.math.pi * 18 / 100)
                              @ self.projective(paw.T)).T
        return [thigh, hip, calf, shin, paw]

    def create_back_legs(self):
        legs = self.create_full_back_leg()
        thigh2 = np.array([np.array([-85, 15], dtype=np.float32),
                        np.array([-35, 25], dtype=np.float32),
                        np.array([-5, -30], dtype=np.float32),
                        np.array([70, 15], dtype=np.float32)])
        calf2 = np.array([np.array([-15, -23], dtype=np.float32),
                        np.array([-10, -25], dtype=np.float32),
                        np.array([15, -10], dtype=np.float32),
                        np.array([15, 20], dtype=np.float32)])
        thigh2 = self.ultimate_bezier_curve(10, thigh2)
        calf2 = self.ultimate_bezier_curve(10, calf2)
        thigh2 = self.cartesian(self.shift_matrix(np.array([10, -25])) @ self.scale_matrix(0.9)
                                @ self.rotation_matrix(np.math.pi / 2 - np.math.pi / 12)
                                @ self.projective(thigh2.T)).T
        calf2 = self.cartesian(self.shift_matrix(np.array([-37, -102])) @ self.scale_matrix(0.9)
                               @ self.rotation_matrix(-np.math.pi / 12)
                               @ self.projective(calf2.T)).T
        legs.append(thigh2); legs.append(calf2)
        return legs

    def create_head(self):
        ear2_1 = np.array([np.array([0, 35], dtype=np.float32),
                        np.array([4, 4], dtype=np.float32),
                        np.array([35, 0], dtype=np.float32)])
        ear2_2 = np.array([np.array([0, 30], dtype=np.float32),
                        np.array([20, 20], dtype=np.float32),
                        np.array([30, 0], dtype=np.float32)])
        ear1_1 = np.array([np.array([0, 30], dtype=np.float32),
                        np.array([20, 20], dtype=np.float32),
                        np.array([30, 0], dtype=np.float32)])
        ear1_2 = np.array([np.array([0, 35], dtype=np.float32),
                           np.array([4, 4], dtype=np.float32),
                           np.array([35, 0], dtype=np.float32)])
        nape = np.array([np.array([0, 35], dtype=np.float32),
                        np.array([30, 30], dtype=np.float32),
                        np.array([35, 0], dtype=np.float32)])
        snout = np.array([np.array([-30, 10], dtype=np.float32),
                        np.array([-5, 0], dtype=np.float32),
                        np.array([0, 50], dtype=np.float32),
                        np.array([35, -15], dtype=np.float32)])
        cheek = np.array([np.array([-33, -33], dtype=np.float32),
                          np.array([-20, 10], dtype=np.float32),
                          np.array([-10, -40], dtype=np.float32),
                          np.array([0, 10], dtype=np.float32),
                          np.array([10, -40], dtype=np.float32),
                          np.array([20, 5], dtype=np.float32)])
        r_upper_eye = np.array([np.array([-12, 0], dtype=np.float32),
                        np.array([-12, 12], dtype=np.float32),
                        np.array([12, 12], dtype=np.float32),
                        np.array([12, 0], dtype=np.float32)])
        r_lower_eye = np.array([np.array([-12, 0], dtype=np.float32),
                        np.array([-12, -12], dtype=np.float32),
                        np.array([12, -12], dtype=np.float32),
                        np.array([12, 0], dtype=np.float32)])
        l_upper_eye = np.array([np.array([-12, 0], dtype=np.float32),
                              np.array([-12, 12], dtype=np.float32),
                              np.array([12, 12], dtype=np.float32),
                              np.array([12, 0], dtype=np.float32)])
        l_lower_eye = np.array([np.array([-12, 0], dtype=np.float32),
                              np.array([-12, -12], dtype=np.float32),
                              np.array([12, -12], dtype=np.float32),
                              np.array([12, 0], dtype=np.float32)])

        ear2_1 = self.ultimate_bezier_curve(10, ear2_1)
        ear1_1 = self.ultimate_bezier_curve(10, ear1_1)
        ear2_2 = self.ultimate_bezier_curve(10, ear2_2)
        ear1_2 = self.ultimate_bezier_curve(10, ear1_2)
        nape = self.ultimate_bezier_curve(10, nape)
        snout = self.ultimate_bezier_curve(10, snout)
        cheek = self.ultimate_bezier_curve(10, cheek)
        r_upper_eye = self.ultimate_bezier_curve(20, r_upper_eye)
        r_lower_eye = self.ultimate_bezier_curve(20, r_lower_eye)
        l_upper_eye = self.ultimate_bezier_curve(20, l_upper_eye)
        l_lower_eye = self.ultimate_bezier_curve(20, l_lower_eye)

        ear2_1 = self.cartesian(self.shift_matrix(np.array([-116, 5]))
                              @ self.rotation_matrix(np.math.pi)
                              @ self.projective(ear2_1.T)).T
        ear1_1 = self.cartesian(self.shift_matrix(np.array([-56, -40]))
                               @ self.rotation_matrix(np.math.pi / 3 + np.math.pi/12)
                               @ self.projective(ear1_1.T)).T
        ear2_2 = self.cartesian(self.shift_matrix(np.array([-117, -18])) @ self.scale_matrix(1.1)
                                @ self.rotation_matrix(np.math.pi * 17 / 24)
                                @ self.projective(ear2_2.T)).T
        ear1_2 = self.cartesian(self.shift_matrix(np.array([-34, -30]))
                                @ self.rotation_matrix(np.math.pi * 3 / 4)
                                @ self.projective(ear1_2.T)).T
        nape = self.cartesian(self.shift_matrix(np.array([-105, -45]))
                              @ self.rotation_matrix(np.math.pi / 7)
                              @ self.projective(nape.T)).T
        snout = self.cartesian(self.shift_matrix(np.array([-119, -95])) @ self.scale_matrix(0.9)
                              @ self.rotation_matrix(np.math.pi * 11 / 12)
                              @ self.projective(snout.T)).T
        cheek = self.cartesian(self.shift_matrix(np.array([-159, -48]))
                               @ self.rotation_matrix(np.math.pi * 9 / 24)
                              @ self.projective(cheek.T)).T
        r_lower_eye = self.cartesian(self.shift_matrix(np.array([-135, -58]))
                                @ self.rotation_matrix(-np.math.pi / 12)
                                @ self.projective(r_lower_eye.T)).T
        l_lower_eye = self.cartesian(self.shift_matrix(np.array([-99, -69]))
                                @ self.rotation_matrix(-np.math.pi * 3/ 48)
                                @ self.projective(l_lower_eye.T)).T
        r_upper_eye = self.cartesian(self.shift_matrix(np.array([-135, -58]))
                                @ self.rotation_matrix(-np.math.pi / 12)
                                @ self.projective(r_upper_eye.T)).T
        l_upper_eye = self.cartesian(self.shift_matrix(np.array([-99, -69]))
                                @ self.rotation_matrix(-np.math.pi * 3 / 48)
                                @ self.projective(l_upper_eye.T)).T
        return [ear2_1, ear2_2, ear1_1, ear1_2, nape, snout, cheek,
                r_lower_eye, l_lower_eye, r_upper_eye, l_upper_eye]

    def create_torso(self):
        back = np.array([np.array([-80, -32], dtype=np.float32),
                         np.array([-35, -27], dtype=np.float32),
                         np.array([-15, 33], dtype=np.float32),
                         np.array([58, 34], dtype=np.float32)])
        tummy = np.array([np.array([-100, -45], dtype=np.float32),
                         np.array([-60, -25], dtype=np.float32),
                         np.array([20, -30], dtype=np.float32),
                         np.array([50, 35], dtype=np.float32)])
        back = self.ultimate_bezier_curve(10,
                    self.cartesian(self.shift_matrix(np.array([5, 0]))
                                   @ self.scale_matrix(1.1)
                                   @ self.rotation_matrix(np.math.pi/24)
                                   @ self.projective(back.T)).T)
        tummy = self.ultimate_bezier_curve(10,
                    self.cartesian(self.shift_matrix(np.array([0, -50]))
                                   @ self.rotation_matrix(np.math.pi/12)
                                   @ self.projective(tummy.T)).T)
        return [back, tummy]

    def create_pupils(self):
        left_pupil1 = np.array([np.array([-7, 0], dtype=np.float32),
                         np.array([-7, -3], dtype=np.float32),
                         np.array([7, -3], dtype=np.float32),
                         np.array([7, 0], dtype=np.float32)])
        left_pupil2 = np.array([np.array([-7, 0], dtype=np.float32),
                         np.array([-7, 3], dtype=np.float32),
                         np.array([7, 3], dtype=np.float32),
                         np.array([7, 0], dtype=np.float32)])
        right_pupil1 = np.array([np.array([-7, 0], dtype=np.float32),
                         np.array([-7, -3], dtype=np.float32),
                         np.array([7, -3], dtype=np.float32),
                         np.array([7, 0], dtype=np.float32)])
        right_pupil2 = np.array([np.array([-7, 0], dtype=np.float32),
                         np.array([-7, 3], dtype=np.float32),
                         np.array([7, 3], dtype=np.float32),
                         np.array([7, 0], dtype=np.float32)])

        left_pupil1 = self.ultimate_bezier_curve(20, left_pupil1)
        left_pupil2 = self.ultimate_bezier_curve(20, left_pupil2)
        right_pupil1 = self.ultimate_bezier_curve(20, right_pupil1)
        right_pupil2 = self.ultimate_bezier_curve(20, right_pupil2)

        left_pupil1 = self.cartesian(self.shift_matrix(np.array([-135, -58]))
                                     @ self.rotation_matrix(-np.math.pi * 7 / 12)
                                     @ self.projective(left_pupil1.T)).T
        right_pupil1 = self.cartesian(self.shift_matrix(np.array([-99, -69]))
                                     @ self.rotation_matrix(-np.math.pi * 27 / 48)
                                     @ self.projective(right_pupil1.T)).T
        left_pupil2 = self.cartesian(self.shift_matrix(np.array([-135, -58]))
                                     @ self.rotation_matrix(-np.math.pi * 7 / 12)
                                     @ self.projective(left_pupil2.T)).T
        right_pupil2 = self.cartesian(self.shift_matrix(np.array([-99, -69]))
                                     @ self.rotation_matrix(-np.math.pi * 27 / 48)
                                     @ self.projective(right_pupil2.T)).T
        return [left_pupil1, left_pupil2, right_pupil1, right_pupil2]

    def construct_kitty(self, tail, back_legs, front_leg, torso, head, pupils):
        kitty = dict()
        kitty["tail"] = tail
        kitty["torso"] = torso
        kitty["back_legs"] = back_legs
        front_leg1 = []
        front_leg2 = []
        for part in front_leg:
            front_leg1.append(self.cartesian(self.shift_matrix(np.array([-150, -249]))
                                   @ self.rotation_matrix(-np.math.pi / 18)
                                   @ self.projective(part.T)).T)
            front_leg2.append(self.cartesian(self.shift_matrix(np.array([-215, -205]))
                                   @ self.rotation_matrix(-np.math.pi / 7)
                                   @ self.projective(part.T)).T)
        kitty["front_leg1"] = front_leg1
        kitty["front_leg2"] = front_leg2
        for i in range(len(head)):
            head[i] = self.cartesian(self.shift_matrix(np.array([-5, 5]))
                                   @ self.projective(head[i].T)).T
        kitty["head"] = head
        for i in range(len(pupils)):
            pupils[i] = self.cartesian(self.shift_matrix(np.array([-5, 5]))
                                   @ self.projective(pupils[i].T)).T
        kitty["pupils"] = pupils
        return kitty

    def draw_stretching_kitty(self, w, h):
        tail = self.create_tail()
        back_legs = self.create_back_legs()
        front_leg = self.create_full_front_leg()
        head = self.create_head()
        torso = self.create_torso()
        pupils = self.create_pupils()

        center = np.array([w/2, h/2], dtype=np.float32)

        for i in range(len(tail)):
            tail[i] = self.cartesian(self.shift_matrix(np.array([30, 35]))
                            @ self.rotation_matrix(0)
                            @ self.projective(tail[i].T)).T

        kitty = self.construct_kitty(tail, back_legs, front_leg, torso, head, pupils)

        number_of_frames = 80
        frames = []
        fig = plt.figure()
        tail_rotation_frames = number_of_frames / 16
        min_angle = -np.math.pi / 90
        max_angle = np.math.pi / 24
        angle = min_angle
        d_angle = (max_angle - min_angle) / tail_rotation_frames

        movement_coef = 0.25
        movement_direction = np.array([-1, -1])
        movement_state_framelength = number_of_frames / 10
        staring_state_framelength = 0

        body_stretch_frames = number_of_frames / 2
        body_stretch_max_angle = np.math.pi / 15
        body_stretch_angle = 0
        body_stretch_d_angle = body_stretch_max_angle / body_stretch_frames
        legs_shift_coef = 80/number_of_frames
        legs_shift_direction = np.array([1.25, -0.75])
        legs_shift_state_framelength = number_of_frames / 2

        for i in range(number_of_frames):
            img = self.create_empty_image(w, h, np.array([255, 255, 255], np.uint8))
            if angle > max_angle:
                d_angle = -(max_angle - min_angle) / tail_rotation_frames
            elif angle < min_angle:
                d_angle = (max_angle - min_angle) / tail_rotation_frames
            angle += d_angle

            if body_stretch_angle > body_stretch_max_angle:
                body_stretch_d_angle = -body_stretch_max_angle / body_stretch_frames
            elif body_stretch_angle < 0:
                body_stretch_d_angle = body_stretch_max_angle / body_stretch_frames
            body_stretch_angle += body_stretch_d_angle

            for k in kitty.keys():
                for j in range(len(kitty[k])):
                    kitty[k][j] = self.cartesian(self.shift_matrix(center)
                                   @ self.projective(kitty[k][j].T)).T
                    self.draw_zigzag(img, kitty[k][j], np.array([0, 0, 0], np.uint8))
                    kitty[k][j] = self.cartesian(self.shift_matrix(-center)
                                   @ self.projective(kitty[k][j].T)).T

            if movement_state_framelength > 0:
                movement_state_framelength -= 1
                for i in range(len(kitty["pupils"])):
                    kitty["pupils"][i] = self.cartesian(self.shift_matrix(movement_coef*movement_direction)
                                                @ self.projective(kitty["pupils"][i].T)).T
            elif staring_state_framelength == 0:
                staring_state_framelength = number_of_frames * 9 / 20
            elif staring_state_framelength > 0:
                staring_state_framelength -= 1
                if movement_state_framelength == 0 and staring_state_framelength == 0:
                    movement_state_framelength = number_of_frames / 10
                    movement_direction = -movement_direction

            for i in range(len(kitty["tail"])):
                kitty["tail"][i] = self.cartesian(self.rotation_matrix(d_angle)
                                          @ self.projective(kitty["tail"][i].T)).T

            for i in range(len(kitty["torso"])):
                kitty["torso"][i] = self.cartesian(self.rotation_matrix(body_stretch_d_angle)
                                          @ self.projective(kitty["torso"][i].T)).T
            for i in range(len(kitty["front_leg1"])):
                kitty["front_leg1"][i] = self.cartesian(self.rotation_matrix(-body_stretch_d_angle)
                                          @ self.projective(kitty["front_leg1"][i].T)).T
            for i in range(len(kitty["front_leg2"])):
                kitty["front_leg2"][i] = self.cartesian(self.rotation_matrix(-body_stretch_d_angle)
                                          @ self.projective(kitty["front_leg2"][i].T)).T


            if legs_shift_state_framelength > 0:
                legs_shift_state_framelength -= 1
                for i in range(len(kitty["front_leg1"])):
                    kitty["front_leg1"][i] = self.cartesian(self.shift_matrix(0.8*legs_shift_coef*legs_shift_direction)
                                                @ self.projective(kitty["front_leg1"][i].T)).T
                for i in range(len(kitty["front_leg2"])):
                    kitty["front_leg2"][i] = self.cartesian(self.shift_matrix(0.8*legs_shift_coef*legs_shift_direction)
                                                @ self.projective(kitty["front_leg2"][i].T)).T
                for i in range(len(kitty["head"])):
                    kitty["head"][i] = self.cartesian(self.shift_matrix(0.25*legs_shift_coef*legs_shift_direction)
                                                @ self.projective(kitty["head"][i].T)).T
                for i in range(len(kitty["pupils"])):
                    kitty["pupils"][i] = self.cartesian(self.shift_matrix(0.25*legs_shift_coef*legs_shift_direction)
                                                @ self.projective(kitty["pupils"][i].T)).T
            elif legs_shift_state_framelength == 0:
                legs_shift_state_framelength = number_of_frames / 2
                legs_shift_direction = -legs_shift_direction

            right_eye_pos = self.cartesian(self.scale_matrix_xy(1,-1)
                            @ self.projective(kitty["pupils"][0].T)).T[0]
            left_eye_pos = self.cartesian(self.scale_matrix_xy(1,-1)
                            @ self.projective(kitty["pupils"][2].T)).T[0]

            right_eye_pos = center+np.flipud(right_eye_pos) + np.array([3, -8])
            left_eye_pos = center+np.flipud(left_eye_pos) + np.array([3, -8])
            img = np.transpose(img, axes=(1, 0, 2))
            img = np.flipud(img)
            self.flood_only_white(img, [325, 210], np.array([0, 0, 0]))
            self.flood_only_white(img, [352, 231], np.array([0, 0, 0]))
            self.flood_only_white(img, [359, 227], np.array([0, 0, 0]))
            self.flood_only_white(img, [365, 222], np.array([0, 0, 0]))
            self.flood_only_white(img, [370, 205], np.array([0, 0, 0]))
            self.flood_only_white(img, [370, 211], np.array([0, 0, 0]))
            self.flood_only_white(img, [371, 207], np.array([0, 0, 0]))
            self.flood_only_white(img, [int(right_eye_pos[0]), int(right_eye_pos[1])], np.array([255, 255, 0]))
            self.flood_only_white(img, [int(right_eye_pos[0])+3, int(right_eye_pos[1])+12], np.array([255, 255, 0]))
            self.flood_only_white(img, [int(left_eye_pos[0]), int(left_eye_pos[1])], np.array([255, 255, 0]))
            self.flood_only_white(img, [int(left_eye_pos[0])+3, int(left_eye_pos[1])+12], np.array([255, 255, 0]))
            self.flood_only_white(img, [int(right_eye_pos[0])+2, int(right_eye_pos[1])+7], np.array([0, 0, 0]))
            self.flood_only_white(img, [int(left_eye_pos[0])+2, int(left_eye_pos[1]+7)], np.array([0, 0, 0]))
            im = plt.imshow(img)
            frames.append([im])

        fs = []
        slowdown = 1
        for i in range(len(frames)):
            for s in range(slowdown):
                fs.append(frames[i].copy())

        ani = animation.ArtistAnimation(fig, fs, interval=40, blit=True, repeat_delay=0)
        writer = PillowWriter(fps=24)
        ani.save("Data\\kitty.gif", writer=writer)

        #plt.show()

splinin = AnimationSplinin()
splinin.draw_stretching_kitty(512, 512)
