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

    def scale_matrix(self, x_scale, y_scale):
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
        x, y = 0, 0
        for k in range(len(Ps)):
            x += np.math.factorial(n) / np.math.factorial(n - k) / np.math.factorial(k) * t ** k * (1 - t) ** (n - k) * Ps[k][0]
            y += np.math.factorial(n) / np.math.factorial(n - k) / np.math.factorial(k) * t ** k * (1 - t) ** (n - k) * Ps[k][1]
        pts.append([x, y])
        return np.array(pts)

    def flood(self, img, point, flood_color):
        start_cell_color = img[point[0]][point[1]][:3].copy()
        img[point[0]][point[1]][:3] = flood_color
        points = [point]
        while len(points) > 0:
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

    def draw_dancing_loop(self, w, h, loop_scale):
        loop_fps = 50

        frames = []
        fig = plt.figure()
        center = np.array([w / 2, h / 2], dtype=np.float32)
        '''
        The first animation part
        '''
        squish_seconds = 3
        stretch_seconds = 3
        time_passed = 0

        M = loop_scale
        loop_width, loop_height = M, M
        delta_x1 = 0.5 * M / (loop_fps*stretch_seconds)
        delta_y1 = loop_height / (loop_fps*squish_seconds)
        while time_passed <= max(squish_seconds, stretch_seconds):
            time_passed += 1/loop_fps
            img = self.create_empty_image(w, h, np.array([255, 255, 255], np.uint8))
            line1Ps = np.array([center + np.array([0, 0], dtype=np.float32),
                        center + np.array([loop_width, loop_height], dtype=np.float32),
                        center + np.array([loop_width, -loop_height], dtype=np.float32),
                        center + np.array([0, 0], dtype=np.float32)])
            line2Ps = self.cartesian(self.scale_matrix(-1, 1) @ self.projective(line1Ps.T)).T
            self.draw_zigzag(img, self.ultimate_bezier_curve(40, line1Ps), np.array([255, 0, 0]))
            self.draw_zigzag(img, self.ultimate_bezier_curve(40, line2Ps), np.array([255, 0, 0]))
            img = np.transpose(img, axes=(1, 0, 2))
            img = np.flipud(img)
            im = plt.imshow(img)
            frames.append([im])
            if time_passed <= stretch_seconds:
                loop_width += delta_x1
            if loop_height > 0 and time_passed <= squish_seconds:
                loop_height -= delta_y1
                if loop_height < 0:
                    loop_height = 0.5

        '''
        The second animation part
        '''
        squish_seconds = 2
        stretch_seconds = 2
        time_passed = 0

        M = loop_scale
        T = loop_scale - 0 # 0 represents the Y coord for the rightmost point of a loop
        delta_x2 = 0.5 * M / (loop_fps*squish_seconds)
        delta_y2 = T / (loop_fps*stretch_seconds)
        while time_passed <= max(squish_seconds, stretch_seconds):
            time_passed += 1/loop_fps
            img = self.create_empty_image(w, h, np.array([255, 255, 255], np.uint8))
            line1Ps = np.array([center + np.array([0, 0], dtype=np.float32),
                        center + np.array([loop_width, loop_height], dtype=np.float32),
                        center + np.array([loop_width, -loop_height], dtype=np.float32),
                        center + np.array([0, 0], dtype=np.float32)])
            line2Ps = self.cartesian(self.scale_matrix(-1, 1) @ self.projective(line1Ps.T)).T
            self.draw_zigzag(img, self.ultimate_bezier_curve(40, line1Ps), np.array([255, 0, 0]))
            self.draw_zigzag(img, self.ultimate_bezier_curve(40, line2Ps), np.array([255, 0, 0]))
            img = np.transpose(img, axes=(1, 0, 2))
            img = np.flipud(img)
            im = plt.imshow(img)
            frames.append([im])
            if time_passed <= squish_seconds:
                loop_width -= delta_x2
            if time_passed <= stretch_seconds:
                loop_height += delta_y2

        ani = animation.ArtistAnimation(fig, frames, interval=1000/loop_fps, blit=True, repeat_delay=0)
        writer = PillowWriter(fps=loop_fps)
        ani.save("Data\\loopy.gif", writer=writer)

        #plt.show()

splinin = AnimationSplinin()
splinin.draw_dancing_loop(200, 200, 50)
