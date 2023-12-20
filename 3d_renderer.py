import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter


class Graphics3D():
    __data = None
    __figure = None
    __texture = None

    #0. Все численные значения задаются параметрами в начале работы.
    #В ходе задания они могут поменяться! Обращаю внимание!
    def __init__(self, source: str, texture_source: str):
        self.__setup_default_parameters()

        #Предусмотреть учет параметров области вывода (левая нижняя точка, размеры) при задании матрицы.
        self.__read_file(source)
        self.__create_figure_tuple()
        self.__upload_texture(texture_source)
        #self.__draw_image()
        self.__draw_animation()

    def __setup_default_parameters(self):
        self.__setup_default_camera()
        self.__setup_default_filters()
        self.__setup_default_lighting()

    def __setup_default_camera(self):
        self.camera_position = np.array([2, 2, 2])
        self.camera_looks_at = np.array([-2, -2, 0])
        self.viewport_size = np.array([800, 800]) # viewport here eq whole graphics window
        #Orthographic projection parameters
        #self.FOV = 45 # таким образом, чтобы модель полностью попадала в VIEW
        #self.FOV %= 90
        self.z_near_plane = 0.1
        self.view_distance = 100
        self.z_far_plane = self.z_near_plane + self.view_distance
        viewport_scale = 40 # таким образом, чтобы модель полностью попадала в VIEW
        self.right = self.viewport_size[0]/viewport_scale
        self.left = -self.viewport_size[0]/viewport_scale
        self.top = self.viewport_size[1]/viewport_scale
        self.bottom = -self.viewport_size[1]/viewport_scale

    def __setup_default_filters(self):
        #Filter parameters
        self.z_buffer = np.ones((self.viewport_size[0], self.viewport_size[1]))

    def __setup_default_lighting(self):
        # Pre-lighting parameters
        self.flat_lighting_direction = self.camera_looks_at - self.camera_position
        # Lighting parameters
        self.i_a = np.array([30, 30, 30]) # ambient lighting intensity
        self.k_a = np.array([0.7, 0.7, 0.7]) #material ambient lighting reception coefficient
        self.i_d = np.array([160, 140, 150]) # diffuse lighting intensity (light color)
        self.k_d = np.array([0.8, 1, 0.6]) # material diffuse lighting reception coefficient (material color)
        self.i_s = np.array([200, 120, 170]) # spatial lighting intensity
        self.k_s = np.array([0.2, 0.3, 0.7]) # material reflection coefficient
        self.alfa = 1 # gloss
        #self.i_a = np.array([170, 170, 170])  # ambient lighting intensity
        #self.k_a = np.array([0.2, 0.2, 0.2]) #material ambient lighting reception coefficient
        #self.i_d = np.array([110, 90, 100]) # diffuse lighting intensity (light color)
        #self.k_d = np.array([0.3,  0.45, 0.1 ]) # material diffuse lighting reception coefficient (material color)
        #self.i_s = np.array([150, 70, 120]) # spatial lighting intensity
        #self.k_s = np.array([0.7, 0.8, 0.2]) # material reflection coefficient
        #self.alfa = 0.5 # gloss
        self.lighting_sources_with_power = [#[np.array([-3, 3, 5]), 40],
                                            [np.array([-3, 3, 30]), 300],
                                            [np.array([100, -15, -3]), 10000]
                                            ]
        self.lighting_sources_camera_space = []
        # Additional Lambert lighting parameters
        self.gradient_direction = np.array([1, -1, 1])
        self.gradient_start_point = np.array([0, 0, 1.5])
        self.gradient_depth = 22
        self.start_gradient_color = np.array([255, 0, 0])
        self.end_gradient_color = np.array([255, 255, 0])

    def __read_file(self, source: str):
        f = open(source, 'r')
        self.__data = f.readlines()

    def __create_figure_tuple(self):
        figures, vertices, texture_vertices, normals = [], [], [], []
        f = self.__data
        if f is not None:
            for line in f:
                vs = re.findall(r'v\s+[+-]?\d*[.]?\d+\s+[+-]?\d*[.]?\d+\s+[+-]?\d*[.]?\d+', line)
                fs = re.findall(r'f\s+[0-9/]+\s+[0-9/]+\s+[0-9/]+\s+(?:[0-9/]+)?(?:\s+[0-9/]+)?(?:\s+[0-9/]+)?', line)
                vts = re.findall(r'vt\s+[+-]?\d*[.]?\d+\s+[+-]?\d*[.]?\d+', line)
                vns = re.findall(r'vn\s+[+-]?\d*[.]?\d+\s+[+-]?\d*[.]?\d+\s+[+-]?\d*[.]?\d+', line)
                if len(vs) > 0:
                    vertices.append(vs[0].split())
                if len(fs) > 0:
                    figures.append(fs[0].split())
                if len(vts) > 0:
                    texture_vertices.append(vts[0].split())
                if len(vns) > 0:
                    normals.append(vns[0].split())
        self.__figure = (vertices, figures, texture_vertices, normals)

    def __upload_texture(self, texture_source):
        self.__texture = plt.imread(texture_source)
        self.__texture = np.flipud(self.__texture)
        self.__texture = np.transpose(self.__texture, axes=(1, 0, 2))
        #self.__texture = np.fliplr(self.__texture)
        #plt.imshow(self.__texture)
        #plt.show()

    def barycentric_line(self, point1, point2, pixel):
        return np.linalg.inv(np.array([point1[0], point2[0]],
                                      [point1[1], point2[1]])) @ np.array([pixel[0], pixel[1], 1])
        '''return np.array([(pixel[0] - point2[0])/(point1[0] - point2[0]),
                         (point1[0] - pixel[0])/(point1[0] - point2[0]),
                         0])'''
        #pixel = a*point1[:2] + b*point2[:2]
        #a + b = 1

    def barycentric(self, point1, point2, point3, pixel):
        points = np.array([[point1[0], point2[0], point3[0]],
                          [point1[1], point2[1], point3[1]],
                          [1,         1,         1]], dtype=np.float64)
        if np.linalg.det(points) == 0:
            if np.array_equal(point1, point2) or np.array_equal(point3, point2):
                return self.barycentric_line(point1, point3, pixel)
            else:
                return self.barycentric_line(point2, point3, pixel)
        else:
            b = np.linalg.inv(points) @ np.array([pixel[0], pixel[1], 1])
            #if not np.array_equal(np.round(point1*b[0] + point2*b[1] + point3*b[2])[:2].astype(int), pixel):
                #print(np.round(point1*b[0] + point2*b[1] + point3*b[2])[:2].astype(int), pixel) - проверка
            return b
        #pixel = a*point1[:2] + b*point2[:2] + c*point3[:2]
        #a + b + c = 1

    def projective(self, x):
        x = np.concatenate([x, np.ones((1, x.shape[1]))], axis=0)
        return x

    def cartesian(self, x):
        return x[:-1] / x[-1]

    def rotation_matrix(self, Ox_rotation_angle : float, Oy_rotation_angle : float, Oz_rotation_angle : float):
        Rz = np.array([[np.cos(Oz_rotation_angle), -np.sin(Oz_rotation_angle), 0, 0],
                          [np.sin(Oz_rotation_angle), np.cos(Oz_rotation_angle), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        Ry = np.array([[np.cos(Oy_rotation_angle),  0, np.sin(Oy_rotation_angle), 0],
                         [0,              1, 0,             0],
                         [-np.sin(Oy_rotation_angle), 0, np.cos(Oy_rotation_angle), 0],
                         [0,              0, 0,             1]])
        Rx = np.array([[1, 0, 0, 0],
                      [0, np.cos(Ox_rotation_angle), -np.sin(Ox_rotation_angle), 0],
                      [0, np.sin(Ox_rotation_angle), np.cos(Ox_rotation_angle), 0],
                      [0, 0, 0, 1]])
        return Rz @ Ry @ Rx

    def translation_matrix(self, x_translation, y_translation, z_translation):
        return np.array([[1, 0, 0, x_translation],
                         [0, 1, 0, y_translation],
                         [0, 0, 1, z_translation],
                         [0, 0, 0, 1]])

    def scale_matrix(self, x_scale: float, y_scale: float, z_scale: float):
        return np.array([[x_scale, 0, 0, 0],
                         [0, y_scale, 0, 0],
                         [0, 0, z_scale, 0],
                         [0, 0, 0,       1]])

    def normalize(self, vector):
        s = 0
        for coord in vector:
            s += coord**2
        normalized_vector = vector/np.sqrt(s)
        return np.array(normalized_vector)

    def projection_onto_vector_matrix(self, vector):
        return np.array([[vector[0]**2, vector[0]*vector[1], vector[0]*vector[2]],
                         [vector[0]*vector[1], vector[1]**2, vector[1]*vector[2]],
                         [vector[0]*vector[2], vector[1]*vector[2], vector[2]**2]])

    def __Mo2w(self, x_translation: float, y_translation: float, z_translation: float,
                   Ox_rotation_angle: float, Oy_rotation_angle: float, Oz_rotation_angle: float,
                   x_scale: float, y_scale: float, z_scale: float):
        T = self.translation_matrix(x_translation, y_translation, z_translation)
        R = self.rotation_matrix(Ox_rotation_angle, Oy_rotation_angle, Oz_rotation_angle)
        S = self.scale_matrix(x_scale, y_scale, z_scale)
        return T @ S @ R
        #- __Mo2w - матрица перехода между системой координат модели и мировой системой координат . Представить в виде произведения трех матриц RTS (вращения, переноса, масштабирования).

    def __Mw2C(self):
        x0, y0, z0 = self.camera_position[0], self.camera_position[1], self.camera_position[2]
        x1, y1, z1 = self.camera_looks_at[0], self.camera_looks_at[1], self.camera_looks_at[2]
        gamma = self.normalize(np.array([x1-x0, y1-y0, z1-z0]))
        beta = self.normalize(np.cross(gamma, np.array([gamma[0], gamma[1], 0])))
        alpha = self.normalize(np.cross(beta, gamma))
        Rc = np.array([[alpha[0], alpha[1], alpha[2], 0],
                       [beta[0],  beta[1],  beta[2], 0],
                       [-gamma[0], -gamma[1], -gamma[2], 0],
                       [0,        0,       0,        1]])
        Tc = self.translation_matrix(-self.camera_position[0],
                                     -self.camera_position[1],
                                     -self.camera_position[2])
        return Tc @ Rc
        #- Mw2c - матрица перехода между мировой системой координат и системой координат камеры.

    def __Mw2C_rot_only(self):
        x0, y0, z0 = self.camera_position[0], self.camera_position[1], self.camera_position[2]
        x1, y1, z1 = self.camera_looks_at[0], self.camera_looks_at[1], self.camera_looks_at[2]
        gamma = self.normalize(np.array([x1-x0, y1-y0, z1-z0]))
        beta = self.normalize(np.cross(gamma, np.array([gamma[0], gamma[1], 0])))
        alpha = self.normalize(np.cross(beta, gamma))
        return np.array([[alpha[0], alpha[1], alpha[2], 0],
                       [beta[0],  beta[1],  beta[2], 0],
                       [-gamma[0], -gamma[1], -gamma[2], 0],
                       [0,        0,       0,        1]])

    def __Mproj(self):
        S = self.scale_matrix(2/(self.right-self.left),
                              2/(self.top-self.bottom),
                              -2 / (self.z_far_plane - self.z_near_plane));
        T = self.translation_matrix(-(self.right+self.left)/2, -(self.top+self.bottom)/2, (self.z_far_plane+self.z_near_plane)/2)
        return S @ T
        #- __Mproj - матрица проекции (ортографическая)

    def __Mviewport(self, x=0.0, y=0.0):
        Tw = self.translation_matrix(x+self.viewport_size[0]/2, y+self.viewport_size[1]/2, 1)
        Sw = self.scale_matrix(self.viewport_size[0]/2, self.viewport_size[1]/2, 1)
        return Tw @ Sw
        #- __Mviewport - матрица перехода в систему координат области вывода (ViewPort)

    #2. Отсечение невидимых и пересекающихся фрагментов модели:


    def normal_vector(self, point1, point2, point3):
        return self.normalize(np.cross(point2 - point1, point3 - point1))

    def __is_to_cull(self, vertices, face_normals=np.array([])):
        normal = self.normal_vector(vertices[0], vertices[1], vertices[2])
        #for i in range(len(face_normals)):
            #if np.dot(vertices[i] - self.camera_position, face_normals[i]) < 0:
                #return False - for perspective projection
        if np.dot(self.camera_looks_at - self.camera_position, normal) < 0:
            return False
        return True

    def __gradient(self, world_points):
        from_gradient_start_to_face = world_points.T[0][:3] - self.gradient_start_point
        projection = self.projection_onto_vector_matrix(self.gradient_direction)
        from_gradient_point_to_vertice_plane = projection @ from_gradient_start_to_face
        if np.dot(from_gradient_point_to_vertice_plane, self.gradient_direction) < 0:
            return 0
        distance = np.sqrt(sum(from_gradient_point_to_vertice_plane**2))
        if distance > self.gradient_depth:
            distance = self.gradient_depth
        if distance < 0:
            distance = 0
        G = distance/self.gradient_depth
        if G < 0:
            G = 0
        return G

    def __flat_lighting(self, world_points):
        light_dir = self.normalize(self.flat_lighting_direction)
        n = self.normal_vector(world_points.T[0][:3], world_points.T[1][:3], world_points.T[2][:3])
        I = -np.dot(light_dir, n)
        if I < 0:
            I = 0
        return I

    def __camera_space_Lambert(self, camera_points, n_face, a, b, c):
        #pixel_world_position = camera_points.T[0][:3]*a + camera_points.T[1][:3]*b + camera_points.T[2][:3]*c
        barycenter = camera_points.T[0][:3]/3 + camera_points.T[1][:3]/3 + camera_points.T[2][:3]/3
        I = self.i_a*self.k_a
        n_face = self.cartesian(self.translation_matrix(self.camera_position[0],
                                                        self.camera_position[1],
                                                        self.camera_position[2])
                                    @ self.__current_Mw2C @ self.projective(n_face.T)).T
        for j in range(len(n_face)):
            n_face[j] = self.normalize(n_face[j])
        #N = self.normalize(a*n_face[0] + b*n_face[1] + c*n_face[2])
        N = self.normalize(n_face[0]/3 + n_face[1]/3 + n_face[2]/3)
        for i in range(len(self.lighting_sources_camera_space)):
            d = np.sqrt(sum((self.lighting_sources_camera_space[i][0] - barycenter)**2))
            L = self.normalize(self.lighting_sources_camera_space[i][0] - barycenter)
            cosTh = np.dot(N, L)
            Id = self.i_d * self.k_d * cosTh # Diffuse lighting
            I += Id*self.lighting_sources_camera_space[i][1]/d**2 # a,b=0; c=1
        I = np.clip(I, 0, 255)
        return I/255

    def __camera_space_Phong(self, camera_points, n_face, a, b, c):
        #barycenter = camera_points.T[0][:3]/3 + camera_points.T[1][:3]/3 + camera_points.T[2][:3]/3
        pixel_camera_space = camera_points.T[0][:3]*a + camera_points.T[1][:3]*b + camera_points.T[2][:3]*c
        n_face = self.cartesian(self.translation_matrix(self.camera_position[0],
                                                        self.camera_position[1],
                                                        self.camera_position[2])
                                    @ self.__current_Mw2C @ self.rotation_matrix(5*np.pi/180, 15*np.pi/180, 10*np.pi/180)\
                                    @ self.projective(n_face.T)).T
        for i in range(len(n_face)):
            n_face[i] = self.normalize(n_face[i])
        d = np.sqrt(sum((self.lighting_sources_camera_space[0][0] - pixel_camera_space)**2)) # distance between lightsource and surface point
        N = self.normalize(a*n_face[0] + b*n_face[1] + c*n_face[2])
        L = self.normalize((self.lighting_sources_camera_space[0][0] - pixel_camera_space)/d)
        cosTh = np.dot(N, L)/(np.sqrt(sum(N**2))*np.sqrt(sum(L**2)))
        Id = self.i_d * self.k_d * cosTh # Diffuse lighting
        #V = self.normalize(-pixel_camera_space)
        V = self.normalize(np.array([0, 0, 1]))
        R = self.normalize(2*cosTh*N - L)
        cosRV = np.dot(R, V)
        if cosRV > 0:
            Is = self.i_s * self.k_s * cosRV**self.alfa  # Spatial lighting
        else:
            Is = 0  # Spatial lighting
        I =  self.i_a*self.k_a + (Id + Is)*self.lighting_sources_camera_space[0][1]/d**2 # a,b=0; c=1
        I = np.clip(I, 0, 255)
        return I/255

    def __Lambert(self, world_points, n_face, a, b, c):
        #pixel_world_position = world_points.T[0][:3]*a + world_points.T[1][:3]*b + world_points.T[2][:3]*c
        barycenter = world_points.T[0][:3]/3 + world_points.T[1][:3]/3 + world_points.T[2][:3]/3
        I = self.i_a*self.k_a
        for j in range(len(n_face)):
            n_face[j] = self.normalize(n_face[j])
        #N = self.normalize(a*n_face[0] + b*n_face[1] + c*n_face[2])
        N = self.normalize(n_face[0]/3 + n_face[1]/3 + n_face[2]/3)
        for i in range(len(self.lighting_sources_with_power)):
            d = np.sqrt(sum((np.array(self.lighting_sources_with_power[i][0]) - barycenter)**2))
            L = self.normalize(self.lighting_sources_with_power[i][0] - barycenter)
            cosTh = np.dot(N, L)
            Id = self.i_d * self.k_d * cosTh # Diffuse lighting
            I += Id*self.lighting_sources_with_power[i][1]/d**2 # a,b=0; c=1
        I = np.clip(I, 0, 255)
        return I/255

    def __Phong(self, world_points, n_face, a, b, c):
        #barycenter = world_points.T[0][:3]/3 + world_points.T[1][:3]/3 + world_points.T[2][:3]/3
        pixel_world_position = world_points.T[0][:3]*a + world_points.T[1][:3]*b + world_points.T[2][:3]*c
        n_face = self.cartesian(self.rotation_matrix(5 * np.pi / 180, 15 * np.pi / 180,
                                                                             10 * np.pi / 180) \
                                @ self.projective(n_face.T)).T
        for i in range(len(n_face)):
            n_face[i] = self.normalize(n_face[i])
        d = np.sqrt(sum((self.lighting_sources_with_power[0][0] - pixel_world_position)**2)) # distance between lightsource and surface point
        N = self.normalize(a*n_face[0] + b*n_face[1] + c*n_face[2])
        L = self.normalize((self.lighting_sources_with_power[0][0] - pixel_world_position)/d)
        cosTh = np.dot(N, L)/(np.sqrt(sum(N**2))*np.sqrt(sum(L**2)))
        Id = self.i_d * self.k_d * cosTh # Diffuse lighting
        #V = self.normalize(self.camera_position - pixel_world_position)
        V = self.normalize(self.camera_position - self.camera_looks_at)
        R = self.normalize(2*cosTh*N - L)
        cosRV = np.dot(R, V)
        if cosRV > 0:
            Is = self.i_s * self.k_s * cosRV**self.alfa  # Spatial lighting
        else:
            Is = 0
        I = self.i_a*self.k_a + (Id + Is)*self.lighting_sources_with_power[0][1]/d**2 # a,b=0; c=1
        I = np.clip(I, 0, 255)
        return I/255

    # Отрисовка

    def __empty_frame(self, background_color):
        img = np.zeros((self.viewport_size[0], self.viewport_size[1], 4), np.uint8)
        img[:, :, :3] = background_color
        img[:, :, 3] = 255
        return img

    def __color_pixel(self, img, x, y, color):
        img[x, y, :3] = color[:3]

    def __draw_line(self, img, point1, point2, color):
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
            self.__color_pixel(img, int(x0), int(y0), color)
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

    def __draw_polygon(self, img, points, texture_points, normals):
        if len(points) > 3:
            self.__draw_polygon(img,
                                np.delete(points, 1, 0),
                                np.delete(texture_points, 1, 0),
                                np.delete(normals, 1, 0))

        self.__draw_triangle(img,
                             points[:3],
                             texture_points[:3],
                             normals[:3])

    def __draw_triangle(self, img, points, texture_points, normals):
        world_points = self.__current_Mo2w @ self.projective(points.T)
        if self.__is_to_cull(world_points[:3].T, normals):
            return
        camera_points = self.__current_Mw2C @ world_points
        projection_points = self.__current_Mproj @ camera_points
        screen_points = self.__current_Mviewport @ projection_points
        screen_points = self.cartesian(screen_points).T

        x_min = int(np.floor(min(screen_points[0][0], screen_points[1][0], screen_points[2][0])))
        x_max = int(np.ceil(max(screen_points[0][0], screen_points[1][0], screen_points[2][0])))
        y_min = int(np.floor(min(screen_points[0][1], screen_points[1][1], screen_points[2][1])))
        y_max = int(np.ceil(max(screen_points[0][1], screen_points[1][1], screen_points[2][1])))
        if x_max > img.shape[0]:
            x_max = img.shape[0]
        if y_max > img.shape[1]:
            y_max = img.shape[1]
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        #I = self.__flat_lighting(world_points)
        #G = self.__gradient(world_points)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                b = self.barycentric(screen_points[0], screen_points[1], screen_points[2], np.array([x, y]))
                z_proj = screen_points[0][2]*b[0] + screen_points[1][2]*b[1] + screen_points[2][2]*b[2]
                if np.any(b < 0) or self.z_buffer[x, y] <= z_proj:
                    continue
                uv = texture_points[0]*b[0] + texture_points[1]*b[1] + texture_points[2]*b[2]

                #I = self.__camera_space_Lambert(camera_points.copy(), normals.copy(), b[0], b[1], b[2])
                #I = self.__Lambert(world_points.copy(), normals.copy(), b[0], b[1], b[2])
                I = self.__camera_space_Phong(camera_points.copy(), normals.copy(), b[0], b[1], b[2])
                #I = self.__Phong(world_points.copy(), normals.copy(), b[0], b[1], b[2])
                color = self.__texture[int(uv[0]*self.__texture.shape[0]), int(uv[1]*self.__texture.shape[1])]*I
                #color = np.array([255, 255*G, 0])*I
                #color = np.array([255, 255, 255])*I
                self.__color_pixel(img, x, y, color)
                self.z_buffer[x, y] = z_proj
        '''
        img = self.__draw_wireframe(img, screen_points, np.array([255,255,255]))
        '''
        return img

    def __draw_wireframe(self, img, screen_points, color):
        for i in range(len(screen_points)-1):
             img = self.__draw_line(img, screen_points[i], screen_points[i+1], color)
        img = self.__draw_line(img, screen_points[0], screen_points[len(screen_points)-1], np.array([255,  255, 255]))
        return img

    def __draw_image(self):
        self.__current_Mo2w = self.__Mo2w(-2, 3, -2,
                                         5*np.pi/180, 15*np.pi/180, 10*np.pi/180,
                                         0.8, 0.8, 0.8
                                         )
        self.__current_Mw2C = self.__Mw2C()
        l_sources = []
        for l in self.lighting_sources_with_power:
            l_sources.append(l[0].tolist())
        l_sources = self.cartesian(self.__current_Mw2C @ self.projective(np.array(l_sources).T)).T
        print(l_sources)
        for i in range(len(self.lighting_sources_with_power)):
            self.lighting_sources_camera_space.append(
                [l_sources[i], self.lighting_sources_with_power[i][1]]
            )
        self.__current_Mproj = self.__Mproj()
        self.__current_Mviewport = self.__Mviewport()

        img = self.__empty_frame(np.array([0, 0, 0], np.uint8))
        print("Total vertices: " + str(len(self.__figure[0])))
        print("Total polygons: " + str(len(self.__figure[1])))
        print("Total texture uvs: " + str(len(self.__figure[2])))
        print("Total normals: " + str(len(self.__figure[3])))
        for i in range(0, len(self.__figure[1])):
            print('\r{0}'.format(np.round(i/len(self.__figure[1])*100, 2)), '%', end='')
            points = self.__form_points_from_parsed_face(self.__figure[1][i])
            texture_points = self.__form_texture_points_from_parsed_face(self.__figure[1][i])
            normals = self.__form_normal_vector_from_parsed_face(self.__figure[1][i])
            self.__draw_polygon(img, points, texture_points, normals)
        img = np.flipud(img)
        im = plt.imshow(img)
        #
        #plt.show()
        #
        return im

    def __draw_animation(self):
        loop_fps = 100
        total_frames = 100

        frames = []
        fig = plt.figure(dpi = 500)
        time_passed = 0
        frame_consequent = 100
        while time_passed <= total_frames/loop_fps: # 1 second animation for alfa not to drop below 0
            if np.round(time_passed*loop_fps) % frame_consequent == 0:
                print(self.i_a, self.i_d, self.i_s)
                print(self.k_a, self.k_d, self.k_s)
                print(self.alfa)
                frames.append([self.__draw_image()])

            self.i_a -= 1
            self.i_d -= 1
            self.i_s -= 1
            for i in range(len(self.i_a)):
                if self.i_a[i] == 10:
                    self.i_a[i] = 200
            for i in range(len(self.i_d)):
                if self.i_d[i] == 10:
                    self.i_d[i] = 200
            for i in range(len(self.i_s)):
                if self.i_s[i] == 10:
                    self.i_s[i] = 200
            self.k_a += 0.05
            self.k_d += 0.05
            self.k_s += 0.05
            for i in range(len(self.k_a)):
                if self.k_a[i] > 1:
                    self.k_a[i] = 0
            for i in range(len(self.k_d)):
                if self.k_d[i] > 1:
                    self.k_d[i] = 0
            for i in range(len(self.k_s)):
                if self.k_s[i] > 1:
                    self.k_s[i] = 0
            self.alfa -= 0.01

            time_passed += 1/loop_fps
            self.__setup_default_filters()

        slowdown = 10
        ani = animation.ArtistAnimation(fig, frames, interval=1000/loop_fps * slowdown, blit=False, repeat_delay=0)
        writer = PillowWriter(fps=loop_fps/slowdown)
        writer.setup(fig, "Data\\PHONG_SKULL_ANIMATION.gif", dpi=500)
        ani.save("Data\\PHONG_SKULL_ANIMATION.gif", writer=writer, dpi="figure")
        plt.show()

    def __form_points_from_parsed_face(self, face):
        ps = []
        for i in range(1, len(face)):
            vertex = int(face[i].split("/")[0])
            point = np.array(self.__figure[0][vertex-1][1:], dtype=np.float64)
            ps.append(point)
        points = np.array(ps)
        return points

    def __form_normal_vector_from_parsed_face(self, face):
        ns = []
        for i in range(1, len(face)):
            normal_vertex = int(face[i].split("/")[2])
            normal = np.array(self.__figure[3][normal_vertex-1][1:], dtype=np.float64)
            ns.append(normal)
        normals = np.array(ns)
        return normals

    def __form_texture_points_from_parsed_face(self, face):
        ts = []
        for i in range(1, len(face)):
            texture_vertex = int(face[i].split("/")[1])
            texture_point = np.array(self.__figure[2][texture_vertex-1][1:], dtype=np.float64)
            ts.append(texture_point)
        texture_points = np.array(ts)
        return texture_points

#12140_Skull_v3_L2
graphics = Graphics3D("Data3D\\12140_Skull_v3_L2.obj", "Data3D\\Skull.jpg")