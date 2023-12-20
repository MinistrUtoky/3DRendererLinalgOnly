import re
import numpy

class UtahTeapotEvaluator:
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

    def evaluate_teapot(self, teapot_tuple):
        total_inner_discus_area = 0
        max_cos = 0
        for figure in teapot_tuple[1]:
            point1 = teapot_tuple[0][int(figure[1])-1][1:]
            point2 = teapot_tuple[0][int(figure[2])-1][1:]
            point3 = teapot_tuple[0][int(figure[3])-1][1:]
            total_inner_discus_area += self.inner_discus_area(point1, point2, point3)
            max_cos = self.max_cos(point1, point2, point3)
        print("Total inner discus area for shape's triangles: " + str(total_inner_discus_area))
        print("Greatest angle cos among all triangles: " + str(max_cos))

    def inner_discus_area(self, point1, point2, point3):
        for i in range(len(point1)):
            point1[i] = float(point1[i])
            point2[i] = float(point2[i])
            point3[i] = float(point3[i])
        a = self.magnitude(self.vector(point2, point1))
        b = self.magnitude(self.vector(point3, point2))
        c = self.magnitude(self.vector(point1, point3))
        p = (a+b+c)/2
        S = numpy.math.sqrt(p*(p-a)*(p-b)*(p-c))
        r = S/p
        dS = numpy.math.pi*r**2
        return dS

    def max_cos(self, point1, point2, point3):
        for i in range(len(point1)):
            point1[i] = float(point1[i])
            point2[i] = float(point2[i])
            point3[i] = float(point3[i])
        vector1 = self.vector(point1, point2)
        vector2 = self.vector(point2, point3)
        vector3 = self.vector(point3, point1)
        return max(self.cos(vector1, [-c for c in vector3]),
                   self.cos(vector2, [-c for c in vector1]),
                   self.cos(vector3, [-c for c in vector2]))

    def vector(self, point1, point2):
        vector = []
        for i in range(len(point1)):
            vector.append(point2[i] - point1[i])
        return vector

    def magnitude(self, vector):
        summ = 0
        for coord in vector:
            summ += coord**2
        return numpy.math.sqrt(summ)

    def dot_product(self, vector1, vector2):
        prod = 0
        for i in range(len(vector1)):
            prod += vector1[i]*vector2[i]
        return prod

    def cos(self, vector1, vector2):
        return self.dot_product(vector1, vector2)/self.magnitude(vector1)/self.magnitude(vector2)

teapotEval = UtahTeapotEvaluator()
teapotEval.evaluate_teapot(teapotEval.teapot_tuple())
