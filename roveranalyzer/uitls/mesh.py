import numpy as np
import scipy.sparse as sp
from matplotlib import tri


class SimpleMesh:
    @classmethod
    def from_string(cls, poly_str):
        v, v1 = np.array([]), np.array([])

        text = poly_str.split("#")
        xy = text[1].splitlines()
        xy = xy[2:]
        for xy_ in xy:
            vals = np.fromstring(xy_, dtype=float, sep=" ")
            v = np.append(v, [vals[3], vals[4]])

        elements = text[4].splitlines()
        elements = elements[1:]

        for ele_ in elements:
            vals = np.fromstring(ele_, dtype=int, sep=" ")
            v1 = np.append(v1, [vals[1], vals[2], vals[3]])

        xy_ = v.reshape((-1, 2)).T
        x = xy_[0]
        y = xy_[1]
        triangles = v1.reshape((-1, 3)).astype(int) - 1

        return cls(x, y, triangles)

    @classmethod
    def from_path(cls, poly_file_path):
        v, v1 = np.array([]), np.array([])

        with open(poly_file_path) as file:
            text = file.read().split("#")

        return cls(text)

    def __init__(self, x, y, triangles):
        self.x = x
        self.y = y
        self.triangles = triangles
        self.mapping_matrices = self.__build_mapping_matrices()
        self.nodal_area = self.__build_nodal_areas()
        self.tri: tri.Triangulation = tri.Triangulation(self.x, self.y, self.triangles)

    def __build_mapping_matrices(self):
        rows, cols = np.array([], dtype=int), np.array([], dtype=int)

        ind = 0
        for triangle in self.triangles:
            rows = np.append(rows, triangle)
            cols = np.append(cols, [ind, ind, ind])
            ind += 1

        data = np.ones((1, len(rows))).ravel()
        mapping_matrix = sp.coo_matrix((data, (rows.ravel(), cols.ravel())))

        return mapping_matrix

    def __build_nodal_areas(self):
        areas = []

        for triangle in self.triangles:
            v0 = triangle[0]
            v1 = triangle[1]
            v2 = triangle[2]

            v0v1 = [self.x[v1] - self.x[v0], self.y[v1] - self.y[v0]]
            v0v2 = [self.x[v2] - self.x[v0], self.y[v2] - self.y[v0]]

            area = 0.5 * np.linalg.norm(np.cross(v0v1, v0v2))
            areas.append(area)

        return areas

    def get_xy_elements(self):
        return self.x, self.y, self.triangles
