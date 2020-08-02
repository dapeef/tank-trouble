""" Module containing the Graph class """

# Imports
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import vectormath as vmath
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module


# Constants
NEAR_THRESHOLD = 0.5


# Classes
class Graph():
    """ Object to manage the graph layout for the maze gen """

    def __init__(self):
        self._points = []
        self._edges = []
        self._faces = []

    # Classes
    class Point():
        """ Point class """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parents = []

        def delete(self, graph):
            """ Function to delete point, clearing up all parents edges """

            for edge in self.parents:
                edge.delete(graph)

            # pylint: disable = protected-access
            graph._points = list(filter(
                lambda point: point != self, graph._points))

        def raw(self):
            """ Returns a point in tuple format """
            return [self.x, self.y]

        def add_parent(self, edge):
            """ Adds an edge as a parent of the point """
            self.parents.append(edge)

        def angle(self):
            """ Returns the angle between the 2 edges for a point with 2 parents """

            if len(self.parents) == 2:
                if self.parents[0].start == self:
                    point0 = self.parents[0].end
                else:
                    point0 = self.parents[0].start

                if self.parents[1].start == self:
                    point1 = self.parents[1].end
                else:
                    point1 = self.parents[1].start

                vec0 = vmath.Vector2(point0.x - self.x, point0.y - self.y)
                vec1 = vmath.Vector2(point1.x - self.x, point1.y - self.y)

                return round(math.acos(vec0.dot(vec1)) / math.pi * 180, 2)

        def num_enabled_parents(self):
            """ Returns the number of enabled edges connected to the point """

            num = 0

            for edge in self.parents:
                if edge.enabled:
                    num += 1

            return num

        @classmethod
        def raw_array(cls, points):
            """ Returns an array of raw points when given an array of Point objects """
            return [i.raw() for i in points]

        @classmethod
        def are_points_near(cls, x1, y1, x2, y2):  # pylint: disable=invalid-name
            """ Returns boolean whether 2 points are near each other """

            dx = x2 - x1  # pylint: disable=invalid-name
            dy = y2 - y1  # pylint: disable=invalid-name

            return math.sqrt(dx * dx + dy * dy) < NEAR_THRESHOLD

    class Edge():
        """ Edge class """

        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.enabled = True
            self.parents = []

        def delete(self, graph):
            """ Function to delete edge, tidied up edgeless points also """

            # pylint: disable = protected-access
            graph._edges = list(filter(
                lambda edge: edge != self, graph._edges))

            if len(self.start.parents) == 1:
                self.start.delete(graph)

            if len(self.end.parents) == 1:
                self.end.delete(graph)

        def raw(self):
            """ Returns an edge in tuple format """
            return [[self.start.x, self.start.y], [self.end.x, self.end.y]]

        def add_parent(self, face):
            """ Adds a face as a parent of the edge """
            self.parents.append(face)

        @classmethod
        def raw_array(cls, edges):
            """ Returns an array of raw edges when given an array of Edge objects """
            return [i.raw() for i in edges]

        @classmethod
        def sort_edge(cls, edge):
            """ Function to sort an edge so all edges are consistent and comparable """

            return sorted(
                sorted(
                    edge,
                    key=lambda point: point[0]),
                key=lambda point: point[1]
            )

    class Face():
        """ Face class """

        def __init__(self, edges):
            self.edges = edges
            self.adjacent_faces = []
            self.children = []
            self.id = 0  # pylint: disable = invalid-name

        def raw(self):
            """ Returns an array of edges is raw format """
            return Graph.Edge.raw_array(self.edges)

        def add_child(self, edge):
            """ Adds an edge as a child of the face """
            self.children.append(edge)

    # Private functions
    def _add_point(self, x: float, y: float):
        """ Adds point to graph and returns point (even if already exists) """

        x = round(x, 3)
        y = round(y, 3)

        exists = False

        for i in self._points:
            if self.Point.are_points_near(x, y, i.x, i.y):
                exists = True
                point = i

        if not exists:
            point = self.Point(x, y)

            self._points.append(point)

        return point

    def _add_point_from_json(self, point: [float]):
        """ Adds point from [x, y] format"""

        self._add_point(point[0], point[1])

    def _add_edge(self, start: Point, end: Point):
        """ Adds edge to graph and returns edge (even if already exists) """

        if not [start, end] in [[edge.start, edge.end] for edge in self._edges]:
            edge = self.Edge(start, end)

            self._edges.append(edge)

            start.parents.append(edge)
            end.parents.append(edge)

        else:
            for i in self._edges:
                if [i.start, i.end] == [start, end]:
                    edge = i

        return edge

    def _add_edge_from_json(self, edge: [[float]]):
        """ Adds edge from [[x1, y1], [x2, y2]] format and returns edge (even if already exists) """

        edge = Graph.Edge.sort_edge(edge)

        start = self._add_point(edge[0][0], edge[0][1])
        end = self._add_point(edge[1][0], edge[1][1])

        return self._add_edge(start, end)

    def _add_face(self, edges: [Edge]):
        """ Adds face to graph from and array of Edge objects """

        if not edges in [face.edges for face in self._faces]:
            face = self.Face(edges)

            self._faces.append(face)

            for edge in edges:
                if len(edge.parents) == 1:
                    face.adjacent_faces.append(edge.parents[0])
                    edge.parents[0].adjacent_faces.append(face)

                edge.parents.append(face)
                face.children.append(edge)

        else:
            face = list(
                filter(
                    lambda a: a.edges == edges, self._faces
                ))[0]

        return face

    def _add_face_from_json(self, edges: [[[float]]]):
        """ Adds face to graph from an array of raw edges and returns the Face """

        return self._add_face([self._add_edge_from_json(edge) for edge in edges])

    def _get_enabled_edges(self):
        return list(filter(lambda edge: edge.enabled, self._get_internal_edges()))

    def _get_internal_edges(self):
        return [edge for edge in self._edges if len(edge.parents) == 2]

    def _find_adjacent_tris(self, tri, tri_group, tris, np_points, edges):
        """ Recursive function to find all tris that make up a face """

        tri_points = [list(i) for i in np_points[tris.simplices][tri]]

        for i in tris.neighbors[tri]:
            if not i in tri_group and i != -1:
                i_points = [list(i) for i in np_points[tris.simplices][i]]

                edge = self.Edge.sort_edge(
                    list(filter(lambda point: point in tri_points, i_points))
                )

                if not edge in edges:
                    tri_group.append(i)

                    self._find_adjacent_tris(
                        i, tri_group, tris, np_points, edges)

        return tri_group

    # Public functions
    def combine_graph_edges_into_graph(self, graph):
        """ Combines existing Graph's edges with another Graph """

        for edge in graph._edges:  # pylint: disable=protected-access
            self._add_edge_from_json(edge.raw())

    def translate(self, x, y):
        """ Translates the graph by (x, y) """

        for point in self._points:
            point.x += x
            point.y += y

    def tidy_edges(self):
        """ A function to tidy up all unnecessary triangles poking out of the edges of the map """

        for point in self._points:
            if len(point.parents) == 2:
                if point.angle() < 80:
                    print("poo")
                    point.delete(self)

    def detect_faces(self):
        """ Used Delaunay to detect faces and adds them to the graph """

        # Implement Delaunay
        np_points = np.array(self.Point.raw_array(self._points))

        tris = Delaunay(np_points)

        # plt.triplot(np_points[:, 0], np_points[:, 1], tris.simplices)

        tri_ids = list(range(len(tris.simplices)))

        raw_edges = self.Edge.raw_array(self._edges)

        faces_edges = []

        while len(tri_ids) > 0:
            tri_group = self._find_adjacent_tris(
                tri_ids[0],
                [tri_ids[0]],
                tris,
                np_points,
                raw_edges
            )

            face_edges = [
                self.Edge.sort_edge([
                    list(np_points[tris.simplices][tri][i]),
                    list(np_points[tris.simplices][tri][(i + 1) % 3])
                ])
                for tri in tri_group
                for i in range(3)
            ]

            exterior_face_edges = []

            for edge in face_edges:
                if not edge in exterior_face_edges:
                    exterior_face_edges.append(edge)

                else:
                    exterior_face_edges.pop(exterior_face_edges.index(edge))

            valid = True

            for edge in exterior_face_edges:
                if not edge in raw_edges:
                    valid = False

            if valid:
                faces_edges.append(exterior_face_edges)

            tri_ids = list(filter(lambda x: x not in tri_group, tri_ids))

        # Create faces
        for face in faces_edges:
            self._add_face_from_json(face)

    def make_maze(self):
        """ Disables edges until all faces are connected into one contiguous group """

        edges = self._get_internal_edges()

        random.shuffle(edges)

        for i, face in enumerate(self._faces):
            face.id = i

        for edge in edges:
            if edge.parents[0].id != edge.parents[1].id:
                edge.enabled = False

                dead_id = edge.parents[1].id

                for face in self._faces:
                    if face.id == dead_id:
                        face.id = edge.parents[0].id

    def apply_density(self, density):
        """ Removes further walls to improve the map """

        enabled_edges = self._get_enabled_edges()

        random.shuffle(enabled_edges)

        removed_edges = 0

        for edge in enabled_edges:
            if edge.start.num_enabled_parents() > 1 and \
                    edge.end.num_enabled_parents() > 1:
                edge.enabled = False

                removed_edges += 1

            if removed_edges / len(enabled_edges) > 1 - density:
                break

    def show(self, show_edges=True, show_points=False, flip=False, bounding_box=(0, 0)):
        """ Applies plt settings and shows plt window """

        if flip:
            flip_sf = -1

        else:
            flip_sf = 1

        if show_edges:
            for edge in self._edges:
                if edge.enabled:
                    plt.plot(
                        [edge.start.x, edge.end.x],
                        [edge.start.y * flip_sf, edge.end.y * flip_sf]
                    )

        if show_points:
            for point in self._points:
                plt.plot(
                    point.x,
                    point.y*flip_sf,
                    "o"
                )

        if bounding_box != [0, 0]:
            plt.plot(
                [0, bounding_box[0], bounding_box[0], 0, 0],
                [0, 0, bounding_box[1] * flip_sf, bounding_box[1] * flip_sf, 0],
                "k:"
            )

        plt.gca().set_aspect('equal', adjustable='box')
        plt.gcf().set_size_inches(9, 9)
        plt.subplots_adjust(left=0.04, right=.999, top=1, bottom=0.03)
        plt.show()

    @classmethod
    def create_graph_from_json_edges(cls, json):
        """
        Function which returns a new Graph object when given json of edges in the form:

        [
            [[x1, y1], [x2, y2]],
            [[x3, y3], [x4, y4]],
            ...
        ]

        """

        graph = Graph()

        for edge in json:
            graph._add_edge_from_json(edge)  # pylint: disable=protected-access

        return graph

    @classmethod
    def translated(cls, graph, x, y):
        """ Translates the graph by (x, y) """

        temp_graph = Graph()

        # pylint: disable=protected-access
        for edge in graph._edges:
            edge_raw = edge.raw()
            temp_graph._add_edge_from_json(
                [[edge_raw[0][0] + x, edge_raw[0][1] + y],
                 [edge_raw[1][0] + x, edge_raw[1][1] + y]]
            )

        return temp_graph


if __name__ == "__main__":
    g = Graph.create_graph_from_json_edges([[[0, 0], [0, 1]],
                                            [[0, 1], [1, 1]],
                                            [[1, 1], [1, 0]],
                                            [[1, 0], [0, 0]]])

    g1 = Graph.create_graph_from_json_edges([[[0, 0.1], [0, 1]],
                                             [[0, 1], [1, 1.1]],
                                             [[1, 1.1], [1, 0]],
                                             [[1, 0], [0, 0.1]]])

    g1.translate(0, 2)

    g2 = Graph.create_graph_from_json_edges([[[1, 0], [1, 1]],
                                             [[1, 1], [2, 1]],
                                             [[2, 1], [2, 0]],
                                             [[2, 0], [1, 0]]])

    g3 = Graph.create_graph_from_json_edges([[[3, 2], [1, 1]],
                                             [[1, 1], [2, 1]],
                                             [[2, 1], [2, 0]],
                                             [[2, 0], [3, 2]]])

    g.combine_graph_edges_into_graph(g1)
    g1.translate(1, 0)
    g.combine_graph_edges_into_graph(g1)
    g1.translate(1, 0)
    g.combine_graph_edges_into_graph(g1)
    g.combine_graph_edges_into_graph(g2)
    g.combine_graph_edges_into_graph(g3)

    # pylint: disable=protected-access
    print(len(g._points), len(g._edges), len(g._faces))
    g.show(show_points=True)
