""" Module containing the Graph class """

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


NEAR_THRESHOLD = 0.1
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

        def raw(self):
            """ Returns a point in tuple format """
            return [self.x, self.y]

        def add_parent(self, edge):
            """ Adds an edge as a parent of the point """
            self.parents.append(edge)

        @ classmethod
        def raw_array(cls, points):
            """ Returns an array of raw points when given an array of Point objects """
            return [i.raw() for i in points]

    class Edge():
        """ Edge class """

        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.enabled = True
            self.children = []
            self.parents = []

        def raw(self):
            """ Returns an edge in tuple format """
            return [[self.start.x, self.start.y], [self.end.x, self.end.y]]

        def add_child(self, point):
            """ Adds a point as a child of the edge """
            self.children.append(point)

        def add_parent(self, face):
            """ Adds a face as a parent of the edge """
            self.parents.append(face)

        @ classmethod
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
            self.id = 0

        def raw(self):
            """ Returns an array of edges is raw format """
            return Graph.Edge.raw_array(self.edges)

        def add_child(self, edge):
            """ Adds an edge as a child of the face """
            self.children.append(edge)

    # Private functions
    def _add_point(self, x: float, y: float):
        """ Adds point to graph and returns point (even if already exists) """

        x = round(x, 2)
        y = round(y, 2)

        if not [x, y] in self.Point.raw_array(self._points):
            point = self.Point(x, y)

            self._points.append(point)

        else:
            point = list(
                filter(
                    lambda a: a.raw() == [x, y], self._points
                ))[0]

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
            edge = list(
                filter(
                    lambda a: [a.start, a.end] == [start, end], self._edges
                ))[0]

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

    def _are_points_near(self, x1, y1, x2, y2):  # pylint: disable=invalid-name
        """ Returns boolean whether 2 points are near each other """

        dx = x2 - x1  # pylint: disable=invalid-name
        dy = y2 - y1  # pylint: disable=invalid-name

        return math.sqrt(dx * dx + dy * dy) < NEAR_THRESHOLD

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

        edges = [edge for edge in self._edges if len(edge.parents) == 2]

        random.shuffle(edges)

        for i, face in enumerate(self._faces):
            face.id = i

        for edge in edges:
            print(edge.parents[0].id, edge.parents[1].id,
                  [face.id for face in self._faces])
            if edge.parents[0].id != edge.parents[1].id:
                edge.enabled = False

                dead_id = edge.parents[1].id

                for face in self._faces:
                    if face.id == dead_id:
                        face.id = edge.parents[0].id

        print([face.id for face in self._faces])

    def show(self, show_edges=True, show_points=False, flip=False, bounding_box=[0, 0]):
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
