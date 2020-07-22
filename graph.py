""" Module containing the Graph class """

import matplotlib.pyplot as plt


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

        def raw(self):
            """ Returns an array of edges is raw format """
            return Graph.Edge.raw_array(self.edges)

        def add_child(self, edge):
            """ Adds an edge as a child of the face """
            self.children.append(edge)

    # Private functions
    def _add_point(self, x: float, y: float):
        """ Adds point to graph and returns point (even if already exists) """

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

                edge.parents.append(face)

        else:
            face = list(
                filter(
                    lambda a: a.edges == edges, self._faces
                ))[0]

        return face

    def _add_face_from_json(self, edges: [[[float]]]):
        """ Adds face to graph from an array of raw edges and returns the Face """

        return self._add_face([self._add_edge_from_json(edge) for edge in edges])

    # Public functions
    def combine_graph_into_graph(self, graph):
        """ Combines existing Graph's edges with another Graph """

        for face in graph._faces:  # pylint: disable=protected-access
            self._add_face_from_json(face.raw())

        for edge in graph._edges:  # pylint: disable=protected-access
            self._add_edge_from_json(edge.raw())

        for point in graph._points:  # pylint: disable=protected-access
            self._add_point_from_json(point.raw())

    def translate(self, x, y):
        """ Translates the graph by (x, y) """

        for point in self._points:
            point.x += x
            point.y += y

    def show(self, show_edges=True, show_points=False, flip=False):
        """ Applies plt settings and shows plt window """

        if flip:
            flip_sf = -1

        else:
            flip_sf = 1

        if show_edges:
            for edge in self._edges:
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

