""" Module containing the Graph class """


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
