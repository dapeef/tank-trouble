""" Module containing the SkeletonGraph and Graph class """

# Imports
import vectormath as vmath


# Classes
class SkeletonGraph():
    """ Object to handle points and edges (not faces) """

    def __init__(self):
        self._points: [vmath.Vector2] = []
        self._edges: [(int, int)] = []

    # Private functions
    def _serialise_to_json(self):
        """ Returns a json of _points and _edges (for the testing suite) """

    # Public functions
    def show(self, show_edges=True, show_points=False, flip=False, bounding_box=(0, 0)):
        """ Draws edges and or points , applies plt settings and shows plt window """

    # Classmethods
    @classmethod
    def _serialise_from_json(cls, json: str):
        """ Returns a SkeletonGraph from a json of _points and _edges (for the testing suite) """

    @classmethod
    def construct_from_json(cls, json: str):
        """
        Returns a SkeletonGraph of edges and points from a json in the form of
        [
            [[x1, y1], [x2, y2]],
            [[x3, y3], [x4, y4]],
            ...
        ]
        """

    @classmethod
    def construct_from_translated_skeletongraphs(cls, translations):
        """ Returns a SkeletonGraph from the translations in the form
        [
            [SkeletonGraph1, [Vec2, Vec2, Vec2, ...]],
            [SkeletonGraph2, [Vec2, Vec2, Vec2, ...]]
        ]
        """


class Graph(SkeletonGraph):
    """ Object to handle points, edges and faces """

    def __init__(self):
        super().__init__()
        self._faces: [[int]] = []

    # Private functions
    def _apply_density(self, density):
        """
        Disables walls until the proportion of "density" has been removed, leaving "lonely" walls
        """

    def _detect_faces(self):
        """ Detects faces and populates self._faces """

    # Public functions
    def make_maze(self, density):
        """ Disables walls from the Graph to make a maze """

    # Classmethods
    @classmethod
    def construct_from_skeletongraph(cls, skeleton_graph: SkeletonGraph):
        """ Returns a Graph with detected faces from a SkeletonGraph """
