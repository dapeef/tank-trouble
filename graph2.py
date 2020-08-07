""" Module containing the SkeletonGraph and Graph class """

# Imports
import json
import matplotlib.pyplot as plt
import vectormath as vmath


# Constants
NEAR_THRESHOLD = .7


# Classes
class SkeletonGraph():
    """ Object to handle points and edges (not faces) """

    def __init__(self):
        self._points: [vmath.Vector2] = []
        self._edges: [(int, int)] = []

    # Private functions
    def _serialise_to_json(self):
        """ Returns a json of _points and _edges (for the testing suite) """

    def _deduplicate(self):
        """ Removes duplicate points and edges """

        new_points = []

        points_lookup = []

        for point in self._points:
            unique = True

            for ind, new_point in enumerate(new_points):
                if self._are_points_near(point, new_point):
                    unique = False

                    reference = ind

                    break

            if unique:
                new_points.append(point)

                reference = len(new_points) - 1

            points_lookup.append(reference)

        self._points = new_points

        new_edges = []

        for edge in self._edges:
            new_edges.append(tuple(sorted([points_lookup[i] for i in edge])))

        self._edges = list(set(new_edges))

    def _are_points_near(self, point1, point2):
        #  pylint: disable=invalid-name
        dx = point1.x - point2.x
        dy = point1.y - point2.y

        return dx * dx + dy * dy < NEAR_THRESHOLD * NEAR_THRESHOLD

    # Public functions
    def show(self, show_edges=True, show_points=False, flip=False, bounding_box=(0, 0)):
        """ Draws edges and or points , applies plt settings and shows plt window """

        if flip:
            flip_sf = -1

        else:
            flip_sf = 1

        if show_edges:
            for edge in self._edges:
                if True:  # Requires an "enabled" value
                    plt.plot(
                        [
                            self._points[edge[0]].x,
                            self._points[edge[1]].x
                        ],
                        [
                            self._points[edge[0]].y * flip_sf,
                            self._points[edge[1]].y*flip_sf
                        ]
                    )

        if show_points:
            for point in self._points:
                plt.plot(
                    point.x,
                    point.y * flip_sf,
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

    # Classmethods
    @classmethod
    def _serialise_from_json(cls, raw_json: str):
        """ Returns a SkeletonGraph from a json of _points and _edges (for the testing suite) """

    @classmethod
    def construct_from_edges(cls, edges: str):
        """
        Returns a SkeletonGraph of edges and points from a json in the form of
        [
            [[x1, y1], [x2, y2]],
            [[x3, y3], [x4, y4]],
            ...
        ]
        """

        #  pylint: disable=protected-access
        skeleton_graph = cls()

        for edge in edges:
            skeleton_graph._points.append(edge[0])
            skeleton_graph._points.append(edge[1])

            len_points = len(skeleton_graph._points)

            skeleton_graph._edges.append([len_points - 2, len_points - 1])

        skeleton_graph._deduplicate()

        return skeleton_graph

    @classmethod
    def construct_from_translated_skeletongraphs(cls, translations):
        """ Returns a SkeletonGraph from the translations in the form
        [
            [SkeletonGraph1, [Vec2, Vec2, Vec2, ...]],
            [SkeletonGraph2, [Vec2, Vec2, Vec2, ...]]
        ]
        """

        #  pylint: disable=protected-access
        skeleton_graph = cls()

        for unit in translations:
            for offset in unit[1]:
                len_points = len(skeleton_graph._points)

                skeleton_graph._points += [
                    point + offset for point in unit[0]._points]

                skeleton_graph._edges += [
                    (edge[0] + len_points,
                     edge[1] + len_points)
                    for edge in unit[0]._edges
                ]

        skeleton_graph._deduplicate()

        return skeleton_graph


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
