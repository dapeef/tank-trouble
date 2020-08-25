""" Module containing the SkeletonGraph and Graph class """

# Imports
import json
import math
import matplotlib.pyplot as plt


# Constants
NEAR_THRESHOLD = .7
ANGLE_THRESHOLD = 80


# Classes
class SkeletonGraph():
    """ Object to handle points and edges (not faces) """

    def __init__(self):
        self._points: [(int, int)] = []
        self._edges: [(int, int)] = []

    # Private functions
    def _serialise_to_json(self):
        """ Returns a json of _points and _edges (for the testing suite) """

        return {
            "points": self._points,
            "edge_refs": self._edges,
            "edges": [(self._points[edge[0]], self._points[edge[1]]) for edge in self._edges]
        }

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
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        return dx * dx + dy * dy < NEAR_THRESHOLD * NEAR_THRESHOLD

    def _tidy_edges(self):
        point_count = [0 for i in self._points]

        for edge in self._edges:
            point_count[edge[0]] += 1
            point_count[edge[1]] += 1

        if 2 in point_count:
            points_with_2_edges = []
            index_pos = 0

            while True:
                try:
                    index_pos = point_count.index(2, index_pos)
                    points_with_2_edges.append(index_pos)
                    index_pos += 1

                except ValueError:
                    break

            for point in points_with_2_edges:
                relevant_points = []
                relevant_edges = []

                for ind, edge in enumerate(self._edges):
                    if edge[0] == point:
                        relevant_points.append(edge[1])
                        relevant_edges.append(ind)

                    elif edge[1] == point:
                        relevant_points.append(edge[0])
                        relevant_edges.append(ind)

                vec0 = (
                    self._points[relevant_points[0]][0] -
                    self._points[point][0],
                    self._points[relevant_points[0]][1] -
                    self._points[point][1]
                )
                vec1 = (
                    self._points[relevant_points[1]][0] -
                    self._points[point][0],
                    self._points[relevant_points[1]][1] -
                    self._points[point][1]
                )

                dot_product = vec0[0] * vec1[0] + vec0[1] * vec1[1]

                vec0_len = math.sqrt(vec0[0] ** 2 + vec0[1] ** 2)
                vec1_len = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)

                angle = math.acos(
                    dot_product / vec0_len / vec1_len
                ) / math.pi * 180

                if round(angle, 2) < ANGLE_THRESHOLD:
                    # Delete point

                    self._edges.pop(relevant_edges[1])
                    self._edges.pop(relevant_edges[0])

                    self._points.pop(point)

                    for ind, (point1, point2) in enumerate(self._edges):
                        self._edges[ind] = (
                            point1 - 1 if point1 > point else point1,
                            point2 - 1 if point2 > point else point2
                        )

    # Public functions
    def _show(self, edge_enable_list, show_edges=True, show_points=False, flip=False,
              bounding_box=(0, 0)):
        """ Draws edges and or points , applies plt settings and shows plt window """

        if flip:
            flip_sf = -1

        else:
            flip_sf = 1

        if show_edges:
            for ind, edge in enumerate(self._edges):
                is_enabled = True

                if ind < len(edge_enable_list):
                    is_enabled = edge_enable_list[ind]

                if is_enabled:  # Requires an "enabled" value
                    plt.plot(
                        [
                            self._points[edge[0]][0],
                            self._points[edge[1]][0]
                        ],
                        [
                            self._points[edge[0]][1] * flip_sf,
                            self._points[edge[1]][1]*flip_sf
                        ]
                    )

        if show_points:
            for point in self._points:
                plt.plot(
                    point[0],
                    point[1] * flip_sf,
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

    def show(self, **kwargs):
        """ Draws edges and or points , applies plt settings and shows plt window """

        self._show([], **kwargs)

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
            skeleton_graph._points.append((
                round(edge[0][0], 2),
                round(edge[0][1], 2)
            ))
            skeleton_graph._points.append((
                round(edge[1][0], 2),
                round(edge[1][1], 2)
            ))

            len_points = len(skeleton_graph._points)

            skeleton_graph._edges.append((len_points - 2, len_points - 1))

        skeleton_graph._deduplicate()

        return skeleton_graph

    @classmethod
    def construct_from_translated_skeletongraphs(cls, translations):
        """ Returns a SkeletonGraph from the translations in the form
        [
            [SkeletonGraph1, [(x1, y1), (x2, y2), ...]],
            [SkeletonGraph2, [(x1, y1), (x2, y2), ...]]
        ]
        """

        #  pylint: disable=protected-access
        skeleton_graph = cls()

        for unit in translations:
            for offset in unit[1]:
                len_points = len(skeleton_graph._points)

                skeleton_graph._points += [
                    (point[0] + offset.x,
                     point[1] + offset.y)
                    for point in unit[0]._points
                ]

                skeleton_graph._edges += [
                    (edge[0] + len_points,
                     edge[1] + len_points)
                    for edge in unit[0]._edges
                ]

        skeleton_graph._deduplicate()

        skeleton_graph._tidy_edges()

        return skeleton_graph


class Graph(SkeletonGraph):
    """ Object to handle points, edges and faces """

    def __init__(self):
        super().__init__()
        self._faces: [[int]] = []
        self._enabled_edges = []

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

    def show(self, **kwargs):  # pylint: disable=signature-differs
        """ Draws edges and or points , applies plt settings and shows plt window """

        self._show(self._enabled_edges, **kwargs)

    # Classmethods
    @classmethod
    def construct_from_skeletongraph(cls, skeleton_graph: SkeletonGraph):
        """ Returns a Graph with detected faces from a SkeletonGraph """
