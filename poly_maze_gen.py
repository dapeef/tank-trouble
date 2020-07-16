""" New module to handle maze generation based on tesselating patterns """

import json
import math
import vectormath as vmath
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

# Maze id constants
TRIANGLE = 0
SQUARE = 1
TALL_SQUARE_TRIANGLE = 2
_SQUARE_ANGLED = 3
PENTAGON = 4
HEXAGON = 5
_PADDED_SQUARE = 6
_PADDED_HEXAGON = 7
HEXAGON_PADDED_HEXAGON = 8
_OCTAGON = 9
_NONAGON = 10
HEART = 11
PADDED_DODECAGON = 12
_DODECAGON = 13

# Read json
patterns = json.loads(open("patterns.json").read())


def generate_maze(width, height, pattern_id=0, density=0.9):
    """ Generate a maze of given width and height in the pattern outlined in patterns.json """

    # Make variables more handy
    pattern = patterns[pattern_id]

    ref_x = pattern["refs"][1]
    ref_y = pattern["refs"][0]

    shift_vec = vmath.Vector2([
        - pattern["unit"]["min_x"],
        - pattern["unit"]["min_y"]
    ])

    # Calculate repetitions
    reps_x = max(math.floor((width - pattern["right_unit"]["max_x"] +
                             pattern["unit"]["min_x"]) / ref_x[0]), 0)
    reps_y = max(math.floor((height - pattern["bottom_unit"]["max_y"] +
                             pattern["unit"]["min_y"]) / ref_y[1]), 0)

    # print(pattern)

    print(reps_x, reps_y, "\t", ref_x, ref_y)

    # Add all general units
    raw_edges = []

    for x in range(reps_x):
        for y in range(reps_y):
            mod_x = 0
            mod_y = 0

            while round(offset_calc(ref_x, ref_y, x + 1 + mod_x, y + mod_y, shift_vec).x +
                        pattern["right_unit"]["max_x"], 2) > width:
                # print("eek x")

                mod_x -= reps_x

            while round(offset_calc(ref_x, ref_y, x + mod_x, y + 1 + mod_y, shift_vec).y +
                        pattern["bottom_unit"]["max_y"], 2) > height:
                # print("eek y")

                mod_y -= reps_y

            offset = offset_calc(ref_x, ref_y, x + mod_x, y + mod_y, shift_vec)

            print(offset)

            if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                    round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
                for edge in pattern["unit"]["edges"]:
                    raw_edges.append(
                        round_edge(
                            transpose_edge(
                                edge,
                                offset
                            ),
                            2
                        )
                    )

    # Add bottom units
    for x in range(reps_x):
        mod_x = 0
        mod_y = 0

        while round(offset_calc(ref_x, ref_y, x, reps_y + mod_y, shift_vec).y +
                    pattern["bottom_unit"]["max_y"], 2) > height:
            # print("bottom eek y")

            mod_y -= 1

        while round(offset_calc(ref_x, ref_y, x + 1 + mod_x, reps_y + mod_y, shift_vec).x +
                    pattern["right_unit"]["max_x"], 2) > width:
            # print("bottom eek x")

            mod_x -= (reps_x)

        offset = offset_calc(
            ref_x, ref_y,
            x + mod_x, reps_y + mod_y,
            shift_vec
        )

        # print(x, offset.x, offset.y)

        if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
            for edge in pattern["bottom_unit"]["edges"]:
                raw_edges.append(round_edge(transpose_edge(edge, offset), 2))

    # Add right units
    for y in range(reps_y + 1):
        mod_x = 0
        mod_y = 0

        while round(offset_calc(ref_x, ref_y, reps_x + mod_x, y, shift_vec).x +
                    pattern["right_unit"]["max_x"], 2) > width:
            # print("right eek x")

            mod_x -= 1

        while round(offset_calc(ref_x, ref_y, reps_x + mod_x, y + mod_y, shift_vec).y +
                    pattern["bottom_unit"]["max_y"], 2) > height:
            # print("right eek y")

            mod_y -= (reps_y + 1)

        offset = offset_calc(
            ref_x, ref_y,
            reps_x + mod_x, y + mod_y,
            shift_vec
        )

        # print(x, offset.x, offset.y)

        if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
            if y != reps_y:
                unit_edges = pattern["right_unit"]["edges"]

            else:
                unit_edges = pattern["corner_unit"]["edges"]

            for edge in unit_edges:
                raw_edges.append(round_edge(transpose_edge(edge, offset), 2))

    # Remove duplicates
    edges = []

    for i in raw_edges:
        if not i in edges:
            edges.append(i)

    # print(len(raw_edges), len(edges))

    points = []
    for edge in edges:
        points.append(edge[0])
        points.append(edge[1])

    np_points = np.array(points)

    tris = Delaunay(np_points)

    plt.triplot(np_points[:, 0], np_points[:, 1], tris.simplices)
    plt.show

    return edges


def transpose_edge(edge, vector):
    """ A function to transpose an edge by a given x and y value """

    new_vec_edge = [
        vmath.Vector2(edge[0]) + vector,
        vmath.Vector2(edge[1]) + vector
    ]

    return [[new_vec_edge[0].x, new_vec_edge[0].y], [new_vec_edge[1].x, new_vec_edge[1].y]]


def round_edge(edge, dec_points=0):
    """ Rounds all points in an edge to a given number of decimal points """

    return [
        [round(edge[0][0], dec_points), round(edge[0][1], dec_points)],
        [round(edge[1][0], dec_points), round(edge[1][1], dec_points)]
    ]


def offset_calc(ref_x, ref_y, x_scale, y_scale, shift_vec):
    """ Calculates the offset for the new unit """

    vec_x = vmath.Vector2(ref_x).as_percent(x_scale)
    vec_y = vmath.Vector2(ref_y).as_percent(y_scale)

    return vec_x + vec_y + shift_vec


def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


def display_edges(edges, bounding_box=(0, 0)):
    """ A function which displays the edges in an array using matplotlib """

    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [-edge[0][1], -edge[1][1]])

    if bounding_box != (0, 0):
        plt.plot(
            [0, 0, bounding_box[0], bounding_box[0], 0],
            [0, -bounding_box[1], -bounding_box[1], 0, 0],
            "k:",
            linewidth=1
        )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    print(num_patterns())

    # display_edges(patterns[3]["right_unit"]["edges"])

    w, h = 40, 20

    display_edges(
        generate_maze(
            w, h,
            pattern_id=2,
            density=0.9),
        bounding_box=(w, h)
    )
