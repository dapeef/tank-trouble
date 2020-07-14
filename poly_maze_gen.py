""" New module to handle maze generation based on tesselating patterns """

import random
import json
import math
import vectormath as vmath
import matplotlib.pyplot as plt

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

    raw_edges = []

    for x in range(reps_x):
        vec_x = vmath.Vector2(ref_x).as_percent(x)

        for y in range(reps_y):
            vec_y = vmath.Vector2(ref_y).as_percent(y)

            offset = vec_x + vec_y + shift_vec
            # print(x, y, offset)

            for edge in pattern["unit"]["edges"]:
                raw_edges.append(round_edge(transpose_edge(edge, offset), 2))

    edges = []

    for i in raw_edges:
        if not i in edges:
            edges.append(i)

    print(len(raw_edges), len(edges))

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


def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


def display_edges(edges):
    """ A function which displays the edges in an array using matplotlib """

    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [-edge[0][1], -edge[1][1]])
    plt.show()


if __name__ == "__main__":
    display_edges(generate_maze(20, 20, pattern_id=0, density=0.9))
