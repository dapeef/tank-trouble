""" New module to handle maze generation based on tesselating patterns """

import random
import json
import math
import vectormath as vmath

# Read json
patterns = json.loads(open("patterns.json").read())


def generate_maze(width, height, pattern_id=0, density=0.9):
    """ Generate a maze of given width and height in the pattern outlined in patterns.json """

    # Make variables more handy
    pattern = patterns[pattern_id]

    ref_x = pattern["refs"][1]
    ref_y = pattern["refs"][0]

    # Calculate repetitions
    reps_x = max(math.floor((width - pattern["right_unit"]
                             ["max_x"] + pattern["unit"]["min_x"]) / ref_x[0]), 0)
    reps_y = max(math.floor((height - pattern["bottom_unit"]
                             ["max_y"] + pattern["unit"]["min_y"]) / ref_y[1]), 0)

    print(pattern)

    print(reps_x, reps_y)

    raw_edges = []

    for x in range(reps_x):
        vec_x = vmath.Vector2(ref_x).as_percent(x)

        for y in range(reps_y):
            vec_y = vmath.Vector2(ref_y).as_percent(y)

            offset = vec_x + vec_y
            print(vec_x, vec_y, offset)

            for edge in pattern["unit"]["edges"]:
                print(edge, offset, transpose_edge(edge, offset))
                raw_edges.append(transpose_edge(edge, offset))


def transpose_edge(edge, vector):
    """ A function to transpose an edge by a given x and y value """

    new_vec_edge = [
        vmath.Vector2(edge[0]) + vector,
        vmath.Vector2(edge[1]) + vector
    ]

    return [[new_vec_edge[0].x, new_vec_edge[0].y], [new_vec_edge[1].x, new_vec_edge[1].y]]


def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


if __name__ == "__main__":
    generate_maze(5, 5, pattern_id=1, density=0.9)
