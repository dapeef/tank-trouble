""" New module to handle maze generation based on tesselating patterns """

import json
import math
import time
import vectormath as vmath
from graph2 import SkeletonGraph, Graph

# Maze id constants
TRIANGLE = 0
SQUARE = 1
TALL_SQUARE_TRIANGLE = 2
SQUARE_ANGLED = 3
PENTAGON = 4
HEXAGON = 5
SQUARE_PADDED = 6
HEXAGON_PADDED = 7
HEXAGON_PADDED_HEXAGON = 8
OCTAGON = 9
NONAGON = 10
HEART = 11
DODECAGON_PADDED = 12
DODECAGON = 13

# Read json
patterns = json.loads(open("patterns.json").read())


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def generate_maze(width, height, pattern_id=0, density=0.9):
    """
    Generate a Graph object of a maze of given width and height in the pattern outlined in
    patterns.json
    """

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

    # Create graphs for units
    general_unit = SkeletonGraph.construct_from_edges(
        pattern["unit"]["edges"])

    bottom_unit = SkeletonGraph.construct_from_edges(
        pattern["bottom_unit"]["edges"])

    right_unit = SkeletonGraph.construct_from_edges(
        pattern["right_unit"]["edges"])

    corner_unit = SkeletonGraph.construct_from_edges(
        pattern["corner_unit"]["edges"])

    positions = [
        [general_unit, []],
        [bottom_unit, []],
        [right_unit, []],
        [corner_unit, []]
    ]

    # Add all general units
    for x in range(reps_x):
        for y in range(reps_y):
            mod_x = 0
            mod_y = 0

            while round(offset_calc(ref_x, ref_y, x + 1 + mod_x, y + mod_y, shift_vec).x +
                        pattern["right_unit"]["max_x"], 2) > width:
                mod_x -= reps_x

            while round(offset_calc(ref_x, ref_y, x + mod_x, y + 1 + mod_y, shift_vec).y +
                        pattern["bottom_unit"]["max_y"], 2) > height:
                mod_y -= reps_y

            offset = offset_calc(ref_x, ref_y, x + mod_x, y + mod_y, shift_vec)

            if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                    round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
                positions[0][1].append(offset)

    # Add bottom units
    for x in range(reps_x):
        mod_x = 0
        mod_y = 0

        while round(offset_calc(ref_x, ref_y, x, reps_y + mod_y, shift_vec).y +
                    pattern["bottom_unit"]["max_y"], 2) > height:
            mod_y -= 1

        while round(offset_calc(ref_x, ref_y, x + 1 + mod_x, reps_y + mod_y, shift_vec).x +
                    pattern["right_unit"]["max_x"], 2) > width:
            mod_x -= (reps_x)

        offset = offset_calc(
            ref_x, ref_y,
            x + mod_x, reps_y + mod_y,
            shift_vec
        )

        if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
            positions[1][1].append(offset)

    # Add right and corner units
    for y in range(reps_y + 1):
        mod_x = 0
        mod_y = 0

        while round(offset_calc(ref_x, ref_y, reps_x + mod_x, y, shift_vec).x +
                    pattern["right_unit"]["max_x"], 2) > width:
            mod_x -= 1

        while round(offset_calc(ref_x, ref_y, reps_x + mod_x, y + mod_y, shift_vec).y +
                    pattern["bottom_unit"]["max_y"], 2) > height:
            mod_y -= (reps_y + 1)

        offset = offset_calc(
            ref_x, ref_y,
            reps_x + mod_x, y + mod_y,
            shift_vec
        )

        if round(offset.x + pattern["unit"]["min_x"], 2) >= 0 and \
                round(offset.y + pattern["unit"]["min_y"], 2) >= 0:
            if y != reps_y:
                unit = 2

            else:
                unit = 3

            positions[unit][1].append(offset)

    skeleton_graph = SkeletonGraph.construct_from_translated_skeletongraphs(
        positions)

    #graph = Graph.construct_from_skeletongraph(skeleton_graph)

    # graph.make_maze(density)

    print("finished in", round(time.time() - start_time, 3), "seconds")

    return skeleton_graph


# Utilities
def offset_calc(ref_x, ref_y, x_scale, y_scale, shift_vec):
    """ Calculates the offset for the new unit """

    vec_x = vmath.Vector2(ref_x).as_percent(x_scale)
    vec_y = vmath.Vector2(ref_y).as_percent(y_scale)

    return vec_x + vec_y + shift_vec


# Debugging fuctions
def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


if __name__ == "__main__":
    print("num patterns =", num_patterns(), "\n")
    start_time = time.time()
    start = time.time()
    print("start")

    w, h = 20, 20

    PROFILING = False

    if not PROFILING:
        generate_maze(
            w, h,
            pattern_id=3,
            density=0.9).show(flip=True, bounding_box=[w, h])

    else:
        for i in range(num_patterns()):
            start_time = time.time()
            print(i, "/", num_patterns() - 1, end=" ", sep="")
            generate_maze(
                w, h,
                pattern_id=i,
                density=0.9)

        print("Total time:", round(time.time() - start, 3), "seconds", sep=" ")
