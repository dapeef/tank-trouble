""" New module to handle maze generation based on tesselating patterns """

import json
import math
import time
import vectormath as vmath
from graph import Graph

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

    # Create base Graph
    graph = Graph()

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
    general_unit = Graph.create_graph_from_json_edges(
        pattern["unit"]["edges"])

    bottom_unit = Graph.create_graph_from_json_edges(
        pattern["bottom_unit"]["edges"])

    right_unit = Graph.create_graph_from_json_edges(
        pattern["right_unit"]["edges"])

    corner_unit = Graph.create_graph_from_json_edges(
        pattern["corner_unit"]["edges"])

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
                graph.combine_graph_edges_into_graph(
                    Graph.translated(general_unit, offset.x, offset.y)
                )

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
            graph.combine_graph_edges_into_graph(
                Graph.translated(bottom_unit, offset.x, offset.y)
            )

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
                unit = right_unit

            else:
                unit = corner_unit

            graph.combine_graph_edges_into_graph(
                Graph.translated(unit, offset.x, offset.y)
            )

    graph.detect_faces()

    print("finished in", round(time.time() - start_time, 3), "seconds")

    return graph


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
    print("start")

    w, h = 8, 4

    generate_maze(
        w, h,
        pattern_id=DODECAGON,
        density=0.9).show(flip=True, bounding_box=[w, h])
