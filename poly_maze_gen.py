""" New module to handle maze generation based on tesselating patterns """

import json
import math
import vectormath as vmath
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import time

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

            # print(offset)

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

    # Add right and corner units
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

    # Implement Delaunay
    points = []
    for edge in edges:
        if not edge[0] in points:
            points.append(edge[0])

        if not edge[1] in points:
            points.append(edge[1])

    np_points = np.array(points)

    tris = Delaunay(np_points)

    #plt.triplot(np_points[:, 0], np_points[:, 1], tris.simplices)
    plt.plot(
        np_points[tris.simplices][0][:, 0],
        np_points[tris.simplices][0][:, 1],
        'o'
    )

    tri_ids = list(range(len(tris.simplices)))

    tri_groups = []

    # for i, tri in enumerate(tris.simplices):
    #    print(i, tri, np_points[tris.simplices][i], tris.neighbors[i])

    while len(tri_ids) > 0:
        tri_group = find_adjacent_tris(
            tri_ids[0],
            [tri_ids[0]],
            tris,
            np_points,
            edges
        )

        print("group", tri_group)

        face_edges = [
            sort_edge([
                list(np_points[tris.simplices][tri][i]),
                list(np_points[tris.simplices][tri][(i + 1) % 3])
            ])
            for tri in tri_group
            for i in range(3)
        ]

        exterior_face_edges = []

        for edge in face_edges:
            if not edge in exterior_face_edges:
                exterior_face_edges.append(edge)

            else:
                exterior_face_edges.pop(exterior_face_edges.index(edge))

        valid = True

        for edge in exterior_face_edges:
            if not edge in edges:
                valid = False
                print("INVALID")

        if valid:
            draw_edges(exterior_face_edges, flip=False)

        tri_ids = list(filter(lambda x: x not in tri_group, tri_ids))

    print(tri_groups)

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


def find_adjacent_tris(tri, tri_group, tris, np_points, edges):
    """ Recursive function to find all tris that make up a face """

    tri_points = [list(i) for i in np_points[tris.simplices][tri]]

    # print("\n")

    for i in tris.neighbors[tri]:
        if not i in tri_group and i != -1:
            i_points = [list(i) for i in np_points[tris.simplices][i]]

            edge = sort_edge(
                list(filter(lambda point: point in tri_points, i_points))
            )

            #print("edge", edge)

            if not edge in edges:
                tri_group.append(i)
                find_adjacent_tris(i, tri_group, tris, np_points, edges)

    return tri_group


def sort_edge(edge):
    """ Function to sort an edge so all edges are consistent and comparable """

    return sorted(
        sorted(
            edge,
            key=lambda point: point[0]),
        key=lambda point: point[1]
    )


def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


def draw_edges(edges, bounding_box=(0, 0), flip=True):
    """ A function which displays the edges in an array using matplotlib """

    if flip:
        flip_sf = -1

    else:
        flip_sf = 1

    for edge in edges:
        plt.plot(
            [edge[0][0], edge[1][0]],
            [edge[0][1]*flip_sf, edge[1][1]*flip_sf]
        )

    if bounding_box != (0, 0):
        plt.plot(
            [0, 0, bounding_box[0], bounding_box[0], 0],
            [0, -bounding_box[1], -bounding_box[1], 0, 0],
            "k:",
            linewidth=1
        )
    plt.gca().set_aspect('equal', adjustable='box')


if __name__ == "__main__":
    print("num patterns =", num_patterns(), "\n")
    start_time = time.time()
    print("start")

    # draw_edges(patterns[3]["right_unit"]["edges"])

    w, h = 10, 5

    draw_edges(
        generate_maze(
            w, h,
            pattern_id=DODECAGON,
            density=0.9),
        bounding_box=(w, h)
    )

    print("finished in", round(time.time() - start_time, 3), "seconds")

    plt.show()
