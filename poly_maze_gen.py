""" New module to handle maze generation based on tesselating patterns """

import random
import json
import math

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


def num_patterns():
    """
    Return the number of patterns available (with the intention of being able to automate the
    pattern choosing)
    """
    return len(patterns)


if __name__ == "__main__":
    generate_maze(10, 10, pattern=0, density=0.9)
