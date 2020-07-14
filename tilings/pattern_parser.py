""" A script to parse all svg files into patterns.json to be read by poly_maze_gen.py """

import os
from xml.dom import minidom
import json

patterns = []

for root, directories, filenames in os.walk(".\\tilings\\svgs"):
    for filename in filenames:
        file = os.path.join(root, filename)

        print("\n", file, sep="\t")

        pattern_type = filename[-5]

        svg = minidom.parse(file)

        edges = []

        refs = []

        min_x = 0  # pylint: disable=invalid-name
        min_y = 0  # pylint: disable=invalid-name
        max_x = 0  # pylint: disable=invalid-name
        max_y = 0  # pylint: disable=invalid-name

        for raw_path in svg.getElementsByTagName("path"):
            path = raw_path.attributes["d"].value

            is_path = not "a" in path

            path = path\
                .replace("M", "")\
                .replace("z", "")\
                .replace("L", "")\
                .replace("a", "")

            print(path)

            points = path.split(" ")

            for i, point in enumerate(points):
                points[i] = list(point.split(","))

                for jnd, j in enumerate(points[i]):
                    if is_path:
                        points[i][jnd] = round(float(j) * 10, 3) + 0

                    else:
                        points[i][jnd] = float(j) * 10

                if is_path:
                    min_x = min(points[i][0], min_x)
                    min_y = min(points[i][1], min_y)
                    max_x = max(points[i][0], max_x)
                    max_y = max(points[i][1], max_y)

            #print("points =", points)

            if is_path:
                print("path")
                for i in range(len(points) - 1):
                    edges.append(
                        sorted(sorted((points[i], points[i + 1]),
                                      key=lambda point: point[0]),
                               key=lambda point: point[1]))

            else:
                print("ref", points)

                refs.append([
                    round(points[0][0] + points[1][0], 3) + 0,
                    round(points[0][1], 3) + 0
                ])

        refs.sort(key=lambda point: point[0])

        if pattern_type == "a":
            patterns.append({
                "name": filename[:-5],
                "refs": refs,
                "unit": {
                    "min_x": min_x,
                    "min_y": min_y,
                    "edges": edges
                },
                "bottom_unit": {
                    "max_x": max_x,
                    "max_y": max_y,
                    "edges": edges
                },
                "right_unit": {
                    "max_x": max_x,
                    "max_y": max_y,
                    "edges": edges
                }
            })

        elif pattern_type == "b":
            patterns[-1]["bottom_unit"] = {
                "max_x": max_x,
                "max_y": max_y,
                "edges": edges
            }

        elif pattern_type == "r":
            patterns[-1]["right_unit"] = {
                "max_x": max_x,
                "max_y": max_y,
                "edges": edges
            }

        elif pattern_type == "d":
            patterns[-1]["bottom_unit"] = {
                "max_x": max_x,
                "max_y": max_y,
                "edges": edges
            }

            patterns[-1]["right_unit"] = {
                "max_x": max_x,
                "max_y": max_y,
                "edges": edges
            }

open("patterns.json", "w").write(json.dumps(patterns, indent=2))
