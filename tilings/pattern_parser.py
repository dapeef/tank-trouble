""" poo """

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
                    points[i][jnd] = str(round(float(j) * 10, 2) + 0)

            print("points =", points)

            if is_path:
                print("path")
                for i in range(len(points) - 1):
                    edges.append((points[i], points[i + 1]))

            else:
                print("ref")

                refs.append(points[0])

        refs.sort(key=lambda point: point[0])

        if pattern_type == "a":
            patterns.append({
                "name": filename[:-5],
                "refs": refs,
                "unit_edges": edges
            })

        elif pattern_type == "b":
            patterns[-1]["bottom_unit_edges"] = edges

        elif pattern_type == "r":
            patterns[-1]["right_unit_edges"] = edges

        elif pattern_type == "d":
            patterns[-1]["dual_unit_edges"] = edges

open("patterns.json", "w").write(json.dumps(patterns, indent=2))
