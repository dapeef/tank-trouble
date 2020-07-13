""" poo """

import os
from xml.dom import minidom

for root, directories, filenames in os.walk(".\\tilings\\svgs"):
    for filename in filenames:
        file = os.path.join(root, filename)

        print(file)

        svg = minidom.parse(file)

        for i in svg.getElementsByTagName("path"):
            print(i.attributes["d"].value)
