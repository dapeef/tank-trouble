""" Testing suite for the maze generation (graph2.py) """

import json
from graph2 import SkeletonGraph, Graph


patterns = json.loads(open("patterns.json").read())


def test_create_skeleton_graph():
    """ Tests that the skeleton graph is created properly """

    graph = SkeletonGraph.construct_from_edges(patterns[1]["unit"]["edges"])

    assert True


def compare_skeleton_graphs(skeleton_graph1, skeleton_graph2):
    """ Takes 2 skeleton graphs and compares them """

    return False
