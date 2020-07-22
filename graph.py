""" Module containing the Graph class """


class Graph():
    """ Object to manage the graph layout for the maze gen """

    def __init__(self):
        self._points = []
        self._edges = []
        self._faces = []
