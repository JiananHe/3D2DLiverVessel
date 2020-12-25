class SegmentNode:
    def __init__(self):
        self.points = []
        self.connect_segments_id = []
        self.start_point = None
        self.end_point = None
        self.segment_id = None

    def set_segment_id(self, id):
        self.segment_id = id

    def get_segment_id(self):
        return self.segment_id

    def set_points(self, points):
        self.points = points

    def add_point(self, p):
        assert type(p) is tuple and (len(p) == 3 or len(p) == 2)
        self.points.append(p)

    def get_points(self):
        return self.points

    def add_connect_segments_id(self, sid):
        self.connect_segments_id.append(sid)

    def get_connect_segments_id(self):
        return self.connect_segments_id


class CenterlineGraph:
    def __init__(self):
        self.segment_nodes = {}
        self.adjacent_matrix = None

    def add_segment_node(self, segment_id: int, node: SegmentNode):
        self.segment_nodes[segment_id] = node

    def get_node_num(self):
        return len(self.segment_nodes)

    def get_segment_nodes(self):
        return self.segment_nodes

    def set_adjacent_matrix(self, m):
        self.adjacent_matrix = m