import queue


class segmentNode:
    def __init__(self):
        self.points = []
        self.start_point = None
        self.end_point = None
        self.branch_id = None
        self.segment_id = None

        self.connect_segments_id = []  # son_segment and next_segment
        self.son_segments = []
        self.father_segment = None
        self.last_segment = None
        self.next_segment = None

    def set_branch_id(self, id):
        self.branch_id = id

    def get_branch_id(self):
        return self.branch_id

    def set_segment_id(self, id):
        self.segment_id = id

    def get_segment_id(self):
        return self.segment_id

    def add_point(self, p):
        assert type(p) is tuple and len(p) == 3
        self.points.append(p)

    def get_points(self):
        return self.points

    def add_connect_segments_id(self, sid):
        self.connect_segments_id.append(sid)

    def get_connect_segments_id(self):
        return self.connect_segments_id

    def add_son_segment(self, n):
        self.son_segments.append(n)

    def get_son_segments(self):
        return self.son_segments

    def set_father_segment(self, n):
        self.father_segment = n

    def get_father_segment(self):
        return self.father_segment

    def set_next_segment(self, n):
        self.next_segment = n

    def get_next_segment(self):
        return self.next_segment

    def set_last_segment(self, n):
        self.last_segment = n

    def get_last_segment(self):
        return self.last_segment


def construct_tree_from_txt(file_path, root_id, root_next_id, root_sons_id):
    with open(file_path, 'r') as f:
        content = f.read()
    branch_sets = content.split("Branch Set")[1:]
    print("number of segments: ", len(branch_sets))
    segment_nodes = {}

    # Initialize segment nodes
    for branch in branch_sets:
        node = segmentNode()
        kid = branch.find(']')
        if kid == 0:
            node.set_branch_id(0)
        else:
            node.set_branch_id(int(branch[:kid]))

        lines = branch.splitlines()
        segment_id = int(lines[0].split(" ")[-1][:-1])
        node.set_segment_id(segment_id)
        # print("branch:", node.get_branch_id(), " segment:", node.get_segment_id())

        lid = 2  # index of line
        while lid < len(lines):
            if lines[lid].find("connects from") != -1:
                lid += 1
                for i, line in enumerate(lines[lid:]):
                    if line.find("Branch Segment") != -1:
                        node.add_connect_segments_id(int(line.split(" ")[-1]))
                    else:
                        lid += (i - 1)
                        break
            if lines[lid].find("connects to") != -1:
                lid += 1
                for i, line in enumerate(lines[lid:]):
                    if line.find("Branch Segment") != -1:
                        node.add_connect_segments_id(int(line.split(" ")[-1]))
                    else:
                        lid += (i - 1)
                        break
            if lines[lid].find("Px") != -1 and lines[lid].find("BNz") != -1:
                lid += 1
                for i, line in enumerate(lines[lid:]):
                    attrs = list(filter(lambda x: x != "", line.split(" ")))
                    if len(attrs) < 12:
                        lid += i
                        break
                    coords = (float(attrs[0]), float(attrs[1]), float(attrs[2]))
                    node.add_point(coords)
            lid += 1
        segment_nodes[segment_id] = node
    print("number of segments: ", len(segment_nodes))

    # Construct tree
    # root = segment_nodes[8]
    # root.add_son_segment(segment_nodes[23])
    # root.add_son_segment(segment_nodes[32])
    # segment_nodes[23].set_father_segment(root)
    # segment_nodes[32].set_father_segment(root)
    # root.set_next_segment(segment_nodes[7])  # prior knowledge
    #
    # son_segs_queue = queue.Queue()
    # son_segs_queue.put(segment_nodes[23])
    # son_segs_queue.put(segment_nodes[32])
    # unvisit_segment_ids = list(segment_nodes.keys())
    # cur_node = segment_nodes[7]
    # unvisit_segment_ids.remove(root.get_segment_id())

    son_segs_queue = queue.Queue()
    root = segment_nodes[root_id]
    for sid in root_sons_id:
        root.add_son_segment(segment_nodes[sid])
        segment_nodes[sid].set_father_segment(root)
        son_segs_queue.put(segment_nodes[sid])
    root.set_next_segment(segment_nodes[root_next_id])  # prior knowledge

    unvisit_segment_ids = list(segment_nodes.keys())
    cur_node = segment_nodes[root_next_id]
    unvisit_segment_ids.remove(root.get_segment_id())

    while True:
        unvisit_segment_ids.remove(cur_node.get_segment_id())

        for cid in cur_node.get_connect_segments_id():
            temp = segment_nodes[cid]
            if temp.get_branch_id() != cur_node.get_branch_id():  # set son nodes
                if temp.get_father_segment() is None and temp.get_segment_id() in unvisit_segment_ids:
                    cur_node.add_son_segment(temp)
                    son_segs_queue.put(temp)
                    temp.set_father_segment(cur_node)
            else:                                                 # set next node
                if temp.get_next_segment() is None and temp.get_father_segment() is None:
                    cur_node.set_next_segment(temp)

        cur_node = cur_node.get_next_segment()
        if cur_node is None:
            if not son_segs_queue.empty():
                cur_node = son_segs_queue.get()
            else:
                break

    return root, segment_nodes


def traverse_tree(root: segmentNode, seg_ids: list):
    print("\n-----Traverse Tree...")

    son_segs_queue = queue.Queue()
    cur_node = root
    s = ''
    new_branch = True
    while True:
        if new_branch:
            if cur_node.get_father_segment() is not None:
                s += "segment %d --> (" % cur_node.get_father_segment().get_segment_id()
            else:
                s += "("
            new_branch = False

        for son_seg in cur_node.get_son_segments():
            son_segs_queue.put(son_seg)

        if cur_node.get_next_segment() is not None:
            s += "segment %d -->" % cur_node.get_segment_id()
            cur_node = cur_node.get_next_segment()
        else:
            s += "segment %d )" % cur_node.get_segment_id()
            print(s)
            s = ''
            new_branch = True
            if not son_segs_queue.empty():
                cur_node = son_segs_queue.get()
            else:
                break


def get_branches_points(root: segmentNode):
    def distance(p1, p2):
        import numpy as np
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    branches_points = []
    branch_points = []
    son_segs_queue = queue.Queue()
    cur_node = root
    while True:
        # plus to front or back
        segment_points = cur_node.get_points()
        if len(branch_points) > 0 and distance(branch_points[-1], segment_points[0]) > distance(branch_points[0], segment_points[-1]):
            segment_points += branch_points
            branch_points = segment_points
        else:
            branch_points += segment_points

        for son_seg in cur_node.get_son_segments():
            son_segs_queue.put(son_seg)

        if cur_node.get_next_segment() is not None:
            cur_node = cur_node.get_next_segment()
        else:
            branches_points.append(branch_points)
            branch_points = []
            if not son_segs_queue.empty():
                cur_node = son_segs_queue.get()
            else:
                break

    return branches_points


if __name__ == '__main__':
    root_segment, segments_dict = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    traverse_tree(root_segment, list(segments_dict.keys()))
    points = get_branches_points(root_segment)