import numpy as np
from centerline_show import show_branches_3d
from centerline_graph import SegmentNode, CenterlineGraph


def construct_graph_from_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    branch_sets = content.split("Branch Set")[1:]
    print("number of segments: ", len(branch_sets))
    centerline_graph = CenterlineGraph()

    # Initialize segment nodes
    for branch in branch_sets:
        node = SegmentNode()

        lines = branch.splitlines()
        segment_id = int(lines[0].split(" ")[-1][:-1])
        node.set_segment_id(segment_id)

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
        centerline_graph.add_segment_node(node.get_segment_id(), node)
    segment_num = centerline_graph.get_node_num()
    assert segment_num == len(branch_sets)

    # construct adjacent matrix
    adj_matrix = np.eye(segment_num, dtype=np.bool)
    segment_nodes = centerline_graph.get_segment_nodes()
    segment_ids = list(segment_nodes.keys())
    for i, seg_id in enumerate(segment_ids):
        seg_node = segment_nodes[seg_id]
        adj_seg_ids = seg_node.get_connect_segments_id()
        for adj_seg_id in adj_seg_ids:
            j = segment_ids.index(adj_seg_id)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    centerline_graph.set_adjacent_matrix(adj_matrix)

    return centerline_graph


if __name__ == '__main__':
    centerline_graph = construct_graph_from_txt("/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks_Centerline/0_left.txt")
    segment_nodes = centerline_graph.get_segment_nodes()
    all_points = []
    segments_length = []
    for segment_node in segment_nodes.values():
        segment_points = segment_node.get_points()
        all_points += segment_points
        segments_length.append(len(segment_points))

    all_points = np.array(all_points)
    show_branches_3d(all_points, segments_length, (512, 512), vessel_stl=["/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks_STL/0_right.stl"], fix_color=True)

    # from centerline_projector import *

    # plane_centerline = projector_main(all_points, segments_length, 0, 0, 0, -30, 0, 0,
    #                                   1000, 765, 512, 512, 0.37, 0.37)
    # show_branches_2d(plane_centerline)
