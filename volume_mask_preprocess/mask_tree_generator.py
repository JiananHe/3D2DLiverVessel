import numpy as np
import cv2
import itk
import os
import SimpleITK as sitk
from scipy import ndimage
import skimage.morphology as morphology
from centerline_show import show_branches_3d
np.random.seed(10)


def find_neighbor(p, image):
    N = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    direct_neighbors = []
    nidx = np.zeros((6,), dtype=np.int)
    for i, n in enumerate(N):
        if image[tuple(p + n)]:
            direct_neighbors.append(p + n)
            nidx[i] = p[i//2]
        else:
            nidx[i] = p[i//2] + n[i//2]

    diagonal_neighbors = np.argwhere(image[nidx[0]:nidx[1]+1, nidx[2]:nidx[3]+1, nidx[4]:nidx[5]+1] == 1) + p - 1
    diagonal_neighbors = diagonal_neighbors[np.any(diagonal_neighbors != p, axis=1)]
    return np.array(direct_neighbors + list(diagonal_neighbors))


def find_start(image):
    sk_points = np.argwhere(image == 1)
    start_point = None
    for p in sk_points:
        if len(find_neighbor(p, image)) == 1:
            start_point = p
            break

    return start_point


def skeletonize_mask_volume(mask_path):
    mask_image = sitk.ReadImage(mask_path)
    origin = mask_image.GetOrigin()
    spacing = mask_image.GetSpacing()
    print(spacing)

    mask_array = sitk.GetArrayFromImage(mask_image)
    print(np.unique(mask_array))
    sk_image = morphology.skeletonize(mask_array)
    sk_image = np.pad(sk_image, 1)

    sk_points = np.argwhere(sk_image == 1)
    num_points = len(sk_points)
    start_point = find_start(sk_image)

    neighbFilter6 = np.zeros((3, 3, 3))
    neighbFilter6[0, 1, 1] = 1
    neighbFilter6[1, 0, 1] = 1
    neighbFilter6[1, 1, 0] = 1
    neighbFilter6[1, 1, 2] = 1
    neighbFilter6[1, 2, 1] = 1
    neighbFilter6[2, 1, 1] = 1
    types = set()
    for p in sk_points:
        pointMatrix = np.array(sk_image[p[0]-1:p[0]+2, p[1]-1:p[1]+2, p[2]-1:p[2]+2], copy=True)
        pointMatrix[1, 1, 1] = 0
        verticeNumber = np.count_nonzero(pointMatrix)
        edgeMap = pointMatrix * ndimage.convolve(pointMatrix, neighbFilter6, mode='constant', cval=0.0)
        edgeNumber = np.sum(edgeMap) / 2
        types.add((verticeNumber, edgeNumber))
    print(types)

    all_segment = []
    branch_point = []
    count = 0

    segment = [list(start_point)]
    sk_image[tuple(start_point)] = 0

    neighbors = find_neighbor(start_point, sk_image)

    while count <= num_points:
        if len(neighbors) == 1:
            start_point = neighbors[0]
            segment.append(list(start_point))
        elif len(neighbors) >= 2:
            if len(segment) == 1:
                branch_point += list(neighbors)
                start_point = branch_point.pop(-1)
                segment.append(list(start_point))
            else:
                all_segment.append(segment)
                branch_point += list(neighbors)
                start_point = branch_point.pop(-1)
                segment = [list(start_point)]
        elif len(neighbors) == 0:
            all_segment.append(segment)
            if len(branch_point) > 0:
                start_point = branch_point.pop(-1)
            else:
                start_point = find_start(sk_image)
                if start_point is None:
                    break
            segment = [list(start_point)]

        if sk_image[tuple(start_point)]:
            sk_image[tuple(start_point)] = 0
            count += 1
        neighbors = find_neighbor(start_point, sk_image)

    print(len(all_segment))
    print(sum(len(s) for s in all_segment))
    print(len(sk_points))

    segment_points = []
    segment_idx = []
    for s in all_segment:
        segment_points += s
        segment_idx.append(len(s))

    segment_points = np.array(segment_points)
    segment_points[:, [0, 1, 2]] = segment_points[:, [2, 1, 0]]
    segment_points = origin + segment_points * spacing
    return segment_points, segment_idx


def findBranchPoints(skeleton, return_image=False):
    pixelPoints = np.argwhere(skeleton)
    neighbFilter4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbFilter8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    branchPoints = np.zeros((1, 2))
    branchImg = np.zeros(skeleton.shape)
    endPoints = np.zeros((1, 2))
    endImg = np.zeros(skeleton.shape)
    skeletonTemp = np.copy(skeleton)
    for selectedPoint in pixelPoints:
        pointMatrix = np.array(
            skeleton[selectedPoint[0] - 1:selectedPoint[0] + 2, selectedPoint[1] - 1:selectedPoint[1] + 2], copy=True)
        pointMatrix[1, 1] = 0
        verticeNumber = np.count_nonzero(pointMatrix)
        edgeMap = pointMatrix * ndimage.convolve(pointMatrix, neighbFilter4, mode='constant', cval=0.0)
        edgeNumber = np.sum(edgeMap) / 2
        euilerNumber = verticeNumber - edgeNumber
        if (euilerNumber > 2):
            branchPoints = np.vstack((branchPoints, selectedPoint))
            branchImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = 2
        elif ((euilerNumber == 1) & (verticeNumber < 5)):
            endPoints = np.vstack((endPoints, selectedPoint))
            endImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = -1
        elif ((euilerNumber == 2) & (verticeNumber >= 4)):
            connectedTrees4, connectedTrees4Num = ndimage.label(pointMatrix, neighbFilter4)
            connectedTrees8, connectedTrees8Num = ndimage.label(pointMatrix, neighbFilter8)
            label4, verticesNumberTrees = np.unique(connectedTrees4[connectedTrees4 > 0], return_counts=True)
            cornerCondition = (np.sum(pointMatrix[0:2, 0:2]) == 3) | (np.sum(pointMatrix[1:3, 1:3]) == 3) | (
                    np.sum(pointMatrix[1:3, 0:2]) == 3) | (np.sum(pointMatrix[0:2, 1:3]) == 3)
            if ((abs(verticesNumberTrees[0] - verticesNumberTrees[1]) >= 2) & cornerCondition & (
                    connectedTrees8Num > 1)):
                branchPoints = np.vstack((branchPoints, selectedPoint))
                branchImg[selectedPoint[0], selectedPoint[1]] = 1
                skeletonTemp[selectedPoint[0], selectedPoint[1]] = 2
    branchPoints = branchPoints[1:, :].astype('int64')
    endPoints = endPoints[1:, :].astype('int64')
    if return_image:
        skeletonGraphPointsImg = np.tile(skeleton, (3, 1, 1))
        skeletonGraphPointsImg = skeletonGraphPointsImg + np.stack(
            (np.zeros(skeleton.shape), -branchImg, -branchImg)) + np.stack((-endImg, np.zeros(skeleton.shape), -endImg))
        skeletonGraphPointsImg = np.moveaxis(skeletonGraphPointsImg, 0, -1)
        return branchPoints, endPoints, skeletonTemp, skeletonGraphPointsImg
    else:
        return branchPoints, endPoints, skeletonTemp


def skeleton2Graph(branch_points, end_points, skeleton):
    skeleton_temp = np.copy(skeleton)
    edge_list = []
    branch_stack = []
    edge = []

    connect_in_search = np.zeros([0, 2])
    branch_in_search = np.zeros([0, 2])
    cur_point = tuple(end_points[0])
    while cur_point is not None or len(branch_stack) > 0:
        print(cur_point)
        if cur_point is None:
            cur_point = branch_stack.pop(0)

        skeleton[cur_point] = 0
        search_matrix = skeleton[cur_point[0]-1:cur_point[0]+2, cur_point[1]-1:cur_point[1]+2]
        end_in_search = np.argwhere(search_matrix == -1) + cur_point - 1
        if len(edge) == 1:
            connect_in_search_last = connect_in_search
            connect_in_search_new = np.argwhere(search_matrix == 1) + cur_point - 1
            connect_in_search_new = set(tuple(i) for i in connect_in_search_new) - \
                                    set(tuple(i) for i in connect_in_search_last)
            connect_in_search = np.array([list(i) for i in connect_in_search_new])

            branch_in_search_last = branch_in_search
            branch_in_search_new = np.argwhere(search_matrix == 2) + cur_point - 1
            branch_in_search_new = set(tuple(i) for i in branch_in_search_new) - \
                                    set(tuple(i) for i in branch_in_search_last)
            branch_in_search = np.array([list(i) for i in branch_in_search_new])
        else:
            connect_in_search = np.argwhere(search_matrix == 1) + cur_point - 1
            branch_in_search = np.argwhere(search_matrix == 2) + cur_point - 1

        edge.append(cur_point)

        # an end point.
        if skeleton_temp[cur_point] == -1:
            if len(edge) != 1:  # end of an edge
                # assert len(branch_in_search) == 0 and len(connect_in_search) == 0
                edge_list.append(edge)
                edge = []
                cur_point = None
            else:  # start of an edge
                assert len(branch_in_search) + len(connect_in_search) == 1
                cur_point = tuple(connect_in_search[0]) if len(connect_in_search) == 1 else tuple(branch_in_search[0])

        # a connect point
        elif skeleton_temp[cur_point] == 1:
            if cur_point == tuple([360, 213]):
                print()

            if len(branch_in_search) == 0:  # not adjacent to a branch point
                assert len(connect_in_search) + len(end_in_search) == 1
                cur_point = tuple(end_in_search[0]) if len(end_in_search) != 0 else tuple(connect_in_search[0])
            else:                           # adjacent to a branch point
                if edge[0] != tuple(branch_in_search[0]):  # reach to a new branch point
                    cur_point = tuple(branch_in_search[0])
                else:                                      # start from a branch point
                    assert len(edge) == 2
                    branch_in_search = set(tuple(i) for i in branch_in_search)
                    branch_in_search.remove(tuple(edge[0]))
                    branch_in_search = np.array([list(i) for i in branch_in_search])
                    if len(branch_in_search) == 0:
                        assert len(connect_in_search) + len(end_in_search) == 1
                        cur_point = tuple(connect_in_search[0]) if len(connect_in_search) == 1 else tuple(end_in_search[0])
                    else:
                        # may reach to >=1 branch points
                        branch_distance = np.sum((branch_in_search - cur_point) ** 2, axis=1)
                        cur_point = tuple(branch_in_search[np.argmin(branch_distance)])

        # a branch point
        elif skeleton_temp[cur_point] == 2:
            if cur_point == tuple([264, 479]):
                print()

            num_adj_nonbranch = len(connect_in_search) + len(end_in_search)
            num_adj_all = num_adj_nonbranch + len(branch_in_search)
            if len(edge) > 1:    # branch point is the end of the edge
                if num_adj_nonbranch > 0:  # this branch point will be visited in the future
                    skeleton[cur_point] = 2
                    branch_stack.append(cur_point)
                edge_list.append(edge)
                edge = []
                cur_point = None
            else:                # branch point is the start of the edge
                if num_adj_all > 0:
                    if num_adj_all > 1:    # this branch point will be visited in the future
                        skeleton[cur_point] = 2
                        branch_stack.append(cur_point)
                    adj_point = np.vstack((connect_in_search, end_in_search, branch_in_search))
                    cur_point = tuple(adj_point[0])
                else:
                    edge = []
                    cur_point = None

        else:
            raise TypeError

    return edge_list


if __name__ == '__main__':
    # mask_dir = '/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks/'
    # files = list(filter(lambda f: f.split(".")[-1] == "nrrd", os.listdir(mask_dir)))

    # segment_points, segment_idx = skeletonize_mask_volume('/home/hja/Projects/3'
    #                                                       'D2DRegister/ASOCA/Train_Masks/32.nrrd')
    # show_branches_3d(segment_points, segment_idx, (512, 512))

    # import os
    # dsa_dir = "/home/hja/Projects/3D2DRegister/HeartDSA(高雨枫)/data/data/left/train/"
    # mask_dir = "/home/hja/Projects/3D2DRegister/HeartDSA(高雨枫)/data/data/left/label/"
    # for f in os.listdir(mask_dir)[19:]:
    #     print(f)

    # dsa = cv2.imread(os.path.join(dsa_dir, f), 0)
    # cv2.imshow("dsa", dsa)
    mask = cv2.imread("/home/hja/Projects/3D2DRegister/Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1_seg.jpg", 0)
    mask[mask > 200] = 255
    mask[mask <= 200] = 0
    mask = np.pad(mask, 1)
    cv2.imshow("mask", mask)

    mask = np.array(mask / 255, np.uint8)
    sk_image = morphology.skeletonize(mask, method='lee') + .0
    cv2.imshow("skeleton", sk_image)

    branchPoints, endPoints, skeletonTemp, BranchPoints = findBranchPoints(sk_image, return_image=True)
    cv2.imshow("BranchPoints", np.array(BranchPoints * 255, dtype=np.uint8))

    edge_list = skeleton2Graph(branchPoints, endPoints, skeletonTemp)
    color_centerline = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for edge in edge_list:
        edge_points = np.array(edge)
        color_centerline[edge_points[:, 0], edge_points[:, 1], :] = np.random.randint(0, 255, (3,))
    cv2.imshow("color", color_centerline)

    cv2.waitKey(0)
