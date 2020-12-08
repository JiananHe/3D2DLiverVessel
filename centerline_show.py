import vtk
import numpy as np
import os
import cv2
from centeline_tree_reader import construct_tree_from_txt, get_branches_points
np.random.seed(0)


def make_stl_actor(stl_file, color, opacity):
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file)
    stl_reader.Update()
    vessel = stl_reader.GetOutput()

    vessel_mapper = vtk.vtkPolyDataMapper()
    vessel_mapper.SetInputData(vessel)

    vessel_actor = vtk.vtkActor()
    vessel_actor.SetMapper(vessel_mapper)
    vessel_actor.GetProperty().SetColor(*color)
    vessel_actor.GetProperty().SetOpacity(opacity)
    return vessel_actor


def get_background_render(image_path):
    # Read the image
    jpeg_reader = vtk.vtkJPEGReader()
    assert jpeg_reader.CanReadFile(image_path), "Error reading file %s" % image_path

    jpeg_reader.SetFileName(image_path)
    jpeg_reader.Update()
    image_data = jpeg_reader.GetOutput()

    # Create an image actor to display the image
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)

    # Create a renderer to display the image in the background
    background_renderer = vtk.vtkRenderer()
    background_renderer.AddActor(image_actor)

    return image_data, background_renderer


def show_branches_3d(branches_points, branch_idx, show_shape, bg_image_path=None, fix_color=False, vessel_stl=None, liver_stl=None,
                  show_window=True, save_name=None, save_dir=None):
    lines_actors = []
    sid = 0
    for bid in branch_idx:
        branch_points = branches_points[sid: sid+bid, :]
        sid += bid

        vtk_points = vtk.vtkPoints()
        for i, bp in enumerate(branch_points):
            vtk_points.InsertPoint(i, *bp)

        spline = vtk.vtkParametricSpline()
        spline.SetPoints(vtk_points)

        spline_source = vtk.vtkParametricFunctionSource()
        spline_source.SetParametricFunction(spline)
        spline_source.Update()

        spline_mapper = vtk.vtkPolyDataMapper()
        spline_mapper.SetInputData(spline_source.GetOutput())

        spline_actor = vtk.vtkActor()
        spline_actor.SetMapper(spline_mapper)
        if not fix_color:
            if bid == 0:
                spline_actor.GetProperty().SetColor(1.0, .0, .0)
            else:
                color = np.random.rand(3)
                while (color == [1.0, .0, .0]).all():
                    continue
                spline_actor.GetProperty().SetColor(np.random.rand(3))
        else:
            spline_actor.GetProperty().SetColor(1.0, .0, .0)

        spline_actor.GetProperty().SetLineWidth(3)
        lines_actors.append(spline_actor)

    scene_renderer = vtk.vtkRenderer()
    for actor in lines_actors:
        scene_renderer.AddActor(actor)

    if vessel_stl is not None:
        for v_stl in vessel_stl:
            vessel_actor = make_stl_actor(v_stl, [1.0, 0.2, 0.1], 0.1)
            scene_renderer.AddActor(vessel_actor)

    if liver_stl is not None:
        liver_actor = make_stl_actor(liver_stl, [.0, 0.2, 1.], 0.1)
        scene_renderer.AddActor(liver_actor)

    render_window = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()

    if bg_image_path is None:
        scene_renderer.SetBackground(1, 1, 1)
        render_window.AddRenderer(scene_renderer)
        render_window.SetSize(*show_shape)
        iren.SetRenderWindow(render_window)
    else:
        bg_image_data, background_renderer = get_background_render(bg_image_path)

        # Set up the render window and renderers such that there is
        # a background layer and a foreground layer
        background_renderer.SetLayer(0)
        background_renderer.InteractiveOff()
        scene_renderer.SetLayer(1)
        render_window.SetNumberOfLayers(2)
        render_window.AddRenderer(background_renderer)
        render_window.AddRenderer(scene_renderer)

        iren.SetRenderWindow(render_window)

        # Render once to figure out where the background camera will be
        render_window.Render()

        # Set up the background camera to fill the renderer with the image
        origin = bg_image_data.GetOrigin()
        spacing = bg_image_data.GetSpacing()
        extent = bg_image_data.GetExtent()

        camera = background_renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
        # xd = (extent[1] - extent[0] + 1) * spacing[0]
        yd = (extent[3] - extent[2] + 1) * spacing[1]
        d = camera.GetDistance()
        camera.SetParallelScale(0.5 * yd)
        camera.SetFocalPoint(xc, yc, 0.0)
        camera.SetPosition(xc, yc, d)

    # camera
    # camera = vtk.vtkCamera()
    # camera.SetPosition(source_point)
    # focal_point = np.sum((origin_2d - source_point) * plane_normal) * plane_normal + source_point
    # camera.SetFocalPoint(focal_point)
    # camera.ComputeViewPlaneNormal()
    # ren1.SetActiveCamera(camera)

    if show_window:
        render_window.Render()

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.SetRenderWindow(render_window)

        render_window.Render()
        if save_name is None:
            iren.Start()

    if save_name is not None:
        render_window.Render()
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(render_window)
        # windowToImageFilter.SetMagnification(3)
        windowToImageFilter.SetInputBufferTypeToRGBA()
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()

        pngWriter = vtk.vtkPNGWriter()
        if save_dir is None:
            pngWriter.SetFileName("data/centerline_projections/" + save_name)
        else:
            pngWriter.SetFileName(os.path.join(save_dir, save_name))

        pngWriter.SetInputData(windowToImageFilter.GetOutput())
        pngWriter.Write()
        print("saved as %s" % save_name)
    del render_window


def show_branches_2d(branches_points, branch_idx, plane_centre, plane_size, plane_spacing, bg_image_path=None,
                     fix_color=False, show=True, save_name=None, save_dir=None):
    # plane_centre --> plane_size//2
    points_coord = (branches_points - plane_centre)[:, :2] / plane_spacing + np.array(plane_size) // 2
    points_coord = np.array(np.round(points_coord), dtype=np.int)
    points_coord[:, 0][points_coord[:, 0] < 0] = 0
    points_coord[:, 0][points_coord[:, 0] > plane_size[0] - 1] = plane_size[0] - 1
    points_coord[:, 1][points_coord[:, 1] < 0] = 0
    points_coord[:, 1][points_coord[:, 1] > plane_size[1] - 1] = plane_size[1] - 1

    if bg_image_path is None:
        plane_image = np.zeros([*plane_size, 3], dtype=np.uint8)
    else:
        plane_image = cv2.imread(bg_image_path, 1)
    # cv2.circle(plane_image, (100, 200), 5, (0, 255, 0))
    sid = 0
    for bid in branch_idx:
        branch_points = points_coord[sid: sid+bid]
        sid += bid

        color = [255, 255, 255]
        if not fix_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        pid = 0
        while pid < len(branch_points) - 1:
            cv2.line(plane_image, tuple(branch_points[pid]), tuple(branch_points[pid+1]), color)
            pid += 1

    cv2.imshow("projection", plane_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    root1, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
    root2, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    branches_points1, branches_index1 = get_branches_points(root1)
    branches_points2, branches_index2 = get_branches_points(root2)
    # branches_points = branches_points1 + branches_points2

    show_branches_3d(branches_points1, branches_index1, (512, 512),
                  vessel_stl=["../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left_002.stl",
                              "../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right_002.stl"])
    # show_branches(branches_points1, "../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1.jpg",
    #               vessel_stl=["../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left_002.stl",
    #                           "../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right_002.stl"])
    # show_branches(branches_points1, vessel_stl="../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left_002.stl")
    # show_branches(branches_points2, vessel_stl="../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right_002.stl")

    # show_branches(branches_points, "data/vessel.stl", "data/liver.stl")
