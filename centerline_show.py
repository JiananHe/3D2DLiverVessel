import vtk
import numpy as np
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


def show_branches(branches_points, source_point=None, origin_2d=None, plane_normal=None,
                  fix_color=False, vessel_stl=None, liver_stl=None, show_window=True, window_save_name=None):
    lines_actors = []
    for bid, branch_points in enumerate(branches_points):
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

    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    for actor in lines_actors:
        ren1.AddActor(actor)

    if vessel_stl is not None:
        for v_stl in vessel_stl:
            vessel_actor = make_stl_actor(v_stl, [1.0, 0.2, 0.1], 0.1)
            ren1.AddActor(vessel_actor)

    if liver_stl is not None:
        liver_actor = make_stl_actor(liver_stl, [.0, 0.2, 1.], 0.1)
        ren1.AddActor(liver_actor)

    ren1.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    renWin.Render()

    # camera
    # camera = vtk.vtkCamera()
    # camera.SetPosition(source_point)
    # focal_point = np.sum((origin_2d - source_point) * plane_normal) * plane_normal + source_point
    # camera.SetFocalPoint(focal_point)
    # camera.ComputeViewPlaneNormal()
    # ren1.SetActiveCamera(camera)

    if window_save_name is not None:
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renWin)
        # windowToImageFilter.SetMagnification(3)
        windowToImageFilter.SetInputBufferTypeToRGBA()
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()

        pngWriter = vtk.vtkPNGWriter()
        pngWriter.SetFileName("data/centerline_projections/" + window_save_name)
        pngWriter.SetInputData(windowToImageFilter.GetOutput())
        pngWriter.Write()
        print("saved as %s" % window_save_name)

    if show_window:
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.SetRenderWindow(renWin)

        renWin.Render()
        iren.Start()


if __name__ == '__main__':
    root1, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
    root2, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    branches_points1 = get_branches_points(root1)
    branches_points2 = get_branches_points(root2)
    branches_points = branches_points1 + branches_points2
    show_branches(branches_points, vessel_stl=["../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left_002.stl",
                                               "../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right_002.stl"])
    # show_branches(branches_points1, vessel_stl="../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left_002.stl")
    # show_branches(branches_points2, vessel_stl="../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right_002.stl")
    # show_branches(branches_points, "data/vessel.stl", "data/liver.stl")
