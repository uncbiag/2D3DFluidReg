import numpy as np
import vtk

# segData = np.load('/home/yr/Desktop/re/SegLung.npy')
# extendbox = np.load('/home/yr/Desktop/re/Parm.npy',allow_pickle=True)[0]

#    {
#     "name": "CT-Lung",
#     "ambient": "0.2",
#     "diffuse": "1",
#     "specular": "0",
#     "specularPower": "1",
#     "interpolation": "1",
#     "shade": "1",
#     "colorTransfer": {
#       "num": "6",
#       "pixelValue": "-1000 -600 -530 -460 -400 2952 ",
#       "color": "(0.3,0.3,1) (0,0,1) (0.134704,0.781726,0.0724558) (0.929244,1,0.109473) (0.888889,0.254949,0.0240258) (1,0.3,0.3) "
#     },
#     "scalarOpacity": {
#       "num": "6",
#       "pixelValue": "-1000 -600 -599 -400 -399 2952 ",
#       "opacity": "0 0 0.15 0.15 0 0 "
#     },
#     "gradientOpacity": {
#       "num": "2",
#       "pixelValue": "0 988 ",
#       "opacity": "1 1 "
#     }

colors = vtk.vtkNamedColors()


# dataImporter = vtk.vtkImageImport()
# data_string = segData.tostring()
# dataImporter.CopyImportVoidPointer(data_string, len(data_string))

# dataImporter.SetDataScalarTypeToUnsignedChar()

# dataImporter.SetNumberOfScalarComponents(1)

# dataImporter.SetDataExtent(tuple(extendbox))
# dataImporter.SetWholeExtent(tuple(extendbox))

dcm = vtk.vtkDICOMImageReader()
dcm.SetDirectoryName('../../Data/Raw/DICOMforMN/S00001/SER00002')


OpacityTransferFunc = vtk.vtkPiecewiseFunction()
OpacityTransferFunc.AddPoint(-1000, 0)
OpacityTransferFunc.AddPoint(-600, 0)
OpacityTransferFunc.AddPoint(-599, 0.15)
OpacityTransferFunc.AddPoint(-400, 0.15)
OpacityTransferFunc.AddPoint(-399, 0)
OpacityTransferFunc.AddPoint(2952, 0)

GradOpacityTransferFunc = vtk.vtkPiecewiseFunction()
GradOpacityTransferFunc.AddPoint(0, 1)
GradOpacityTransferFunc.AddPoint(988, 1)

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(-1000,0.3, 0.3, 1)
colorFunc.AddRGBPoint(-600, 0, 0, 1)
colorFunc.AddRGBPoint(-530, 0.134704, 0.781726, 0.0724558)
colorFunc.AddRGBPoint(-460, 0.929244, 1, 0.109473)
colorFunc.AddRGBPoint(-400, 0.888889, 0.254949, 0.0240258)
colorFunc.AddRGBPoint(2952, 1, 0.3, 0.3)


volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(OpacityTransferFunc)
volumeProperty.SetGradientOpacity(GradOpacityTransferFunc)
volumeProperty.SetAmbient(0.2)
volumeProperty.SetDiffuse(1)
volumeProperty.SetSpecular(0)
volumeProperty.SetSpecularPower(1)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()

volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
#volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
volumeMapper.SetInputConnection(dcm.GetOutputPort())

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)


##########################################################

# planeData = vtk.vtkPlaneSource()
# planeData.SetCenter(0,0,-120)
# planeData.SetNormal(0,1,0)
# planeData.SetPoint1(512,0,-120)
# planeData.SetPoint2(0,0,500)
# mapPlane = vtk.vtkPolyDataMapper()
# mapPlane.SetInputConnection(planeData.GetOutputPort())
# plane = vtk.vtkActor()
# plane.SetMapper(mapPlane)
# plane.GetProperty().SetDiffuseColor(colors.GetColor3d("MistyRose"))
# plane.GetProperty().SetAmbient(0.2)
# plane.GetProperty().SetDiffuse(1)
# plane.GetProperty().SetSpecular(0)
# plane.GetProperty().SetSpecularPower(1)

planeReader = vtk.vtkJPEGReader()
planeReader.SetFileName('./figure/projection.jpg')
planeReader.Update()

planeData = planeReader.GetOutput()

plane1 = vtk.vtkImageActor()
plane1.SetInputData(planeData)
plane1.SetScale(0.5,0.5,0)
plane1.SetInterpolate(1)

trans1 = vtk.vtkTransform()
trans1.Translate(0,0,430)
trans1.RotateX(270)
plane1.SetUserTransform(trans1)
cen1 = plane1.GetCenter() 




# plane2 = vtk.vtkImageActor()
# plane2.SetInputData(planeData)
# plane2.SetScale(1.5,1.5,0)
# plane2.SetInterpolate(1)

# trans2 = vtk.vtkTransform()
# trans2.Translate(20,0,340)
# trans2.RotateX(273)
# plane2.SetUserTransform(trans2)
# cen2 = plane2.GetCenter() 
# translate_z = ((cen2[2]-cen1[2])*(cen2[2]-cen1[2]) + (cen2[1]-cen1[1])*(cen2[1]-cen1[1]))**0.5
# trans2.Translate(0,0,-translate_z)
# plane2.SetUserTransform(trans2)


# plane3 = vtk.vtkImageActor()
# plane3.SetInputData(planeData)
# plane3.SetScale(1.5,1.5,0)
# plane3.SetInterpolate(1)
# trans3 = vtk.vtkTransform()
# trans3.Translate(20,0,340)
# trans3.RotateX(270)
# plane3.SetUserTransform(trans3)
# cen3 = plane3.GetCenter() 
# translate_z = ((cen3[2]-cen1[2])*(cen3[2]-cen1[2]) + (cen3[1]-cen1[1])*(cen3[1]-cen1[1]))**0.5
# trans3.Translate(0,0,-translate_z)
# plane3.SetUserTransform(trans3)





planePanel = vtk.vtkPlaneSource()
planePanel.SetOrigin(-100,-100,-2)
planePanel.SetPoint1(512,-100,-2)
planePanel.SetPoint2(-100,512,-2)
mapPlane = vtk.vtkPolyDataMapper()
mapPlane.SetInputConnection(planePanel.GetOutputPort())
plane = vtk.vtkActor()
plane.SetMapper(mapPlane)
trans = vtk.vtkTransform()
trans.Translate(20,0,340)
trans.RotateX(270)
plane.SetUserTransform(trans)
plane.GetProperty().SetDiffuseColor(1, 1, 1)
plane.GetProperty().SetSpecular(0)
plane.GetProperty().SetSpecularPower(1)
plane.GetProperty().SetDiffuse(1)
##########################################################

renderer = vtk.vtkRenderer()
renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)

renderer.AddVolume(volume)
renderer.AddActor(plane1)
# renderer.AddActor(plane2)
# renderer.AddActor(plane3)
renderer.AddActor(plane)
renderer.SetBackground(colors.GetColor3d("LightGrey"))

renderWin.SetSize(400, 400)


def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)

renderWin.AddObserver("AbortCheckEvent", exitCheck)

renderInteractor.Initialize()
renderWin.Render()
renderInteractor.Start()