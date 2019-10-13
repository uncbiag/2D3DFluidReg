import os
import sys
import pylab
import glob

import vtk
import numpy as np
from vtk.util import numpy_support
from vtk import VTK_FLOAT


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def generateVolume(matrix_full, colorDict, alphaDict):
    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    dataImporter.SetImportVoidPointer(matrix_full)
    # The type of the newly imported data is set to unsigned short (uint16)
    dataImporter.SetDataScalarTypeToFloat()
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)

    # The following two functions describe how the data is stored and the dimensions of the array it is stored in.
    w, h, d = matrix_full.shape
    dataImporter.SetDataExtent(0, d-1, 0, h-1, 0, w-1)
    dataImporter.SetWholeExtent(0, d-1, 0, h-1, 0, w-1)
    dataImporter.SetDataSpacing((1, 1, 1))

    # This class stores color data and can create color tables from a few color points.
    colorFunc = vtk.vtkColorTransferFunction()
    for c in colorDict:
        colorFunc.AddRGBPoint(c[0], c[1], c[2], c[3])

    # Create transfer mapping scalar value to opacity
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    for a in alphaDict:
        alphaChannelFunc.AddPoint(a[0], a[1])

    # The previous two classes stored properties. Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.SetInterpolationTypeToLinear()
    # volumeProperty.ShadeOn()

    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    # volumeMapper.SetBlendModeToComposite()

    # The class vtkVolume is used to pair the previously declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    return volume


if __name__ == "__main__":
    I0_file = "./input2.npy"
    warped_file = "./warped2.npy"
    #################
    # Load data
    I0 = np.load(I0_file)[0, 0]
    warped = np.load(warped_file)[0, 0]

    #################
    # Set colorDict and alphaDict
    I0_colorDict = [[-700, 1, 0, 0], [-500, 1, 0, 0]]
    warped_colorDict = [[-700, 0, 0, 1], [-500, 0, 0, 1]]
    alphaDict = [[-710, 0.], [-700, 0.3], [-550, 0.6], [-500, 0.]]

    #################
    # Generate a mask
    d, w, h = I0.shape
    mask = create_circular_mask(w, h, radius=h/3)
    mask_3d = np.repeat(mask[:, :, np.newaxis], d, axis=2)
    mask_3d = np.transpose(mask_3d, (2, 0, 1))
    temp = np.logical_and(I0>-700, I0<-500)
    mask_3d = np.invert(mask_3d)
    mask_3d = np.logical_and(mask_3d, temp)
  
    ################
    # Apply mask
    I0[mask_3d] = -1060.
    warped[mask_3d] = -1060.

    ################
    # Generate volume
    I0_volume = generateVolume(I0, I0_colorDict, alphaDict)
    warped_volume = generateVolume(warped, warped_colorDict, alphaDict)

    ################
    # Start render
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # We add the volume to the renderer ...
    renderer.AddViewProp(I0_volume)
    renderer.AddViewProp(warped_volume)
    # ... set background color to white ...
    renderer.SetBackground(1, 1, 1)
    # ... and set window size.
    renderWin.SetSize(550, 550)
    # renderWin.SetMultiSamples(4)

    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()
    # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()
