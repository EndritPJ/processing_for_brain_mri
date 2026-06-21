### Processing scripts for nifti to raw conversion (for unity3d) and aligning mesh to nifti

Usage:  
Activate the conda environment from the provided yml file.

To convert a nifti file to .raw format  
```python nifti_to_raw.py image.nii.gz (optional flag --center)```

It will save a .raw file with properties as .json in the same directory is the image.

In Unity3D, when loading the .raw files, the volume will be centered around 0,0,0 with extents -0.5 to 0.5.  
For this reason, loading in meshes is not trivial as they will not be aligned to the .raw volume.  
To solve this, use ```align_mesh_to_nifti.py``` which computes the .nii -> .raw (in Unity3D) transformation, and applies it to the mesh.  

Following this, the meshes will be correctly aligned once importing them, although some flips may still be needed depending on which coordinate system was used to export the meshes in the first place.

IMPORTANT: this code needs to be applied on the uncentered images and meshes (which should already be aligned, but not necessarily centered!). You can check this in Slicer3D by importing the image as a volume and the mesh as a segmentation, they should overlap.  

```python align_mesh_to_nifti.py image.nii.gz mesh.stl (optional flag --flip_180)```

Note: be mindful of file sizes! Niftis or meshes may need to be downsampled. A good target to aim for is <30 Mb RAW and <50 Mb mesh file.
