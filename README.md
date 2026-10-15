### Processing scripts for nifti to raw conversion (for unity3d) while keeping meshes aligned

Usage:  
Activate the conda environment from the provided yml file.

This script is used for preparing nifti volumes and mesh files for Unity import. Use case: you have a nifti and some meshes you want to import in Unity. Important: it is a prerequisite for the images and meshes to be already aligned! You can check this in Slicer3D.

Simply converting the .nii to a .raw file and importing in Unity loses information about the image center and rotation (direction matrices). Slicer3D uses that information to align the images to the meshes properly, so we need to resample the raw data so that it works with a center and direction matrix of 0,0,0 and identity, respectively. This may also require translating the mesh in 3D.

### Recommended all-in-one: resample + center + remove rotation, and carry the meshes along

Note: you can use this without meshes as arguments, in case you just want to process images.

```python process_nifti_meshes.py image.nii.gz mesh1.obj mesh2.stl --odir out/```

Resamples the image to isotropic spacing (1mm default, `--spacing X`, or `--min_voxel_spacing` to keep the finest input spacing), centres it at (0,0,0) with an identity direction matrix, and saves `.nii.gz` + `.raw`/`.json` (same intensity windowing as `nifti_to_raw.py`). The meshes (aligned to the image in Slicer, LPS) get the same rigid transform, so the output image and meshes still overlap in Slicer. The 4x4 transform is stored in the `.json` as `world_to_output`.

NOTE: `--world_aligned` is recommended as otherwise the results are in scanner space.

By default the voxel axes are kept (no rotation interpolation; anatomy may look rotated relative to Slicer's labels). Use `--world_aligned` to resample onto the LPS axes instead.  

Other flags: `--interp {linear,bspline,nearest}`, `--percentiles LO HI` (computed on the input image), `--pad_value` (default 0, keeps background at 0 in the .raw), `--suffix`, `--mesh_space {auto,LPS,RAS}`.

Note: be mindful of file sizes! Niftis or meshes may need to be downsampled. A good target to aim for is <30 Mb RAW and <50 Mb mesh file.


### Legacy scripts: 
To convert a nifti file to .raw format  
```python nifti_to_raw.py image.nii.gz (optional flag --center)```

```python align_mesh_to_nifti.py image.nii.gz mesh.stl (optional flag --flip_180)```