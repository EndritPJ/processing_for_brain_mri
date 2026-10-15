import argparse
import json
import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk


INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "nearest": sitk.sitkNearestNeighbor,
}

# Slicer (and SimpleITK) default to LPS for mesh files; RAS -> LPS is a flip of x and y.
LPS_RAS_FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def print_image_info(title: str, image: sitk.Image) -> None:
    print(f"\n========== {title} ==========")
    print("Size      :", image.GetSize())
    print("Spacing   :", image.GetSpacing())
    print("Origin    :", image.GetOrigin())
    print("Direction :", image.GetDirection())


def strip_nifti_ext(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def output_spacing(image: sitk.Image, args: argparse.Namespace) -> float:
    """
    Isotropic output spacing: either the finest input spacing or a fixed value (default 1mm).
    """
    if args.min_voxel_spacing:
        return float(min(image.GetSpacing()))
    return float(args.spacing)


def compute_transform(image: sitk.Image, spacing: float, world_aligned: bool):
    """
    Build the rigid world -> output map p_new = M @ (p_old - c_old).
    Default keeps the voxel axes (M = R^-1, no rotation interpolation);
    world_aligned keeps the LPS orientation (M = I, translation only).
    Returns the 4x4 mesh transform T, M, c_old and the output grid size.
    """
    R = np.array(image.GetDirection()).reshape(3, 3)
    s_old = np.array(image.GetSpacing())
    n_old = np.array(image.GetSize())
    origin = np.array(image.GetOrigin())
    c_old = origin + R @ (s_old * (n_old - 1) / 2.0)
    if world_aligned:
        M = np.eye(3)
        # bounding box of the voxel-edge corners after mapping into the output frame
        corners = np.array([[i, j, k] for i in (-0.5, n_old[0] - 0.5)
                            for j in (-0.5, n_old[1] - 0.5)
                            for k in (-0.5, n_old[2] - 0.5)])
        corners_world = origin + (R @ (corners * s_old).T).T
        corners_new = (M @ (corners_world - c_old).T).T
        extent = corners_new.max(axis=0) - corners_new.min(axis=0)
        out_size = np.maximum(1, np.ceil(extent / spacing - 1e-6)).astype(int)
    else:
        M = np.linalg.inv(R)
        out_size = np.maximum(1, np.round(n_old * s_old / spacing)).astype(int)
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = -M @ c_old
    return T, M, c_old, [int(n) for n in out_size]


def resample_image(image: sitk.Image, M: np.ndarray, c_old: np.ndarray, out_size: list[int],
                   spacing: float, interp: str, pad_value: float) -> sitk.Image:
    """
    Single resample onto an identity-direction grid centred at (0,0,0).
    The sitk transform maps output points to input points: p_old = M^-1 @ p_new + c_old.
    Voxels outside the input field of view are filled with pad_value.
    """
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(np.linalg.inv(M).ravel().tolist())
    tx.SetTranslation(c_old.tolist())
    out_origin = [-spacing * (n - 1) / 2.0 for n in out_size]
    return sitk.Resample(image, out_size, tx, INTERPOLATORS[interp], out_origin,
                         [spacing] * 3, np.eye(3).ravel().tolist(), float(pad_value),
                         image.GetPixelID())


def window(volume: np.ndarray, level: float, width: float) -> np.ndarray:
    """
    Decide how to convert voxel intensities into greyscale values 0-1.
    """
    minv = level - width / 2.0
    maxv = level + width / 2.0
    volume = np.clip(volume, minv, maxv)
    volume = (volume - minv) / (maxv - minv)
    return volume


def write_volume_metadata(image: sitk.Image, volume: np.ndarray, json_path: str, T: np.ndarray) -> None:
    metadata = {
        "width": image.GetSize()[0],
        "height": image.GetSize()[1],
        "depth": image.GetSize()[2],
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "dtype": str(volume.dtype),
        "order": "zyx",
        "world_to_output": T.tolist()
    }
    with open(json_path, "w") as fp:
        json.dump(metadata, fp, indent=2)


def write_raw(image: sitk.Image, output_base: str, p_lo: float, p_hi: float, T: np.ndarray) -> None:
    """
    Window to [p_lo, p_hi] and save as uint16. The window comes from the input image,
    so padding added by resampling does not shift it (values <= p_lo become 0).
    """
    volume = sitk.GetArrayFromImage(image)
    level = (p_lo + p_hi) / 2.0
    width = p_hi - p_lo
    volume = window(volume, level=level, width=width)
    volume = (volume * 65535).astype(np.uint16)
    volume.tofile(output_base + ".raw")
    write_volume_metadata(image, volume, output_base + ".json", T)


def read_mesh_space(mesh_path: str) -> str:
    """
    Look for the 'SPACE=RAS' / 'SPACE=LPS' tag Slicer writes in the file header (LPS if absent).
    """
    with open(mesh_path, "rb") as fp:
        header = fp.read(256).decode("latin-1").upper()
    return "RAS" if "SPACE=RAS" in header else "LPS"


def write_mesh(mesh: pv.PolyData, output_path: str) -> None:
    """
    Save the mesh, tagging STL/OBJ headers with SPACE=LPS so Slicer reads them unambiguously.
    """
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(mesh)
        writer.SetFileTypeToBinary()
        writer.SetHeader("SPACE=LPS")
        writer.Write()
    elif ext == ".obj":
        mesh.save(output_path)
        with open(output_path, "r") as fp:
            content = fp.read()
        with open(output_path, "w") as fp:
            fp.write("# SPACE=LPS\n" + content)
    else:
        mesh.save(output_path)


def transform_mesh(mesh_path: str, output_path: str, T: np.ndarray, mesh_space: str) -> None:
    space = read_mesh_space(mesh_path) if mesh_space == "auto" else mesh_space
    T_total = T @ LPS_RAS_FLIP if space == "RAS" else T
    print(f"\n{mesh_path} ({space}) -> {output_path}")
    mesh = pv.read(mesh_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.transform(T_total, inplace=False)
    if np.linalg.det(T_total[:3, :3]) < 0:
        # a reflection reverses the face winding, flip it back so normals stay outward
        mesh = mesh.flip_faces(inplace=False)
    write_mesh(mesh, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample a NIfTI to isotropic spacing, centre it at (0,0,0) with identity direction, "
                    "save .nii.gz + .raw/.json, and apply the same transform to any meshes.")
    parser.add_argument("image", help="Input NIfTI image (.nii / .nii.gz)")
    parser.add_argument("meshes", nargs="*", help="Meshes aligned to the image in Slicer (.obj, .stl, .ply, .vtk, .vtp)")
    parser.add_argument("--odir", default=None, help="Output directory (default: image directory)")
    spacing_group = parser.add_mutually_exclusive_group()
    spacing_group.add_argument("--min_voxel_spacing", action="store_true",
                               help="Resample to the smallest input spacing (e.g. 0.7x1x1 -> 0.7 isotropic)")
    spacing_group.add_argument("--spacing", type=float, default=1.0, help="Isotropic output spacing in mm (default 1.0)")
    parser.add_argument("--world_aligned", action="store_true",
                        help="Resample onto LPS world axes (keeps anatomical orientation) instead of keeping voxel axes")
    parser.add_argument("--interp", choices=list(INTERPOLATORS), default="linear",
                        help="Image interpolator (use nearest for label maps)")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[1.0, 99.0], metavar=("LO", "HI"),
                        help="Intensity percentiles of the input image used to window the .raw (default 1 99)")
    parser.add_argument("--pad_value", type=float, default=0.0,
                        help="Intensity for voxels outside the input field of view (default 0, MRI background)")
    parser.add_argument("--suffix", default="", help="Suffix appended to every output basename")
    parser.add_argument("--mesh_space", choices=["auto", "LPS", "RAS"], default="auto",
                        help="Coordinate system of the input meshes (auto reads the Slicer header)")
    args = parser.parse_args()

    odir = args.odir if args.odir is not None else os.path.dirname(os.path.abspath(args.image))
    os.makedirs(odir, exist_ok=True)
    image_base = os.path.join(odir, strip_nifti_ext(args.image) + args.suffix)
    mesh_outputs = []
    for mesh_path in args.meshes:
        name, ext = os.path.splitext(os.path.basename(mesh_path))
        mesh_outputs.append(os.path.join(odir, name + args.suffix + ext))
    inputs = {os.path.realpath(p) for p in [args.image] + args.meshes}
    outputs = [image_base + ".nii.gz", image_base + ".raw", image_base + ".json"] + mesh_outputs
    clashes = [p for p in outputs if os.path.realpath(p) in inputs]
    if clashes:
        parser.error(f"output would overwrite input: {clashes} (use a different --odir or --suffix)")

    print("Reading:")
    print(args.image)
    image = sitk.ReadImage(args.image)
    print_image_info("INPUT IMAGE", image)

    spacing = output_spacing(image, args)
    T, M, c_old, out_size = compute_transform(image, spacing, args.world_aligned)
    p_lo, p_hi = np.percentile(sitk.GetArrayViewFromImage(image), args.percentiles)
    if args.pad_value > p_lo:
        print(f"\nWARNING: --pad_value {args.pad_value} is above the lower window bound {p_lo:.4g}, "
              "padded voxels will not be 0 in the .raw")
    image_out = resample_image(image, M, c_old, out_size, spacing, args.interp, args.pad_value)
    print_image_info("OUTPUT IMAGE", image_out)
    print("\nWorld -> output transform:")
    print(np.array2string(T, precision=4, suppress_small=True))

    sitk.WriteImage(image_out, image_base + ".nii.gz")
    write_raw(image_out, image_base, p_lo, p_hi, T)
    print(f"\nSaved {image_base}.nii.gz / .raw / .json")

    for mesh_path, output_path in zip(args.meshes, mesh_outputs):
        transform_mesh(mesh_path, output_path, T, args.mesh_space)


if __name__ == "__main__":
    main()
