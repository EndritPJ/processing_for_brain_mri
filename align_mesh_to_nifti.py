import argparse
import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk


def get_world_to_image_transform(image: sitk.Image) -> np.ndarray:
    """
    Get transformation for uncentered image to 0,0,0
    """
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    inv_direction = np.linalg.inv(direction)
    T = np.eye(4)
    # rotation + scaling
    T[:3, :3] = inv_direction / spacing
    # translation
    T[:3, 3] = -inv_direction @ origin / spacing
    return T


def add_centering(T: np.ndarray, size: np.ndarray) -> np.ndarray:
    """
    Shift it for unity rendering (values from -0.5 to 0.5 width not 0 - 1)
    """
    C = np.eye(4)
    C[:3, 3] = -0.5 * np.array(size)
    return C @ T


def apply_transform_to_mesh(mesh_path: str, output_path: str, T: np.ndarray, flip_180: bool) -> None:
    mesh = pv.read(mesh_path)
    mesh.transform(T)
    if flip_180:
        print("\nApplying 180 degree Z rotation")
        mesh.rotate_z(180, inplace=True)
    mesh.save(output_path)


def center_mesh_to_image(image: sitk.Image, mesh_path: str, output_path: str, flip_180: bool) -> None:
    T = get_world_to_image_transform(image)
    size = np.array(image.GetSize())
    T = add_centering(T, size)
    apply_transform_to_mesh(mesh_path, output_path, T, flip_180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Center a mesh into image space.")
    parser.add_argument("image", help="Input NIfTI image")
    parser.add_argument("mesh", help="Input mesh (.stl, .ply, .vtk, etc.)")
    parser.add_argument("--flip180", action="store_true", help="Apply additional 180 degree Z rotation")
    parser.add_argument("--output", default=None, help="Output mesh path")
    args = parser.parse_args()
    image = sitk.ReadImage(args.image)
    if args.output is None:
        mesh_dir = os.path.dirname(args.mesh)
        mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
        output_path = os.path.join(mesh_dir,f"{mesh_name}_aligned.stl")
    else:
        output_path = args.output
    center_mesh_to_image(image=image, mesh_path=args.mesh, output_path=output_path, flip_180=args.flip180)


if __name__ == "__main__":
    main()