import argparse
import json
import os
import numpy as np
import SimpleITK as sitk


def resample_isotropic(image: sitk.Image, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> sitk.Image:
    """
    Resample image so it has isotropic (1, 1, 1) spacing in mm.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [int(size[i] * original_spacing[i] / spacing[i]) for i in range(3)]
    resampler.SetSize(new_size)
    return resampler.Execute(image)


def window(volume: np.ndarray, level: float, width: float) -> np.ndarray:
    """
    Decide how to convert voxel intensities into greyscale values 0-1.
    """
    minv = level - width / 2.0
    maxv = level + width / 2.0
    volume = np.clip(volume, minv, maxv)
    volume = (volume - minv) / (maxv - minv)
    return volume


def center_image(image: sitk.Image) -> sitk.Image:
    """
    Reorient to identity axes and place the image centre at (0,0,0).
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    identity_direction = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    half_extent = [0.5 * (size[i] - 1) * spacing[i] for i in range(3)]
    new_origin = tuple(-h for h in half_extent)
    reference = sitk.Image(size, image.GetPixelID())
    reference.SetSpacing(spacing)
    reference.SetDirection(identity_direction)
    reference.SetOrigin(new_origin)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


def write_volume_metadata(image: sitk.Image, volume: np.ndarray, json_path: str) -> None:
    metadata = {
        "width": image.GetSize()[0],
        "height": image.GetSize()[1],
        "depth": image.GetSize()[2],
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "dtype": str(volume.dtype),
        "order": "zyx"
    }
    with open(json_path, "w") as fp:
        json.dump(metadata, fp, indent=2)


def nifti_to_raw(image: sitk.Image, output_base: str) -> None:
    print("\n========== INPUT IMAGE ==========")
    print("Size      :", image.GetSize())
    print("Spacing   :", image.GetSpacing())
    print("Origin    :", image.GetOrigin())
    print("Direction :", image.GetDirection())
    image_resampled = resample_isotropic(image)
    print("\n========== RESAMPLED IMAGE ==========")
    print("Size      :", image_resampled.GetSize())
    print("Spacing   :", image_resampled.GetSpacing())
    print("Origin    :", image_resampled.GetOrigin())
    print("Direction :", image_resampled.GetDirection())
    volume = sitk.GetArrayFromImage(image_resampled)
    p1, p99 = np.percentile(volume, [1, 99])
    level = (p1 + p99) / 2.0
    width = p99 - p1
    volume = window(volume, level=level, width=width)
    volume = (volume * 65535).astype(np.uint16)
    raw_path = output_base + ".raw"
    json_path = output_base + ".json"
    volume.tofile(raw_path)
    write_volume_metadata(image_resampled, volume, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NIfTI to RAW + JSON")
    parser.add_argument("ifile", help="Input NIfTI file")
    parser.add_argument("--center", action="store_true", help="Center image at (0,0,0) and remove rotation")
    args = parser.parse_args()
    print("Reading:")
    print(args.ifile)
    image = sitk.ReadImage(args.ifile)
    if args.center:
        print("\n========== CENTERING ==========")
        print("Original origin:")
        print(image.GetOrigin())
        print("Original direction:")
        print(image.GetDirection())
        image = center_image(image)
        print("\nNew origin:")
        print(image.GetOrigin())
        print("New direction:")
        print(image.GetDirection())
    output_base = args.ifile
    if output_base.endswith(".nii.gz"):
        output_base = output_base[:-7]
    elif output_base.endswith(".nii"):
        output_base = output_base[:-4]
    else:
        output_base = os.path.splitext(output_base)[0]
    nifti_to_raw(image, output_base)


if __name__ == "__main__":
    main()