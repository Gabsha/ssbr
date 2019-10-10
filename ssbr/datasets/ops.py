from typing import List, Callable, Tuple, Iterator
import SimpleITK as sitk
import numpy as np

from functools import reduce


def load_dicom(folder: str) -> sitk.Image:
    """Load a dicom volume from a folder and returns a numpy array"""
    # Read dicom volume
    series_reader = sitk.ImageSeriesReader()
    files = series_reader.GetGDCMSeriesFileNames(folder)
    series_reader.SetFileNames(files)
    volume = series_reader.Execute()
    # orig_size = volume.GetSize()
    # orig_spacing = volume.GetSpacing()

    # Read dicom info
    # info_I: sitk.Image = sitk.ReadImage(files[0])
    # dcm_keys = info_I.GetMetaDataKeys()
    # series_uid = info_I.GetMetaData('0020|000e')
    # direction = I.GetDirection()
    # volume_id = series_uid

    return volume


def image2np(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def resize(target_size: Tuple[float, float, float]):
    """Resize an image to the target size"""
    def f(image: sitk.Image):
        orig_size = image.GetSize()
        orig_spacing = image.GetSpacing()
        size_ratio = (float(orig_size[0]) / float(target_size[0]), float(orig_size[1]) / float(target_size[1]))
        new_size = target_size + (orig_size[2], )
        new_spacing = (size_ratio[0] * orig_spacing[0], size_ratio[1] * orig_spacing[1]) + (orig_spacing[2], )
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(image)
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resized = resample.Execute(image)

        return resized

    return f


def rescale(low, high, scale=255, dtype=np.uint8):
    """Rescale intensity of array"""
    def f(array: np.ndarray) -> np.ndarray:
        array[array < low] = low
        array[array > high] = high
        array = array - low
        array = array / (high - low)
        array = array * scale
        array = array.astype(dtype=dtype)

        return array

    return f


def grey2rgb(array: np.ndarray) -> np.ndarray:
    """Convert a greyscale matrix to 3 channel rgb by triplicating the channel"""
    new_size = array.shape
    rgb_stack = np.zeros((new_size[0] - 2, new_size[1], new_size[2], 3), dtype=np.int16)
    rgb_stack[:, :, :, 0] = array[0:-2, :, :]
    rgb_stack[:, :, :, 1] = array[1:-1, :, :]
    rgb_stack[:, :, :, 2] = array[2:, :, :]

    return rgb_stack


def compose(transforms: List[Callable]):
    def f(inp: Iterator):
        return reduce(lambda f, g: (g(x) for x in f), transforms, inp)

    return f