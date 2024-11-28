import torch

def _is_tensor_imu_data(imu):
    if not torch.is_tensor(imu):
        raise TypeError("imu should be Tensor. Got %s" % type(imu))

    if not imu.ndimension() == 2 :
        raise ValueError("imu should be 2D. Got %dD" % imu.dim())

    return True

def to_tensor(imu):
    return torch.tensor(imu).float().permute(1, 0)

def resize(imu, target_size, interpolation_mode):
    if not _is_tensor_imu_data(imu):
        raise ValueError("imu should be a 2D torch.tensor")    
    if type(target_size) != int:
        raise ValueError(f"target size should be temporal duration, instead got {target_size}")
    #missing batch dimension, so unsqueee
    return torch.nn.functional.interpolate(imu.unsqueeze(0), size=target_size, mode=interpolation_mode).squeeze(0)

def crop_temporal(clip, crop_size):
    if not _is_tensor_imu_data(clip):
        raise ValueError("imu should be a 2D torch.tensor")
    t = clip.size(-1)
    if t < crop_size:
        raise ValueError("imu duration must be no smaller than crop_size")

    i = int(round((t - crop_size) / 2.0))
    return clip[..., i : i + crop_size]