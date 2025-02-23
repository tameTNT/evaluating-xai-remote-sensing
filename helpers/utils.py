import torch


def make_device_batches(
        x: torch.Tensor,
        max_batch_size: int,
        target_device: torch.device
) -> torch.Tensor:
    """
    Splits a tensor `x` into mini batches and moves them to the target device.
    Yields these mini batches (of size up to max_batch_size) one by one.
    """

    for i in range(0, x.shape[0], max_batch_size):
        yield x[i:i + max_batch_size].to(target_device)
