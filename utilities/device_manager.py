import torch


class DeviceManager:
    @staticmethod
    def get_device() -> torch.device:
        enable_mps = False
        return ('mps' if torch.backends.mps.is_built() else ('cuda' if torch.backends.cuda.is_built() else 'cpu')) if enable_mps else torch.device('cuda' if torch.backends.cuda.is_built() else 'cpu')
