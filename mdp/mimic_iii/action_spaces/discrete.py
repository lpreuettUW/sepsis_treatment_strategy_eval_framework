import math
import torch
from enum import Enum, unique


@unique
class Discrete(Enum):
    NoOp = 0 # IV = 0, Vp = 0

    NoIV_MinVp = 1 # IV = 0, 0 < Vp <= 0.07
    NoIV_LowVp = 2 # IV = 0, 0.07 < Vp < 0.2
    NoIV_MidVp = 3 # IV = 0, 0.2 <= Vp <= 0.4
    NoIV_HighVp = 4 # IV = 0, 0.4 < Vp

    MinIV_NoVp = 5 # 0 < IV <= 48.5, Vp = 0
    MinIV_MinVp = 6 # 0 < IV <= 48.5, 0 < Vp <= 0.07
    MinIV_LowVp = 7 # 0 < IV <= 48.5, 0.07 < Vp < 0.2
    MinIV_MidVp = 8 # 0 < IV <= 48.5, 0.2 <= Vp <= 0.4
    MinIV_HighVp = 9 # 0 < IV <= 48.5, 0.4 < Vp

    LowIV_NoVp = 10 # 48.5 < IV <= 150, Vp = 0
    LowIV_MinVp = 11 # 48.5 < IV <= 150, 0 < Vp <= 0.07
    LowIV_LowVp = 12 # 48.5 < IV <= 150, 0.07 < Vp < 0.2
    LowIV_MidVp = 13 # 48.5 < IV <= 150, 0.2 <= Vp <= 0.4
    LowIV_HighVp = 14 # 48.5 < IV <= 150, 0.4 < Vp

    MidIV_NoVp = 15 # 150 < IV <= 499.1, Vp = 0
    MidIV_MinVp = 16 # 150 < IV <= 499.1, 0 < Vp <= 0.07
    MidIV_LowVp = 17 # 150 < IV <= 499.1, 0.07 < Vp < 0.2
    MidIV_MidVp = 18 # 150 < IV <= 499.1, 0.2 <= Vp <= 0.4
    MidIV_HighVp = 19 # 150 < IV <= 499.1, 0.4 < Vp

    HighIV_NoVp = 20 # 499.1 < IV, Vp = 0
    HighIV_MinVp = 21 # 499.1 < IV, 0 < Vp <= 0.07
    HighIV_LowVp = 22 # 499.1 < IV, 0.07 < Vp < 0.2
    HighIV_MidVp = 23 # 499.1 < IV, 0.2 <= Vp <= 0.4
    HighIV_HighVp = 24 # 499.1 < IV, 0.4 < Vp

    @staticmethod
    def Extract_IV_Fluids_From_Action(action: int) -> int: # Discrete | int
        if type(action) is int:
            return action // 5
        else:
            return Discrete(action.value // 5)

    @staticmethod
    def Extract_VP_From_Action(action: int) -> int: # Discrete | int
        if type(action) is int:
            return action % 5
        else:
            return Discrete(action.value % 5)

    @staticmethod
    def Continuous_Tensor_To_Discrete_Tensor(actions: torch.FloatTensor) -> torch.LongTensor:
        discrete_iv_fluids = Discrete.Continuous_IV_Fluids_Tensor_To_Discrete_Tensor(actions[:, 0])
        discrete_vps = Discrete.Continuous_VP_Tensor_To_Discrete_Tensor(actions[:, 1])
        return discrete_iv_fluids * 5 + discrete_vps

    @staticmethod
    def Continuous_To_Discrete(iv_fluid: float, vp: float) -> 'Discrete':
        iv_fluid_discrete = Discrete.Continuous_IV_Fluids_To_Discrete(iv_fluid)
        vp_discrete = Discrete.Continuous_VP_To_Discrete(vp)
        return Discrete(iv_fluid_discrete.value * 5 + vp_discrete.value)

    @staticmethod
    def Continuous_IV_Fluids_Tensor_To_Discrete_Tensor(iv_fluids: torch.FloatTensor) -> torch.LongTensor:
        if iv_fluids.ndim > 1:
            raise ValueError('iv_fluids must be 1D')
        no_op_mask = iv_fluids.isclose(torch.tensor(0.0))
        min_iv_mask = (iv_fluids <= 48.5).logical_or(iv_fluids.isclose(torch.tensor(48.5)))
        low_iv_mask = (iv_fluids > 48.5).logical_and(iv_fluids <= 150).logical_or(iv_fluids.isclose(torch.tensor(150.0)))
        mid_iv_mask = (iv_fluids > 150).logical_and(iv_fluids <= 499.1).logical_or(iv_fluids.isclose(torch.tensor(499.1)))
        high_iv_mask = iv_fluids > 499.1
        iv_fluids_discrete = torch.zeros(iv_fluids.size(0), dtype=torch.long, device=iv_fluids.device)
        iv_fluids_discrete[no_op_mask] = Discrete.Extract_IV_Fluids_From_Action(Discrete.NoOp).value
        iv_fluids_discrete[min_iv_mask] = Discrete.Extract_IV_Fluids_From_Action(Discrete.MinIV_NoVp).value
        iv_fluids_discrete[low_iv_mask] = Discrete.Extract_IV_Fluids_From_Action(Discrete.LowIV_NoVp).value
        iv_fluids_discrete[mid_iv_mask] = Discrete.Extract_IV_Fluids_From_Action(Discrete.MidIV_NoVp).value
        iv_fluids_discrete[high_iv_mask] = Discrete.Extract_IV_Fluids_From_Action(Discrete.HighIV_NoVp).value
        return iv_fluids_discrete

    @staticmethod
    def Continuous_IV_Fluids_To_Discrete(iv_fluid: float) -> 'Discrete':
        if math.isclose(iv_fluid, 0):
            return Discrete.Extract_IV_Fluids_From_Action(Discrete.NoOp)
        elif iv_fluid <= 48.5 or math.isclose(iv_fluid, 48.5):
            return Discrete.Extract_IV_Fluids_From_Action(Discrete.MinIV_NoVp)
        elif iv_fluid <= 150 or math.isclose(iv_fluid, 150):
            return Discrete.Extract_IV_Fluids_From_Action(Discrete.LowIV_NoVp)
        elif iv_fluid <= 499.1 or math.isclose(iv_fluid, 499.1):
            return Discrete.Extract_IV_Fluids_From_Action(Discrete.MidIV_NoVp)
        else:
            return Discrete.Extract_IV_Fluids_From_Action(Discrete.HighIV_NoVp)

    @staticmethod
    def Continuous_VP_Tensor_To_Discrete_Tensor(vps: torch.FloatTensor) -> torch.LongTensor:
        if vps.ndim > 1:
            raise ValueError('vps must be 1D')
        no_op_mask = vps.isclose(torch.tensor(0.0))
        min_vp_mask = (vps <= 0.07).logical_or(vps.isclose(torch.tensor(0.07)))
        low_vp_mask = (vps > 0.07).logical_and(vps < 0.2)
        mid_vp_mask = (vps >= 0.2).logical_and(vps <= 0.4).logical_or(vps.isclose(torch.tensor(0.4)))
        high_vp_mask = vps > 0.4
        vps_discrete = torch.zeros(vps.size(0), dtype=torch.long, device=vps.device)
        vps_discrete[no_op_mask] = Discrete.Extract_VP_From_Action(Discrete.NoOp).value
        vps_discrete[min_vp_mask] = Discrete.Extract_VP_From_Action(Discrete.NoIV_MinVp).value
        vps_discrete[low_vp_mask] = Discrete.Extract_VP_From_Action(Discrete.NoIV_LowVp).value
        vps_discrete[mid_vp_mask] = Discrete.Extract_VP_From_Action(Discrete.NoIV_MidVp).value
        vps_discrete[high_vp_mask] = Discrete.Extract_VP_From_Action(Discrete.NoIV_HighVp).value
        return vps_discrete

    @staticmethod
    def Continuous_VP_To_Discrete(vp: float) -> 'Discrete':
        if math.isclose(vp, 0):
            return Discrete.Extract_VP_From_Action(Discrete.NoOp)
        elif vp <= 0.07 or math.isclose(vp, 0.07):
            return Discrete.Extract_VP_From_Action(Discrete.NoIV_MinVp)
        elif vp < 0.2:
            return Discrete.Extract_VP_From_Action(Discrete.NoIV_LowVp)
        elif vp <= 0.4 or math.isclose(vp, 0.4):
            return Discrete.Extract_VP_From_Action(Discrete.NoIV_MidVp)
        else:
            return Discrete.Extract_VP_From_Action(Discrete.NoIV_HighVp)

    @staticmethod
    def Dim() -> int:
        return 25
