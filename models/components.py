from typing import Dict

class RewardSystem:
    def __init__(self, R: dict,  f_or_dict: bool, rw_dict: dict = None, rw_func: function = None):
        self.R = R
        self.f_or_dict = f_or_dict # determines whether implemented as a function or dict



    

class TransitionKernel:

    def __init__(self, P: Dict, actions:bool = False):
        self.P = P # maps (current state, action) or just (current state) to distribution over next states

        self.actions = actions # whether or not depends on actions
