from models import AutoTimes_Llama, AutoTimes_Gpt2, AutoTimes_Opt_1b, ECG_Llama, ECG_Chat, PatchTST, TimesNet, Autoformer, DLinear, FEDformer
from models import ECG_Llama


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ECG_Llama': ECG_Llama,
            'PatchTST': PatchTST,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
