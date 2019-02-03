try:
    import matlab
    import matlab.engine
finally:
    pass

import io
from collections import OrderedDict as ODict
from datetime import datetime
import shutil
from typing import Dict
import inspect

import numpy as np


class CallableSingletonMeta(type):
    def __call__(cls, *args, **kwargs):
        if hasattr(cls, 'instance') and cls.instance:
            if (args or kwargs) and callable(cls.instance):
                return cls.instance(*args, **kwargs)
            else:
                return cls.instance
        else:
            if len(inspect.getfullargspec(cls.__init__)[0]) == 1:
                instance = type.__call__(cls)
                if (args or kwargs) and callable(instance):
                    return instance(*args, **kwargs)
                else:
                    return instance
            else:
                return type.__call__(cls, *args, **kwargs)


class Evaluation(metaclass=CallableSingletonMeta):
    instance = None

    def __init__(self):
        self.eng = matlab.engine.start_matlab('-nojvm')
        self.eng.addpath(self.eng.genpath('./matlab_lib'))
        self.strio = io.StringIO()
        Evaluation.instance: Evaluation = self

    def __call__(self, clean: np.ndarray, noisy: np.ndarray, fs: int) -> ODict:
        clean = matlab.double(clean.tolist())
        noisy = matlab.double(noisy.tolist())
        fs = matlab.double([fs])
        fwsegsnr, pesq, stoi = self.eng.se_eval(clean, noisy, fs,
                                                nargout=3, stdout=self.strio)

        return ODict([('fwSegSNR', fwsegsnr), ('PESQ', pesq), ('STOI', stoi)])

    def _exit(self):
        self.eng.quit()

        fname = datetime.now().strftime('log_matlab_eng_%Y-%m-%d %H.%M.%S.txt')
        with io.open(fname, 'w') as flog:
            self.strio.seek(0)
            shutil.copyfileobj(self.strio, flog)

        self.strio.close()

    def __del__(self):
        self._exit()