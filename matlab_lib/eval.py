try:
    import matlab
    import matlab.engine
finally:
    pass

import atexit
import io
from collections import OrderedDict as ODict
from datetime import datetime
import shutil
from typing import Dict

import numpy as np


class Evaluation:
    exist_instance = False

    def __init__(self):
        assert not self.exist_instance
        self.eng = matlab.engine.start_matlab('-nojvm')
        self.eng.addpath(self.eng.genpath('.'))
        self.strio = io.StringIO()
        atexit.register(self._exit)
        self.exist_instance = True

    def __call__(self, clean: np.ndarray, noisy: np.ndarray, fs: int) -> ODict:
        clean = matlab.double(clean.tolist())
        noisy = matlab.double(noisy.tolist())
        fs = matlab.double([fs])
        fwsegsnr, pesq, stoi = self.eng.se_eval(clean, noisy, fs, nargout=3, stdout=self.strio)

        return ODict([('fwSegSNR', fwsegsnr), ('PESQ', pesq), ('STOI', stoi)])

    def _exit(self):
        self.eng.quit()

        with io.open(
                datetime.now().strftime('log_matlab_eng_%Y-%m-%d %H.%M.%S.txt'),
                'w', encoding='UTF-8') as flog:
            self.strio.seek(0)
            shutil.copyfileobj(self.strio, flog)

        self.strio.close()
