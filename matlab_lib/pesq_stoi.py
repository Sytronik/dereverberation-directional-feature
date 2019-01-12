import matlab
import matlab.engine

import atexit
import io
from datetime import datetime
from typing import Dict

import numpy as np

import shutil


class PESQ_STOI:
    def __init__(self):
        self.eng = matlab.engine.start_matlab('-nojvm')
        self.eng.addpath(self.eng.genpath('.'))
        self.strio = io.StringIO()
        atexit.register(self._exit)

    def __call__(self, clean: np.ndarray, noisy: np.ndarray, fs: int) -> Dict[str, float]:
        clean = matlab.double(clean.tolist())
        noisy = matlab.double(noisy.tolist())
        fs = matlab.double([fs])
        pesq, stoi = self.eng.se_eval(clean, noisy, fs, nargout=2, stdout=self.strio)

        return dict(PESQ=pesq, STOI=stoi)

    def _exit(self):
        self.eng.quit()

        with io.open(
                datetime.now().strftime('log_matlab_eng_%Y-%m-%d %H.%M.%S.txt'),
                'w', encoding='UTF-8') as flog:
            self.strio.seek(0)
            shutil.copyfileobj(self.strio, flog)

        self.strio.close()
