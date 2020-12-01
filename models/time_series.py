# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:56:38 2020

@author: Yubo
"""

from statsmodels.tsa.arima_model import ARIMA
from numpy.linalg import LinAlgError

class AutoARIMA:
    """
    AutoARIMA model to automatically find best p, d, q using aic or bic
    """
    def __init__(self, p_range=(0, 1), d_range=(0, 1), 
                 q_range=(0, 1), criteria='bic', no_print=True):
        self._p, self._d, self._q = p_range, d_range, q_range
        self._criteria = criteria.lower()
        self._no_print = no_print
        self.model_ = None
        self.best_bic_ = float("inf")
        self.best_aic_ = float("inf")
        self.order_ = None
        
        
    def fit(self, data):
        _p, _d, _q = self._p, self._d, self._q
        
        for p in _p:
            for d in _d:
                for q in _q:
                    try:
                        model = ARIMA(data, (p, d, q)).fit()
                    except (ValueError, LinAlgError) as e:
                        if not self._no_print:
                            print("when order is ({},{},{}), not stationary".format(p,d,q))
                        continue
                    if self._criteria == 'bic':
                        if model.bic < self.best_bic_:
                            self.best_bic_ = model.bic
                            self.order_ = (p, d, q)
                            self.model_ = model
                    elif self._criteria == 'aic':
                        if model.aic < self.best_aic_:
                            self.best_aic_ = model.aic
                            self.order_ = (p, d, q)
                            self.model_ = model
                        
        return self.model_
    

