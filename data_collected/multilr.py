import pandas as pd
import math
import numpy as np
from sklearn.linear_model import  LinearRegression as lr

from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np

class ConstrainedLinearRegression(LinearModel, RegressorMixin):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, nonnegative=False, tol=1e-15):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.tol = tol

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X)
        self.min_coef_ = min_coef if min_coef is not None else np.repeat(-np.inf, X.shape[1])
        self.max_coef_ = max_coef if max_coef is not None else np.repeat(np.inf, X.shape[1])
        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        while not (np.abs(prev_beta - beta)<self.tol).all():
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X,beta) - y, X)
                beta[i] = np.minimum(self.max_coef_[i],
                                     np.maximum(self.min_coef_[i],
                                                beta[i]-grad[i] / hessian[i,i]))

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


data = pd.read_csv("vivado_energy.csv")
data = np.array(data.values.tolist())
# print(data)
print(data[:, 0:4], data[:, 4])
# print(data.head())
# median=math.floor(data.DSP.median())
# print(median)

model = ConstrainedLinearRegression(nonnegative=True)
model.fit(data[:, 0:4], data[:, 4])
print(model.intercept_)
print(model.coef_)
# print(data.head())
# print(model.predict([[1,1,1,1]]))
# reg = lr()
# reg.fit(data[['BRAM','DSP','FF', 'LUT']], data.Power)

# m= reg.coef_
# b= reg.intercept_

# # print(reg.predict([[1308,    118, 106679,  208976]]))
df = pd.read_csv('vgg.csv')
# print(df.head())
pred_data = df.values.tolist()
pred = []
for i in range(len(pred_data)):
    print(pred_data[i])
    pred.append(model.predict([pred_data[i]]))
for i in range(len(pred)):
    print(np.round(pred[i], 3)[0])
#     print(pred[i])