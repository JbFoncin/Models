# -*- coding: utf-8 -*-
cimport cython
from libc.math cimport log
from libc.math cimport exp as c_exp
from cython.parallel import prange
import scipy.optimize
from scipy.misc import derivative
from scipy.stats import norm
import numpy as np
import numdifftools as n
import matplotlib.pyplot as pp
from math import sqrt, exp
import pandas as pd

def converter(array) :
    """function returns buffer for fast 1-d array access"""
    cdef double[:] view
    if (type(array) != np.ndarray) & (array.dtype != "float64") :
        raise TypeError("only numpy arrays")
    else :
        view = array 
        return view[:]

cdef void _forbid_res(double[:] X) nogil:
    cdef int i, I
    I = X.shape[0]
    for i in range(I-1) :
        if X[i] < X[i+1] :
            X[i+1] = X[i]

def converter_2d(matrix) :
    """function returns buffer for fast 2d array access"""
    cdef double[:, :] view
    if (type(matrix) != np.ndarray) & (matrix.dtype != "float64") :
        raise TypeError("only numpy arrays")
    else :
        view = matrix
    return view

#function to estimate the variance of the cumulative hazard at a given point
def create_cum_hazard_error(t, intervals, sigma_hat):
    grad = np.zeros(shape=(1, intervals.shape[0] - 1))
    i = 0
    while t > intervals[i+1] :
        grad[0, i] += intervals[i+1] - intervals[i]
        i+=1
    grad[0, i] += t - intervals[i]
    e = grad.dot(sigma_hat).dot(grad.T)
    return e
    
#function to estimate the variance of the cumulative hazard at a given point with
#starting point different of zero
def create_cum_bay_hazard_error(t, actual, intervals, sigma_hat):
    grad = np.zeros(shape=(1, intervals.shape[0] - 1))
    i = 0
    while intervals[i+1] <= actual :
        i += 1
    if t < intervals[i+1] :
            grad[0, i] += (t-actual)
    else :
        grad[0, i] += (intervals[i+1]-actual)
        while t > intervals[i+1] :
            grad[0, i] += intervals[i+1] - intervals[i]
            i+=1
        grad[0, i] += t - intervals[i]
    e = grad.dot(sigma_hat).dot(grad.T)
    return e
#function to estimate the variance of the cumulative hazard at a given point
#with regression
def create_cum_prop_hazard_error(t, intervals, h, c_hat, explicatives, sigma_hat) :
    grad = np.zeros(shape = (1, sigma_hat.shape[0]))
    if type(explicatives) != np.array :    
        explicatives = np.array(explicatives)
    i = 0
    while t > intervals[i+1] :
        grad[0, i] -= exp(h[i]+(explicatives*c_hat).sum()) * (intervals[i+1] - intervals[i])
        i+=1
    grad[0, i] -= exp(h[i]+(explicatives*c_hat).sum()) * (t - intervals[i])
    temp = grad.sum()
    for j in range(c_hat.shape[0]) :
        grad[0, -j] -= explicatives[-j]*temp
    e = grad.dot(sigma_hat).dot(grad.T)
    return e
#function to estimate the variance of the cumulative hazard at a given point with
#starting point different of zero and explicative variables
def create_cum_prop_bay_hazard_error(t, actual, intervals, h, c_hat, explicatives, sigma_hat) :
    grad = np.zeros(shape = (1, sigma_hat.shape[0]))
    i = 0
    while intervals[i+1] <= actual :
        i += 1
    if t < intervals[i+1] :
        grad[0, i] += (t-actual) * exp(h[i]+(explicatives*c_hat).sum())
    else :
        grad[0, i] -= exp(h[i]+(explicatives*c_hat).sum()) * (intervals[i+1]-actual)
        while t > intervals[i+1] :
            grad[0, i] -= exp(h[i]+(explicatives*c_hat).sum()) * (intervals[i+1] - intervals[i])
            i+=1
        grad[0, i] -= exp(h[i]+(explicatives*c_hat).sum()) * (t - intervals[i])
    temp = grad.sum()
    for j in range(c_hat.shape[0]) :
        grad[0, -j] -= explicatives[-j]*temp
    e = grad.dot(sigma_hat).dot(grad.T)
    return e
        
@cython.boundscheck(False)
cdef double LL_onepiece(double[:] data, double[:] intervals,
                        double[:] values, double[:] finished) nogil:
    """parallel low-level likelihood func of the piecewise constant hazard model"""
    cdef double LL=0
    cdef int J = data.shape[0]
    cdef int I = intervals.shape[0]
    cdef int i, j
    for j in prange(J, schedule = "static", chunksize = 500) :
        for i in range(I-1) :
            if data[j] <= intervals[i+1] :
                LL += values[i] * (data[j] - intervals[i])
                LL += -log(values[i]) * finished[j]
                break
            else : 
                LL += values[i] * (intervals[i+1] - intervals[i])
    return LL

@cython.boundscheck(False)    
cdef double LL_reg_onepiece(double[:,:] data, double[:] intervals, double[:] values, int k,
                            double[:] Xc) nogil:
    """parallel low-level likelihood func of the piecewise constant hazard model
     with accelerated time values or proportionnal hazard (same likelihood)"""
    cdef double LL = 0
    cdef int J = data.shape[0]
    cdef int I = intervals.shape[0]
    cdef double[:] h_hat = values[:I-1]
    cdef double[:] c = values[I-1:]
    cdef int i, j, l
    for j in prange(J, schedule = "static", chunksize = 500) :
        for l in range(k) :
            Xc[j] += c[l]*data[j, 2+l]
        for i in range(I-1) :
            if data[j, 0] <= intervals[i+1] :
                LL += (c_exp(h_hat[i]+Xc[j])) * (data[j, 0] - intervals[i])
                LL += -(log(c_exp(h_hat[i]))+Xc[j]) * data[j, 1]
                break
            else : 
                LL += c_exp(h_hat[i] + Xc[j])* (intervals[i+1] - intervals[i])
    return LL

def cumulative_hazard(t, intervals, values) :
    """survival func for the piecewise constant hazard model"""
    H = 0
    j = 0
    while t > intervals[j+1]:
        H += values[j] * (intervals[j+1] - intervals[j])
        j += 1            
    H += values[j] * (t - intervals[j])
    return H

#class to reshape inputs and estimating the model without explicative variables
    
class _PCH_estimation :
    """subclass of user's tool PCH_estimator"""
    def __init__(self, data, intervals, finished, time = "") :
        if (type(data) == pd.DataFrame) & (type(finished) == str) & (time != "") :
            self._dta = data[[time]+[finished]]
            self._dta = self._dta[~pd.isnull(self._dta)]
            self.finshed = self._dta[finished].astype(float).values
            self.dta = self._dta[time].astype(float).values
        else : raise TypeError("incorrect input")
        self.intrvals = np.array(np.hstack((intervals, 1e100)), dtype = np.float)
        self.data = converter(self.dta)
        self.intervals = converter(self.intrvals)
        self.finished = converter(self.finshed)
    def _for_estimation(self, values) :
        values_ = np.array(values, dtype = np.float)
        values_view = converter(values_)
        return LL_onepiece(self.data, self.intervals, values_view, self.finished)
    def _fit(self):
        return scipy.optimize.minimize(self._for_estimation, np.zeros(shape = (self.intervals.shape[0]-1))+0.5,
                 bounds = [[1e-9, 1e9]]*(self.intervals.shape[0]-1), method = "L-BFGS-B")
    def get_results(self) :
        estimations = self._fit()
        self.h_hat = estimations["x"]
        self.neg_hessian = n.Hessian(self._for_estimation, step = 0.0001)(self.h_hat)
        self.sigma_hat = np.linalg.inv(self.neg_hessian)   
        return {"h_hat" : self.h_hat, "sigma_hat" : self.sigma_hat,
                "std_h_hat" : np.sqrt(np.diag(self.sigma_hat))}
                
#Object to be used to estimate model and get graphs, stats and forecasts        
class PCH_estimator :
    """python interface for low-level likelihood func optimisation"""
    def __init__(self, data, intervalle, finished, time = "") :
        self.results = _PCH_estimation(data, intervalle, finished, time).get_results()
        self._dta_max = data[time].max()
        self.intrvals = np.hstack((intervalle, 1e100))
    def graph_surv(self, dim = (10, 5), alpha = 5, nb_dots = 2500):
        """plot estimated survival with confidence intervals"""
        dataplot_x = np.linspace(0, np.max(self._dta_max)*1.2, nb_dots)[1:]
        err = [sqrt(create_cum_hazard_error(x,
                    self.intrvals, self.results["sigma_hat"])) for x in dataplot_x]
        Y = [cumulative_hazard(x, self.intrvals, self.results["h_hat"]) for x in dataplot_x]
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_down = np.maximum(CI_down, 0).astype(float)
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        dataplot_y = [exp(-y) for y in Y]
        dataplot_CI_down = np.array([exp(-up) for up in CI_up])
        dataplot_CI_up = np.array([exp(-down) for down in CI_down])
        _forbid_res(converter(dataplot_CI_down))
        _forbid_res(converter(dataplot_CI_up))
        pp.figure(figsize = dim)
        pp.style.use("fivethirtyeight")
        pp.xlim([0, self._dta_max*1.2])
        pp.ylim([-0.05, 1])
        pp.plot(dataplot_x, dataplot_CI_up, linewidth = 6, color = "c", linestyle = "--")
        pp.plot(dataplot_x, dataplot_CI_down, linewidth = 6, color = "c", linestyle = "--")        
        pp.plot(dataplot_x, dataplot_y, linewidth = 3, color = "k")
    def graph_hazard(self, dim = (10, 5), alpha = 5, y_max = None) :
        """plot estimated hasard with confidence intervals"""
        dataplot_x = list(self.intrvals[:-1])*2 ; dataplot_x.sort() ; dataplot_x.append(1e100)
        CI_down = []
        CI_up = []
        for i in range(len(self.results["h_hat"])) :
            for j in range(2) :
                CI_down.append(norm.ppf(alpha/200, loc = self.results["h_hat"][i], 
                           scale = sqrt(self.results["sigma_hat"][i, i])))
                CI_up.append(norm.ppf(1-alpha/200, loc = self.results["h_hat"][i], 
                           scale = sqrt(self.results["sigma_hat"][i, i])))
        dataplot_y = []
        for i in self.results["h_hat"] :
            for j in range(2) :
                dataplot_y.append(i)
        dataplot_y =[0] + dataplot_y
        CI_down = [0] + CI_down
        CI_up = [0] + CI_up
        CI_down = np.maximum(CI_down, 0)
        if y_max == None :
            y_max = np.max(CI_up)*1.2
        pp.figure(figsize = dim)
        pp.style.use("fivethirtyeight")
        pp.xlim([0, self._dta_max])
        pp.ylim([0, y_max])
        pp.plot(dataplot_x, CI_up, linewidth = 3, color = "c", linestyle = "--" )
        pp.plot(dataplot_x, CI_down, linewidth = 3, color = "c", linestyle = "--")
        pp.plot(dataplot_x, dataplot_y, linewidth = 3, color = "k")
    def get_stats(self, alpha = 5, stats=[0.5], nb_dots = 2500):
        """returns mean, median and other stats with custom precision"""
        X = np.linspace(0, self._dta_max*1.2, nb_dots)[1:]
        err = [sqrt(create_cum_hazard_error(x,
                    self.intrvals, self.results["sigma_hat"])) for x in X]
        Y = [cumulative_hazard(x, self.intrvals, self.results["h_hat"]) for x in X]
        Y = np.array(Y)
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_up = np.maximum(CI_up,0)
        y_up = np.exp(-np.array(CI_up))
        _forbid_res(converter(y_up))
        y_down = np.exp(-np.array(CI_down))
        _forbid_res(converter(y_down))
        E = sum([(X[i+1]-X[i])*exp(-Y[i+1]) for i in range(nb_dots-2)])
        E_down = sum([(X[i+1]-X[i])*y_down[i+1] for i in range(nb_dots-2)])
        E_up = sum([(X[i+1]-X[i])*y_up[i+1] for i in range(nb_dots-2)])
        results = [{"E" : E, "E_up "+str(alpha/2)+"%" : E_up,
                    "E_down "+str(alpha/2)+"%" : E_down}]
        for stat in stats : 
            i = np.argmin((np.exp(-Y)-stat)**2)
            j = np.argmin((y_up-stat)**2)
            k = np.argmin((y_down-stat)**2)
            results.append({str(stat*100)+"%" : X[i],
                            str(alpha/2)+"% up" : X[j],
                             str(alpha/2)+"% down" : X[k]})
        return results
    def forecast(self, duration, alpha = 5,  stats=[0.5], nb_dots = 2500) :
        """returns P(t>T|t) by the bayesian formula, may fail if p(t > T) is around zero"""
        X = np.linspace(duration, self._dta_max*1.2, nb_dots)[1:]
        p_t = exp(-cumulative_hazard(duration, self.intrvals, self.results["h_hat"]))
        K = 2
        while exp(-cumulative_hazard(X[-1], self.intrvals, self.results["h_hat"]))/p_t > 0.02 :
            X = np.linspace(duration, self._dta_max*1.2*K, nb_dots)[1:]
            K += 1
        err = [sqrt(create_cum_bay_hazard_error(x, duration,
                    self.intrvals, self.results["sigma_hat"])) for x in X]
        Y = np.exp([-cumulative_hazard(x, self.intrvals, self.results["h_hat"]) for x in X])/p_t
        _Y = [cumulative_hazard(x, self.intrvals, self.results["h_hat"]) for x in X]
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(_Y, err)]
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(_Y, err)]
        CI_up = np.maximum(CI_up, 0)
        CI_down = np.array(CI_down)
        y_up = np.exp(-np.array(CI_up))/p_t
        _forbid_res(converter(y_up))
        y_down = np.exp(-np.array(CI_down))/p_t
        _forbid_res(converter(y_down))
        E = sum([(X[i+1]-X[i])*Y[i+1] for i in range(nb_dots-2)])
        E_down = sum([(X[i+1]-X[i])* y_down[i+1] for i in range(nb_dots-2)])
        E_up = sum([(X[i+1]-X[i])*y_up[i+1] for i in range(nb_dots-2)])
        results = [{"E" : E+duration , "E_up "+str(alpha/2)+"%" : E_up+duration,
                    "E_down "+str(alpha/2)+"%" : E_down+duration}]
        for stat in stats : 
            i = np.argmin((Y-stat)**2)
            j = np.argmin((y_up-stat)**2)
            k = np.argmin((y_down-stat)**2)
            results.append({str(stat*100)+"%" : X[i],
                            str(alpha/2)+"% up" : X[j],
                             str(alpha/2)+"% down" : X[k]})
        return results
class _PCH_regression :
    """python interface for low-level likelihood optimisation including regression."""
    def __init__(self, data, intervals, finished, explicatives, time = "") :
        if (type(data) == pd.DataFrame) & (type(finished) == str) & \
            ((type(explicatives) == str)|(type(explicatives) == list)) & (time != "") :
            if type(explicatives) == str :
                self.k = 1
                temp = data[[time, finished, explicatives]].dropna(axis = 0)
            else : 
                self.k = len(explicatives)
                temp = data[explicatives+[time]+[finished]].dropna(axis = 0)
            self.dta = temp[time].astype(float).values
            self.finshed = temp[finished].astype(float).values
            self.explictives = temp[explicatives].astype(float).values 
            if self.k == 1 :
                self.means = self.explictives.mean()
                self.explictives -= self.explictives.mean()
            else : 
                for i in range(self.k) :
                    self.explictives[:, i] -= self.explictives[:, i].mean()
        else : raise TypeError("invalid input")
        self.intrvals = np.array(np.hstack((intervals, 1e100)), dtype = np.float)
        if self.k == 1 :
            self.total = np.vstack((self.dta, self.finshed, self.explictives)).T
        else : 
            self.total = np.hstack((np.vstack((self.dta, self.finshed)).T, self.explictives))
        self.total = np.array(self.total, order = "F")
        self.total_view = converter_2d(self.total)
        self.intervals_view = converter(self.intrvals)
    def _for_estimation(self, values) :
        values_ = np.array(values, dtype = np.float)
        values_view = converter(values_)
        reg_container = np.zeros(shape = self.total_view.shape[0], dtype = float)
        reg_container_view = converter(reg_container)
        return LL_reg_onepiece(self.total_view, self.intervals_view ,
                               values_view, self.k, reg_container_view)
    def _fit(self):
        return scipy.optimize.minimize(self._for_estimation,
                                      np.zeros(shape = (self.intervals_view.shape[0]+self.k-1),
                                      dtype = float),  method = "L-BFGS-B")
    def get_results(self) : 
        estimations = self._fit()
        h_hat = np.exp(estimations["x"][:-self.k])
        c_hat = estimations["x"][-self.k:]
        neg_hessian = n.Hessian(self._for_estimation, step = 0.001)(estimations["x"])
        sigma_hat = np.linalg.inv(neg_hessian)
        return {"h_hat" : h_hat, "c_hat" : c_hat, "sigma_hat" : sigma_hat, 
                "std_h_hat" : np.sqrt(np.diag(sigma_hat)[:-self.k]*h_hat**2),
                "theta_hat" : estimations["x"][:-self.k],
                "std_c_hat" : np.sqrt(np.diag(sigma_hat)[-self.k:])}


        
class PCH_regressor :
    def __init__(self, data, intervalle, finished, explicatives, time = "") :
        self.results = _PCH_regression(data, intervalle, finished, explicatives, time).get_results()
        if (type(data) == pd.DataFrame) & (time != "") :
            self._dta_max = np.max(data[time])
            if type(explicatives) == str :
                self.k = 1
                self.means = data[explicatives].dropna(axis = 0).mean()
            else : 
                self.k = len(explicatives)
                self.means = data[explicatives].dropna(axis = 0).mean().values
        else : 
            raise TypeError("Invalid input")   
        self.intrvals = np.hstack((intervalle, 1e100))
    def graph_common_surv(self, dim = (10, 5), alpha = 5, nb_dots = 2500):
        explicatives_ = np.zeros(shape = self.k)
        Xc = exp(sum(explicatives_ * self.results["c_hat"]))
        X = np.linspace(0, self._dta_max*1.2, nb_dots)[1:]
        nb_dots = X.shape[0]
        err = [sqrt(create_cum_prop_hazard_error(x,
                    self.intrvals, self.results["theta_hat"], self.results["c_hat"],
                    explicatives_,
                    self.results["sigma_hat"])) for x in X]
        Y = [cumulative_hazard(x, self.intrvals, self.results["h_hat"]) for x in X]
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_up = np.maximum(CI_up, 0)
        CI_down = np.array(CI_down)
        dataplot_y = np.exp(-np.array(Y))
        dataplot_ci_down = np.exp(-CI_down)
        _forbid_res(converter(dataplot_ci_down))
        dataplot_ci_up = np.exp(-CI_up)
        _forbid_res(converter(dataplot_ci_up))
        pp.figure(figsize = dim)
        pp.style.use("fivethirtyeight")
        pp.xlim([0, self._dta_max*1.2])
        pp.ylim([-0.05, 1.1])
        pp.plot(X, dataplot_ci_down, linewidth = 6, color = "c", linestyle = "--")
        pp.plot(X, dataplot_ci_up, linewidth = 6, color = "c", linestyle = "--")        
        pp.plot(X, dataplot_y, linewidth = 3, color = "k")
    def graph_common_hazard(self, dim = (10, 5), alpha = 5, y_max = None) :
        dataplot_x = list(self.intrvals[:-1])*2 ; dataplot_x.sort() ; dataplot_x.append(1e100)
        CI_down = []
        CI_up = []
        for i in range(len(self.results["h_hat"])) :
            for j in range(2) :
                CI_down.append(norm.ppf(alpha/200, loc = self.results["h_hat"][i], 
                           scale = self.results["std_h_hat"][i]))
                CI_up.append(norm.ppf(1-alpha/200, loc = self.results["h_hat"][i], 
                           scale = self.results["std_h_hat"][i]))
        
        dataplot_y = []
        for i in self.results["h_hat"] :
            for j in range(2) :
                dataplot_y.append(i)
        dataplot_y =[0] + dataplot_y
        CI_down = [0] + CI_down
        CI_up = [0] + CI_up
        CI_down = np.maximum(CI_down, 0)
        if y_max == None :
            y_max = np.max(CI_up)*1.2
        pp.figure(figsize = dim)
        pp.style.use("fivethirtyeight")
        pp.xlim([0, self._dta_max*1.2])
        pp.ylim([-0.05, y_max])
        pp.plot(dataplot_x, CI_up, linewidth = 3, color = "c", linestyle = "--" )
        pp.plot(dataplot_x, CI_down, linewidth = 3, color = "c", linestyle = "--")
        pp.plot(dataplot_x, dataplot_y, linewidth = 3, color = "k")
    def get_stats(self, explicatives = [], alpha = 5, stats=[0.5], nb_dots = 1500):
        """returns mean, median and other stats with custom precision"""
        if (type(explicatives) == list) & (len(explicatives) == 0) :
            explicatives_ = np.zeros(shape = self.k)
        else :
            explicatives_ = np.array(explicatives) - self.means
        Xc = exp(sum(explicatives_ * self.results["c_hat"]))
        X = np.linspace(0, self._dta_max*1.2, nb_dots)[1:]
        ytest = exp(-cumulative_hazard(X[-1]*Xc, self.intrvals, self.results["h_hat"]))
        K=2
        while ytest > 0.02 :
            X = np.linspace(0, self._dta_max*1.2*K, nb_dots)[1:]
            ytest = exp(-cumulative_hazard(X[-1]*Xc, self.intrvals, self.results["h_hat"])) 
            K += 1
        err = [sqrt(create_cum_prop_hazard_error(x,
                    self.intrvals, self.results["theta_hat"], self.results["c_hat"],
                    explicatives_,
                    self.results["sigma_hat"])) for x in X]
        Y = [cumulative_hazard(x*Xc, self.intrvals, self.results["h_hat"]) for x in X]
        Y = np.array(Y)
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_down = np.array(CI_down, dtype = float)
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(Y, err)]
        CI_up = np.maximum(CI_up, 0).astype(float)
        y_up = np.exp(-CI_up)
        _forbid_res(converter(y_up))
        y_down = np.exp(-CI_down)
        _forbid_res(converter(y_down))
        E = sum([(X[i+1]-X[i])*exp(-Y[i+1]) for i in range(nb_dots-2)])
        E_down = sum([(X[i+1]-X[i])*y_down[i+1] for i in range(nb_dots-2)])
        E_up = sum([(X[i+1]-X[i])*y_up[i+1] for i in range(nb_dots-2)])
        results = [{"E" : E, "E_up "+str(alpha/2)+"%" : E_up,
                    "E_down "+str(alpha/2)+"%" : E_down}]
        for stat in stats : 
            i = np.argmin((np.exp(-Y)-stat)**2)
            j = np.argmin((y_up-stat)**2)
            if j == 0 : 
                j = nb_dots-2
            k = np.argmin((y_down-stat)**2) 
            results.append({str(stat*100)+"%" : X[i],
                            str(alpha/2)+"% up" : X[j],
                             str(alpha/2)+"% down" : X[k]})
        return results
    def forecast(self, duration, explicatives = [], alpha = 5,  stats=[0.5], nb_dots = 2500) :
        """returns P(t>T|t) by the bayesian formula"""
        if (type(explicatives) == list) & (len(explicatives) == 0) :
            explicatives_ = np.zeros(shape = self.k)
        else :
            explicatives_ = np.array(explicatives) - self.means
        Xc = exp(sum(explicatives_ * self.results["c_hat"]))
        X = np.linspace(duration, self._dta_max*1.2, nb_dots)[1:]
        p_t = exp(-cumulative_hazard(duration*Xc, self.intrvals, self.results["h_hat"]))
        ytest = exp(-cumulative_hazard(X[-1]*Xc, self.intrvals, self.results["h_hat"]))/p_t
        K=2
        while ytest > 0.02 :
            X = np.linspace(duration, self._dta_max*1.2*K, nb_dots)[1:]
            ytest = exp(-cumulative_hazard(X[-1]*Xc, self.intrvals, self.results["h_hat"]))/p_t
            K += 1
        err = [sqrt(create_cum_prop_bay_hazard_error(x, duration,
                    self.intrvals, self.results["theta_hat"], self.results["c_hat"],
                    explicatives_,
                    self.results["sigma_hat"])) for x in X]
        Y = np.exp([-cumulative_hazard(x*Xc, self.intrvals, self.results["h_hat"]) for x in X])/p_t
        _Y = [cumulative_hazard(x*Xc, self.intrvals, self.results["h_hat"]) for x in X]
        CI_down = [norm.ppf(1-alpha/200, loc = y, scale = e) for y, e in zip(_Y, err)]
        CI_down = np.array(CI_down, dtype = float)
        CI_up = [norm.ppf(alpha/200, loc = y, scale = e) for y, e in zip(_Y, err)]
        CI_up = np.maximum(CI_up, 0)
        y_up = np.exp(-CI_up)/p_t
        _forbid_res(converter(y_up))
        y_down = np.exp(-CI_down)/p_t
        _forbid_res(converter(y_down))
        E = sum([(X[i+1]-X[i])*Y[i+1] for i in range(nb_dots-2)])
        E_down = sum([(X[i+1]-X[i])*y_down[i+1] for i in range(nb_dots-2)])
        E_up = sum([(X[i+1]-X[i])*y_up[i+1] for i in range(nb_dots-2)])
        results = [{"E" : E+duration, "E_up "+str(alpha/2)+"%" : E_up+duration,
                    "E_down "+str(alpha/2)+"%" : E_down+duration}]
        for stat in stats : 
            i = np.argmin((Y-stat)**2)
            j = np.argmin((y_up-stat)**2)
            if j == 0 : 
                j = nb_dots-2
            k = np.argmin((y_down-stat)**2)
            results.append({str(stat*100)+"%" : X[i],
                            str(alpha/2)+"% up" : X[j],
                             str(alpha/2)+"% down" : X[k]})
        return results
