# -*- coding: utf-8 -*-
from GHK_tools cimport *
import numpy as np
import pandas as pd
from libcpp.vector cimport vector
from cython.parallel import prange
cimport cython
import numdifftools as n
from pybrain.optimization import SNES
from libc.math cimport log

cdef void set_matrix_element(MatrixXd& m, long int row, long int col, double elm) nogil:
    cdef double* d = <double*> &(m.element(row,col))
    d[0] = elm

def converter_2d(matrix) :
    """function returns buffer for fast 2d array access"""
    cdef double[:, :] view
    if (type(matrix) != np.ndarray) & (matrix.dtype != "float64") :
        raise TypeError("only numpy arrays")
    else :
        view = matrix
    return view
    
def converter_int(array) :
    """function returns buffer for fast 1-d array access"""
    cdef long int[:] view
    if (type(array) != np.ndarray) & (array.dtype != int) :
        raise TypeError("only numpy arrays")
    else :
        view = array 
        return view[:]

def converter(array) :
    """function returns buffer for fast 1-d array access"""
    cdef double[:] view
    if (type(array) != np.ndarray) & (array.dtype != "float64") :
        raise TypeError("only numpy arrays")
    else :
        view = array 
        return view[:]
        
cdef MatrixXd from_view_to_mat(double[:,:] view, long int& row)  nogil:
    cdef long int i
    cdef MatrixXd result = MatrixXd(1, view.shape[1])
    for i in range(view.shape[1]) :
        set_matrix_element(result, 0, i, view[row, i])
    return result

    
cdef double wrap_individual_likelihood(double[:, :] X, long int row, long int y, MatrixXd L_j, MatrixXd beta,
                                  Truncated_normal_generator& gen, long int nb_draw) nogil:
    cdef MatrixXd X_mat = from_view_to_mat(X, row)
    return individual_likelihood(X_mat, beta, L_j, nb_draw, y, gen)

@cython.boundscheck(False)
cdef double likelihood(double[:] params, long int[:] Y, double[:, :] data,
                       int J, long int nb_draw) nogil:
    cdef long int K = data.shape[1]
    cdef long int N = data.shape[0]
    cdef double N_d = data.shape[0]
    cdef double result = 0.0
    cdef long int sel = 0
    cdef long int i, j, k
    cdef Truncated_normal_generator gen
    cdef Omega* omega = new Omega(J)
    cdef MatrixXd beta = MatrixXd(K, J)
    for j in range((J-1)) :
        for i in range(K) :
            set_matrix_element(beta, i, j+1, params[sel])
            sel+=1
    cdef long int nb_params_O = (((J-1)*J)/2)-1
    for i in range(nb_params_O) :
        omega.input(params[sel])
        sel+=1
    cdef MatrixXd _omega = omega.get_omega()
    del omega
    cdef vector[MatrixXd] L_j = vector[MatrixXd](J)
    for j in range(J) :
        L_j[j] = create_L_j(_omega, j)
    for i in prange(N, schedule = "static", chunksize = 250)  :
        result -= log(wrap_individual_likelihood(data, i, Y[i], L_j[Y[i]], beta, gen, nb_draw))
    return result/N_d
    
class mnp_estimator:
    def __init__(self, Y, X, data) :
        """X is the list of names for explicative variable, Y the name of the target value
        and data for the dataset"""
        if type(X) == str :
            _data = data[[Y]+[X]].dropna(how = "any", axis = 0)
            self.labels = [X]
        if type(X) == list :
            _data = data[[Y]+X].dropna(how = "any", axis = 0)
            self.labels = X
        self.data = np.array(np.hstack((np.ones(shape = (_data.shape[0],1)), data[X].values)),
                                 order = "F", dtype = float)                              
        self.data_view = converter_2d(self.data)
        self.Y = np.array(_data[Y].values.flatten(), dtype = int)
        self.Y_view = converter_int(self.Y)        
        self.J = data[Y].drop_duplicates().shape[0]
        self.K = self.data_view.shape[1]
        self.nb_draw = 5
    def _for_estimation(self, params) :
        params = np.array(params, dtype = float)
        params_view = converter(params)
        return likelihood(params_view, self.Y_view, self.data_view, self.J , self.nb_draw)
    def start_fit(self, nb_draw, maxiter) :
        self.nb_draw = nb_draw
        self.es = SNES(self._for_estimation,
                        [0.0]*int(-1+self.K*(self.J-1)+(((self.J-1)*self.J)/2)),
                        minimize = True, verbose = True, batchSize = 100) 
        self.es.maxEvaluations = maxiter
        self.es.learn()
    def get_params(self) :
        labels_beta = []
        for j in range(self.J-1) :
            for label in ["const"] + self.labels :
                labels_beta.append("beta"+str(j+1)+" "+label)
        beta = {labels_beta[i] : self.es.bestEvaluable[i] for i in range(len(labels_beta))}
        covar_mat = np.eye(self.J, self.J, dtype = float)
        covar_mat[1, 0] = 1
        np.fill_diagonal(covar_mat, 1)
        select = (self.J-1)*self.K
        i, j = 2, 0
        while (i < self.J) :
            while (j < (i)) :
                covar_mat[i, j] = self.es.bestEvaluable[select]
                select += 1
                j += 1
            i += 1
        return beta, covar_mat.dot(covar_mat.T)
            
            
