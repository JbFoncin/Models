# -*- coding: utf-8 -*-
cdef extern from "Dense" namespace "Eigen" nogil:
    cdef cppclass MatrixXd :
        MatrixXd() 
        MatrixXd(int J, int K)         
        double& element "operator()"(int row,int col) 

cdef extern from "Truncated_normal_generator.h" namespace "TNG" nogil :
    cdef inline MatrixXd gen_select_mat(long int j, long int J) 
    cdef inline MatrixXd create_Xb(MatrixXd& X, MatrixXd& beta, long int j, long int J) 
    cdef inline MatrixXd create_L_j(MatrixXd& omega, long int j) 
    cdef cppclass Truncated_normal_generator :
        Truncated_normal_generator()  
        double create(double& point)  
    cdef cppclass Omega : 
        Omega(long int J)   
        void input(double value)   
        MatrixXd get_omega()   
    cdef inline double individual_likelihood(MatrixXd& X, MatrixXd& beta,
                                      MatrixXd& L_j, long int nb_draw,
                                      long int y, Truncated_normal_generator& gen)  
        
