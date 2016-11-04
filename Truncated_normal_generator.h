#ifndef TRUNCATED_NORMAL_GENERATOR_H
#define TRUNCATED_NORMAL_GENERATOR_H
#include <boost/math/special_functions.hpp>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace TNG
{
Eigen::MatrixXd dot(Eigen::MatrixXd& first,Eigen::MatrixXd& second) ;
Eigen::MatrixXd transpose(Eigen::MatrixXd mat) ;
Eigen::MatrixXd gen_select_mat(long int j, long int J) ;
Eigen::MatrixXd create_L_j(Eigen::MatrixXd& omega,long int j) ;
Eigen::MatrixXd create_mat(long int rows,long int cols) ;
Eigen::MatrixXd create_Xb(Eigen::MatrixXd& X, Eigen::MatrixXd& beta,long int j,long int J) ;
double std_normal_ppf(double& x) ;
double std_normal_density(double& x) ;
double std_normal_cdf(double& x);
class Truncated_normal_generator
{
    public:
        Truncated_normal_generator();
        double create(double& point);
        ~Truncated_normal_generator();
        std::random_device m_rd ;
        std::mt19937 m_gen ;
        std::uniform_real_distribution<double> m_distrib ;
};
class Omega
{
public :
    Omega(long int J) ;
    void input(double value) ;
    Eigen::MatrixXd get_omega() ;
    ~Omega() ;
    Eigen::MatrixXd L;
    long int i, j, m_J ;
    };
double individual_likelihood(Eigen::MatrixXd& X, Eigen::MatrixXd& beta, Eigen::MatrixXd& L_j, long int nb_draw, long int y, Truncated_normal_generator& gen) ;
}
#endif // TRUNCATED_NORMAL_GENERATOR_H
