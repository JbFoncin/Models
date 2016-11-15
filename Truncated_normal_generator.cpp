#include "Truncated_normal_generator.h"

using namespace std;
namespace bm = boost::math ;
namespace e = Eigen ;
namespace TNG
{
    //Not used anymore, will be deleted in next update
    e::MatrixXd create_mat(long int rows, long int cols)
    {
         e::MatrixXd result = e::MatrixXd::Zero(rows, cols) ;
         return result ;
    }
    //generates the product of variables and coefficients, then apply the dimension transform
    e::MatrixXd create_Xb(e::MatrixXd& X, e::MatrixXd& beta, long int j, long int J)
    {
        e::MatrixXd temp = X*beta ;
        e:: MatrixXd result = gen_select_mat(j, J)*temp.transpose() ;
        return result ;
    }
    // Generates the matrix used for dimension transform
    e::MatrixXd gen_select_mat(long int j, long int J)
    {
        e::MatrixXd result = e::MatrixXd::Zero(J-1, J) ;
        int i(0), k(0), l ;
        while (k<j)
        {
            result(i, k) = 1.0; i++; k++;
        }
        for (l=0 ; l<(J-1); l++)
        {
            result(l, k) = -1.0;
        }
        k++ ;
        while (k < J)
        {
            result(i, k) = 1.0 ; i++ ; k++ ;
        }
        return result ;
    }
    //wrapper for eigen dot
    e::MatrixXd dot(e::MatrixXd& first, e::MatrixXd& second)
    {
        e::MatrixXd result = (first*second) ;
        return result ;
    }
    e::MatrixXd transpose(e::MatrixXd mat)
    {
        e::MatrixXd result = mat.transpose() ;
        return result ;
    }
    e::MatrixXd create_L_j(e::MatrixXd& omega, long int j)
    {
        e::MatrixXd temp, result ;
        temp = gen_select_mat(j, omega.cols()) ;
        result = temp*omega*temp.transpose() ;
        e::LDLT <e::MatrixXd> llt(result) ;
        result = llt.matrixL() ;
        return result ;
    }
    double std_normal_ppf(double& value)
    {
        return sqrt(2)*bm::erf_inv<double>(2*value-1.0) ;
    }
    //Not used anymore, will be deleted
    double std_normal_density(double& x)
    {
        double result = 1.0/(sqrt(2.0 * 3.141592)) * exp(-(x*x)/2.0);
        if (result > 0.000001)
        {
            return result ;
        }
        else
        {
            return 0.000001 ;
        }

    }
    //Please note the extreme values are bound to avoid NaNs
    double std_normal_cdf(double& x)
    {
        double result = (1.0+erf(x/sqrt(2.0)))/2.0 ;
        if (result < 0.00000001)
        {
         return 0.00000001 ;
        }
        else if (result > 0.99999999)
        {
            return 0.99999999 ;
        }
        else
        {
            return result ;
        }
    }
    Truncated_normal_generator::Truncated_normal_generator() : m_gen(m_rd()),m_distrib(0.99999999, 0.00000001)
    {

    }
    double Truncated_normal_generator::create(double& point)
    {
            double temp(m_distrib(m_gen)*std_normal_cdf(point)) ;
            return std_normal_ppf(temp) ;
    }

    Truncated_normal_generator::~Truncated_normal_generator()
    {

    }
    Omega::Omega(long int J) : i(2), j(0), m_J(J)
    {
        L = e::MatrixXd::Zero(m_J, m_J) ;
        for (int I(0); I<m_J ; I++)
        {
            L(I, I) = 1.0 ;
        }
        L(1, 0) = 1.0 ;

    }

    void Omega::input(double value)
    {
        if (j == i)
        {
            j = 0; i++;
        }
        L(i, j) = value ; j++;
    }
    e::MatrixXd Omega::get_omega()
    {
        e::MatrixXd LL = L*L.transpose();
        return LL ;
    }
    Omega::~Omega()
    {

    }

double individual_likelihood(e::MatrixXd& X, e::MatrixXd& beta, e::MatrixXd& L_j, long int nb_draw, long int y, Truncated_normal_generator& gen)
    {
    const long int J_1(beta.cols()-1) ;
    e::MatrixXd Xb = create_Xb(X, beta, y, beta.cols()) ;
    double sample[J_1] ;
    double truncations[J_1] ;
    double nb_draw_f = static_cast<double>(nb_draw) ;
    double tmp, tmp2, result(0) ;
    //tmp is a variable containing product of covariance parameter and previous samples
    //tmp2 contains the product of cdf draws
    int i, j , k ;
    for (j = 0; j < nb_draw ; j++)
    {
        for (i = 0; i < (J_1) ; i++)
        {
            tmp = 0 ;
            k = 0 ;
            while (k<i)
            {
                tmp += L_j(i, k) * sample[k] ;
                k += 1;
            }
            truncations[i] = -(Xb(i,0)+tmp)/L_j(i, i);
            sample[i] = gen.create(truncations[i]);
            while (sample[i] > truncations[i])
            {
                sample[i] += gen.create(truncations[i]) ;
            }

        }
        tmp2 = 1.0 ;
        for (i = 0; i< (J_1); i++)
        {
            tmp2 *= std_normal_cdf(truncations[i]) ;
        }
        result += tmp2 ;
    }
    return result/nb_draw_f ;
}
}
