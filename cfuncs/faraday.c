// Licensed under a 3-clause BSD style license - see LICENSE

#include "common.h"
#include "math.h"
#include "edist.h"
#include "synch_utils.h"

double p0_f(double gamma){
    return sqrt(POW2(gamma)-1);
}

double XA_f(double nu, double B, double theta) {

    double gyr_f = get_nu_B(B);
    return 1e2*sqrt((M_SQRT2*sin(theta)*gyr_f)/(nu));
}

double HX_f(double gamma, double nu, double B, double theta){

    double XA = XA_f(nu, B, theta);

    if (XA*gamma < 40.){
        return 9.29e-9*sqrt(1-1./gamma)*pow(XA*gamma, 3.036);
    } else {
        return -0.000203*pow(XA*gamma, 0.4343)
                -0.0013*cos(0.5646*log(XA*gamma)-4.03)
                +0.002*exp(-POW2(log(XA*gamma)-4.2137)/0.5429)
                +0.00083*exp(-POW2(log(XA*gamma)-4.2137)/0.2121);
    }

}

double HB_f(double gamma, double nu, double B, double theta){

    double XA = XA_f(nu, B, theta);

    if (XA*gamma < 40.){
        return 4.67e-9*pow(1-1./gamma, 1.5)*pow(XA*gamma, 3.84);
    } else {
        double logXAgam2 = POW2(log(XA*gamma));
        return 0.864-0.2082*logXAgam2+0.0175*POW2(logXAgam2)
                    -0.000626*pow(logXAgam2,3)+1.0175e-5*pow(logXAgam2,4)
                    -7.686e-8*pow(logXAgam2,5)
                -0.01*exp(-POW2(log(XA*gamma)-4.0755)/0.0763);
    }

}

double gX_f(double gamma, double nu, double B, double theta){

    double XA = XA_f(nu, B, theta);

    return 1-0.4*exp(-POW2(log(XA*gamma)-9.21)/11.93)
            -0.05*exp(-POW2(log(XA*gamma)-5.76)/1.33)
            +0.075*exp(-POW2(log(XA*gamma)-4.03)/0.65);

}

double gB_f(double gamma, double nu, double B, double theta){

    double XA = XA_f(nu, B, theta);

    return 1-0.0045*pow(XA*gamma, 0.52);

}

int rho_nu_brute(double *res, int sz, double *nu, Source* source_t){

    double gyr_f = get_nu_B(source_t->B);
    double B = source_t->B;
    stokes pol = source_t->pol;
    double theta = source_t->incang;

    int i,j;

    int len_gamma       = (log10(source_t->gamma_max)-log10(source_t->gamma_min))*source_t->gamma_steps;
    double dgam         = (log10(source_t->gamma_max)-log10(source_t->gamma_min))/(len_gamma-1);

    double* gamma = (double*) malloc(len_gamma*sizeof(double));
    double* e_dist = (double*) malloc(len_gamma*sizeof(double));

    for (j=0; j<len_gamma; j++){
        gamma[j] = pow(10, j*dgam + log10(source_t->gamma_min));
    }

    eDist(e_dist, len_gamma, gamma, source_t);

    if (pol == STOKES_Q) {

        double* rho_nu_int;
        double rho_nu_factor = 4*M_PI*POW2(e)/(M_e*c);
        double g_min = source_t->gamma_min;
        double g_max = source_t->gamma_max;

        #pragma omp parallel private(rho_nu_int, j)
        {

        rho_nu_int = (double*) malloc(len_gamma*sizeof(double));

        #pragma omp for
        for (i=0; i<sz; i++){

            for (j=0; j<len_gamma; j++){

                rho_nu_int[j] = XA_f(nu[i], B, theta)*HX_f(gamma[j], nu[i], B, theta)*
                                e_dist[j]/(4*M_PI*gamma[j]*p0_f(gamma[j]));

            }

            res[i] = trapz(gamma, rho_nu_int, len_gamma);

            res[i] += (-HB_f(g_max, nu[i], B, theta)*
                        eDist_s(g_max, source_t)/(4*M_PI*g_max*p0_f(g_max))
                        +HB_f(g_min, nu[i], B, theta)* 
                        eDist_s(g_min, source_t)/(4*M_PI*g_min*p0_f(g_min)))*
                        source_t->ne/source_t->n_func((void*) source_t->params);

            res[i] = res[i]*rho_nu_factor/nu[i];

        }

        free(rho_nu_int);

        }

    } else if (pol == STOKES_V) {

        double* rho_nu_int;
        double rho_nu_factor = 4*M_PI*POW2(e)*gyr_f/(M_e*c);
        double g_min = source_t->gamma_min;
        double g_max = source_t->gamma_max;

        #pragma omp parallel private(rho_nu_int, j)
        {

        rho_nu_int = (double*) malloc(len_gamma*sizeof(double));        

        #pragma omp for      
        for (i=0; i<sz; i++){

            for (j=0; j<len_gamma; j++){

                rho_nu_int[j] = log((gamma[j]+p0_f(gamma[j]))/(gamma[j]-p0_f(gamma[j])))* 
                                gX_f(gamma[j], nu[i], B, theta)*
                                e_dist[j]/(4*M_PI*gamma[j]*p0_f(gamma[j]));

            }

            res[i] = trapz(gamma, rho_nu_int, len_gamma);

            
            res[i] += (-(g_max*
                        log((g_max+p0_f(g_max))/(g_max-p0_f(g_max)))-2*p0_f(g_max))*
                        gB_f(g_max, nu[i], B, theta)*
                        eDist_s(g_max, source_t)/(4*M_PI*g_max*p0_f(g_max))
                        +(g_min*
                        log((g_min+p0_f(g_min))/(g_min-p0_f(g_min)))-2*p0_f(g_min))*
                        gB_f(g_min, nu[i], B, theta)*
                        eDist_s(g_min, source_t)/(4*M_PI*g_min*p0_f(g_min)))*
                        source_t->ne/source_t->n_func((void*) source_t->params);

            res[i] = res[i]*rho_nu_factor/POW2(nu[i])*cos(theta);

        }

        free(rho_nu_int);

        }

    }

    free(e_dist);
    free(gamma);

    return 0;

}

int rho_nu_fit_huang11(double *res, int sz, double *nu, Source* source_t){

    double gyr_f = get_nu_B(source_t->B);

    double theta = source_t->incang;
    stokes pol = source_t->pol;
    double theta_k = source_t->params[0];
    double Xe; 
    int i;

    if (pol == STOKES_Q) {

        for (i=0; i<sz; i++){

            Xe = theta_k*sqrt(M_SQRT2*sin(theta)*(1e3*gyr_f/nu[i]));
            
            res[i] = POW2(e)*POW2(gyr_f)/(M_e*c*POW2(nu[i])*nu[i]);
            res[i] *= (BESSELK(1, 1./theta_k)/BESSELK(2, 1./theta_k)+6*theta_k);
            res[i] *= POW2(sin(theta));
            res[i] *= (2.011*exp(-pow(Xe,1.035)/4.7)
                    -cos(Xe/2.)*exp(-pow(Xe, 1.2)/2.73)-0.011*exp(-Xe/47.2));

        }

    } else if (pol == STOKES_V) {

        for (i=0; i<sz; i++){

            Xe = theta_k*sqrt(M_SQRT2*sin(theta)*(1e3*gyr_f/nu[i]));
            
            res[i] = 2*POW2(e)*gyr_f/(M_e*c*POW2(nu[i]));
            res[i] *= (BESSELK(0, 1./theta_k)/BESSELK(2, 1./theta_k));
            res[i] *= cos(theta);
            res[i] *= (1-0.11*log(1+0.035*Xe));

        }

    }

    return 0;

}

