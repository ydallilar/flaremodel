// Licensed under a 3-clause BSD style license - see LICENSE

#include "math.h"
#include "stdlib.h"
#include "common.h"
#include "edist.h"
#include "stdio.h"
#include "gsl/gsl_integration.h"

/*****************************************************************************/
// Calculate synchrotron F(x) function with interpolation

const double sF_x[50] = {7.943e-05,  1.014e-04,  1.295e-04,  1.653e-04,  2.111e-04,
        2.695e-04,  3.441e-04,  4.394e-04,  5.610e-04,  7.163e-04,
        9.146e-04,  1.168e-03,  1.491e-03,  1.904e-03,  2.431e-03,
        3.103e-03,  3.962e-03,  5.059e-03,  6.460e-03,  8.248e-03,
        1.053e-02,  1.345e-02,  1.717e-02,  2.192e-02,  2.799e-02,
        3.573e-02,  4.562e-02,  5.825e-02,  7.438e-02,  9.496e-02,
        1.212e-01,  1.548e-01,  1.977e-01,  2.524e-01,  3.222e-01,
        4.114e-01,  5.253e-01,  6.707e-01,  8.564e-01,  1.093e+00,
        1.396e+00,  1.782e+00,  2.276e+00,  2.906e+00,  3.710e+00,
        4.737e+00,  6.048e+00,  7.722e+00,  9.860e+00,  1.259e+01};

const double sF_val[50] = {9.226e-02,  1.001e-01,  1.085e-01,  1.177e-01,  1.276e-01,
        1.384e-01,  1.500e-01,  1.626e-01,  1.763e-01,  1.910e-01,
        2.070e-01,  2.242e-01,  2.429e-01,  2.630e-01,  2.846e-01,
        3.079e-01,  3.330e-01,  3.598e-01,  3.886e-01,  4.193e-01,
        4.521e-01,  4.868e-01,  5.234e-01,  5.619e-01,  6.019e-01,
        6.434e-01,  6.856e-01,  7.281e-01,  7.700e-01,  8.101e-01,
        8.469e-01,  8.784e-01,  9.025e-01,  9.161e-01,  9.161e-01,
        8.990e-01,  8.614e-01,  8.009e-01,  7.166e-01,  6.104e-01,
        4.882e-01,  3.602e-01,  2.397e-01,  1.396e-01,  6.855e-02,
        2.704e-02,  8.056e-03,  1.675e-03,  2.198e-04,  1.600e-05};

double synchF_s(double x, gsl_interp_accel* sFacc, gsl_spline* sFsp)
{

    if (x < 1e-4){
        return 4.*M_PI / sqrt(3) / GAMMAF(1/3.) * pow(x*0.5, 1/3.);
    } else if (x < 10.){
        return gsl_spline_eval (sFsp, x, sFacc);
    } else {
        return sqrt(M_PI*0.5) * exp(-x) * sqrt(x);
    }
}

int synchF(double *res, int sz, double *x)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, 50);
    gsl_spline_init(spline, &sF_x[0], &sF_val[0], 50);
	int i;

    for (i=0; i<sz; i++)
        res[i] = synchF_s(x[i], acc, spline);

    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);

    return 0;
}

/*****************************************************************************/
// Brute F(x) calc

double b_synchF_integrand(double x, void* params){

    gsl_sf_result eval;
    gsl_error_handler_t *old_error_handler=gsl_set_error_handler_off (); // turn off the error handler
    int code = gsl_sf_bessel_Knu_e(5./3, x, &eval); // compute the density, see gsl/gsl_errno.h for the error code
    gsl_set_error_handler(old_error_handler); //reset the error handler 
    return (code==GSL_SUCCESS ? eval.val : 0.0);
}

double b_synchF_s(double x, gsl_integration_workspace* w_integ){
    double epsabs=0;
    double epsrel=1e-3;
    double result;
    double err;

    gsl_function F;
    F.function = &b_synchF_integrand;
    F.params = NULL;

    if (x < 1e-4){
        return 4.*M_PI / sqrt(3) / GAMMAF(1/3.) * pow(x*0.5, 1/3.);
    } else if (x > 10.){
        return sqrt(M_PI*0.5) * exp(-x) * sqrt(x);
    } else {
        gsl_integration_qagiu(&F, x, epsabs, epsrel, 1000, w_integ, &result, &err);
        return result;
    }
}

int b_synchF(double *res, int sz, double *x){

    gsl_integration_workspace* w_integ = gsl_integration_workspace_alloc(1000);
	int i;

    for (i=0; i<sz; i++)
        res[i] = x[i]*b_synchF_s(x[i], w_integ);
    
    gsl_integration_workspace_free(w_integ);

    return 0;
}


/*****************************************************************************/

double inline get_nu_B(double B)
{
    return e*B/(2*M_PI*M_e*c);
} 

/*****************************************************************************/
// {j,a}_nu_brute

int j_nu_brute(double *res, int sz, double *nu, Source* source_t){

    double B            = source_t->B;
    //If incang == -1, use angle averaged factor instead of sin_theta
    double sin_theta    = (source_t->incang == -1) ? M_PI/4. : sin(source_t->incang);
    double P_el_factor  = sqrt(3.0)*e*e*e*B/M_e/c/c*sin_theta;
    double P_dist_factor = 1./4./M_PI;

    int len_gamma       = (log10(source_t->gamma_max)-log10(source_t->gamma_min))*source_t->gamma_steps;
    double dgam         = (log10(source_t->gamma_max)-log10(source_t->gamma_min))/(len_gamma-1);

    double* gamma = (double*) malloc(len_gamma*sizeof(double));
    double* e_dist = (double*) malloc(len_gamma*sizeof(double));

    double* j_nu_int;
    double* nu_over_nu_crit;
    double* sFs;
	int i,j;

    for (j=0; j<len_gamma; j++){
        gamma[j] = pow(10, j*dgam + log10(source_t->gamma_min));
    }

    eDist(e_dist, len_gamma, gamma, source_t);

    #pragma omp parallel private(nu_over_nu_crit, sFs, j_nu_int)
    {
    
    j_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synchF(sFs, len_gamma, nu_over_nu_crit);
        for(j=0; j< len_gamma; j++){
            j_nu_int[j] = P_dist_factor*P_el_factor*sFs[j]*e_dist[j];
        }
        res[i] = trapz(gamma, j_nu_int, len_gamma);
    }

    free(j_nu_int);
    free(nu_over_nu_crit);
    free(sFs);

    }

    free(gamma);
    free(e_dist);
    return 0;
}

int a_nu_brute(double *res, int sz, double *nu, Source* source_t){

    double B            = source_t->B;
    //If incang == -1, use angle averaged factor instead of sin_theta
    double sin_theta    = (source_t->incang == -1) ? M_PI/4. : sin(source_t->incang);
    double P_el_factor  = sqrt(3.0)*e*e*e*B/M_e/c/c*sin_theta;
    double a_dist_factor= - 1./8./M_PI/M_e;

    int len_gamma       = (log10(source_t->gamma_max)-log10(source_t->gamma_min))*source_t->gamma_steps;
    double dgam         = (log10(source_t->gamma_max)-log10(source_t->gamma_min))/(len_gamma-1);

    double* gamma = (double*) malloc(len_gamma*sizeof(double));
    double* e_dist = (double*) malloc(len_gamma*sizeof(double));
    double* de_distdgam = (double*) malloc(len_gamma*sizeof(double));

    double* a_nu_int;
    double* nu_over_nu_crit;
    double* sFs;
	int i,j;

    for (j=0; j<len_gamma; j++){
        gamma[j] = pow(10, j*dgam + log10(source_t->gamma_min));
    }

    eDist(e_dist, len_gamma, gamma, source_t);
    deDistdgam(de_distdgam, len_gamma, gamma, source_t);

    #pragma omp parallel private(nu_over_nu_crit, sFs, a_nu_int)
    {
    
    a_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synchF(sFs, len_gamma, nu_over_nu_crit);
        for(j=0; j< len_gamma; j++){
            a_nu_int[j] =  (a_dist_factor/POW2(nu[i]))*(P_el_factor*sFs[j])*(de_distdgam[j]-2*e_dist[j]/gamma[j]);
        }
        res[i] = trapz(gamma, a_nu_int, len_gamma);
    }

    free(a_nu_int);
    free(nu_over_nu_crit);
    free(sFs);
    }

    free(gamma);
    free(e_dist);
    free(de_distdgam);

    return 0;
}

/*****************************************************************************/

int j_nu_userdist(double *res, int sz, double *nu, int len_gamma, double *gamma, double* e_dist, Source* source_t){

    double B            = source_t->B;
    //If incang == -1, use angle averaged factor instead of sin_theta
    double sin_theta    = (source_t->incang == -1) ? M_PI/4. : sin(source_t->incang);
    double P_el_factor  = sqrt(3.0)*e*e*e*B/M_e/c/c*sin_theta;
    double P_dist_factor = 1./4./M_PI;

    double* j_nu_int;
    double* nu_over_nu_crit;
    double* sFs;
	int i,j;

    #pragma omp parallel private(j_nu_int, nu_over_nu_crit, sFs)
    {
    
    j_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synchF(sFs, len_gamma, nu_over_nu_crit);
        for(j=0; j< len_gamma; j++){
            j_nu_int[j] = P_dist_factor*P_el_factor*sFs[j]*e_dist[j];
        }
        res[i] = trapz(gamma, j_nu_int, len_gamma);
    }

    free(j_nu_int);
    free(nu_over_nu_crit);
    free(sFs);
    }

    return 0;
}

int a_nu_userdist(double *res, int sz, double *nu, int len_gamma, double *gamma, double *e_dist, Source* source_t){

    double B            = source_t->B;
    //If incang == -1, use angle averaged factor instead of sin_theta
    double sin_theta    = (source_t->incang == -1) ? M_PI/4. : sin(source_t->incang);
    double P_el_factor  = sqrt(3.0)*e*e*e*B/M_e/c/c*sin_theta;
    double a_dist_factor= - 1./8./M_PI/M_e;

    double* de_distdgam = (double*) malloc(len_gamma*sizeof(double)); 

    double* a_nu_int;
    double* nu_over_nu_crit;
    double* sFs;
	int i,j;

    vec_deriv_num(de_distdgam, gamma, e_dist, len_gamma);

    #pragma omp parallel private(nu_over_nu_crit, sFs, a_nu_int)
    {
    
    a_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synchF(sFs, len_gamma, nu_over_nu_crit);
        for(j=0; j< len_gamma; j++){
            a_nu_int[j] =  (a_dist_factor/POW2(nu[i]))*(P_el_factor*sFs[j])*(de_distdgam[j]-2*e_dist[j]/gamma[j]);
        }
        res[i] = trapz(gamma, a_nu_int, len_gamma);
    }

    free(a_nu_int);
    free(nu_over_nu_crit);
    free(sFs);
    }

    free(de_distdgam);

    return 0;
}

/*****************************************************************************/
