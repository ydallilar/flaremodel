// Licensed under a 3-clause BSD style license - see LICENSE

#include "math.h"
#include "stdlib.h"
#include "common.h"
#include "edist.h"
#include "stdio.h"
#include "gsl/gsl_integration.h"

/*****************************************************************************/
// Calculate synchrotron F(x) function with interpolation

#define SYNCHF_LO 1e-4
#define SYNCHF_HI 1e2
#define SYNCHG_LO 1e-20
#define SYNCHG_HI 5e2
#define SYNCHH_LO 1e-4
#define SYNCHH_HI 5e2

const double sF_x[50] = {-4.3   , -4.1653, -4.0306, -3.8959, -3.7612, -3.6265, -3.4918,
       -3.3571, -3.2224, -3.0878, -2.9531, -2.8184, -2.6837, -2.549 ,
       -2.4143, -2.2796, -2.1449, -2.0102, -1.8755, -1.7408, -1.6061,
       -1.4714, -1.3367, -1.202 , -1.0673, -0.9327, -0.798 , -0.6633,
       -0.5286, -0.3939, -0.2592, -0.1245,  0.0102,  0.1449,  0.2796,
        0.4143,  0.549 ,  0.6837,  0.8184,  0.9531,  1.0878,  1.2224,
        1.3571,  1.4918,  1.6265,  1.7612,  1.8959,  2.0306,  2.1653,
        2.3   };

const double sF_val[50] = {-1.1015e+00, -1.0567e+00, -1.0119e+00, -9.6722e-01, -9.2254e-01,
       -8.7790e-01, -8.3333e-01, -7.8883e-01, -7.4442e-01, -7.0012e-01,
       -6.5597e-01, -6.1198e-01, -5.6821e-01, -5.2470e-01, -4.8152e-01,
       -4.3874e-01, -3.9645e-01, -3.5479e-01, -3.1391e-01, -2.7399e-01,
       -2.3529e-01, -1.9812e-01, -1.6288e-01, -1.3011e-01, -1.0050e-01,
       -7.4954e-02, -5.4703e-02, -4.1385e-02, -3.7228e-02, -4.5282e-02,
       -6.9738e-02, -1.1639e-01, -1.9324e-01, -3.1140e-01, -4.8617e-01,
       -7.3869e-01, -1.0980e+00, -1.6037e+00, -2.3103e+00, -3.2919e+00,
       -4.6495e+00, -6.5208e+00, -9.0936e+00, -1.2624e+01, -1.7460e+01,
       -2.4077e+01, -3.3124e+01, -4.5485e+01, -6.2363e+01, -8.5403e+01};

const double sH_x[50] = {-4.3   , -4.1551, -4.0102, -3.8653, -3.7204, -3.5755, -3.4306,
       -3.2857, -3.1408, -2.9959, -2.851 , -2.7061, -2.5612, -2.4163,
       -2.2714, -2.1265, -1.9816, -1.8367, -1.6918, -1.5469, -1.402 ,
       -1.2571, -1.1122, -0.9673, -0.8224, -0.6776, -0.5327, -0.3878,
       -0.2429, -0.098 ,  0.0469,  0.1918,  0.3367,  0.4816,  0.6265,
        0.7714,  0.9163,  1.0612,  1.2061,  1.351 ,  1.4959,  1.6408,
        1.7857,  1.9306,  2.0755,  2.2204,  2.3653,  2.5102,  2.6551,
        2.8   };

const double sH_val[50] = {2.5832e-01,  2.5822e-01,  2.5816e-01,  2.5805e-01,  2.5792e-01,
        2.5775e-01,  2.5754e-01,  2.5728e-01,  2.5695e-01,  2.5654e-01,
        2.5603e-01,  2.5538e-01,  2.5457e-01,  2.5355e-01,  2.5227e-01,
        2.5065e-01,  2.4861e-01,  2.4602e-01,  2.4272e-01,  2.3850e-01,
        2.3308e-01,  2.2607e-01,  2.1693e-01,  2.0491e-01,  1.8891e-01,
        1.6739e-01,  1.3809e-01,  9.7669e-02,  4.1269e-02, -3.8244e-02,
       -1.5124e-01, -3.1266e-01, -5.4383e-01, -8.7492e-01, -1.3483e+00,
       -2.0231e+00, -2.9818e+00, -4.3395e+00, -6.2560e+00, -8.9547e+00,
       -1.2747e+01, -1.8066e+01, -2.5518e+01, -3.5949e+01, -5.0537e+01,
       -7.0932e+01, -9.9432e+01, -1.3925e+02, -1.9486e+02, -2.7252e+02};

double synchF_s(double x, gsl_interp_accel* sFacc, gsl_spline* sFsp)
{

    if (x < SYNCHF_LO){
        return 4.*M_PI / sqrt(3) / GAMMAF(1/3.) * pow(x*0.5, 1/3.);
    } else if (x < SYNCHF_HI){
        return pow(10,gsl_spline_eval (sFsp, log10(x), sFacc));
    } else {
        return sqrt(M_PI*0.5) * exp(-x) * sqrt(x);
    }
}

int synchF(double *res, int sz, double *x)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, 50);
    gsl_spline_init(spline, &sF_x[0], &sF_val[0], 50);

    for (int i=0; i<sz; i++)
        res[i] = synchF_s(x[i], acc, spline);

    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);

    return 0;
}

double synchG_s(double x){

    if ((x > SYNCHG_LO) && (x < SYNCHG_HI)) {
        return x*BESSELK(2/3., x);
    } else {
        return 0;
    }

}

int synchG(double *res, int sz, double *x)
{
    for (int i=0; i<sz; i++)
        res[i] = synchG_s(x[i]);

    return 0;
}

double synchH_s(double x, gsl_interp_accel* sFacc, gsl_spline* sFsp)
{

    if (x < SYNCHH_LO){
        return 1.813;
    } else if (x < SYNCHH_HI){
        return pow(10,gsl_spline_eval (sFsp, log10(x), sFacc));
    } else {
        return 0;
    }
}

int synchH(double *res, int sz, double *x)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, 50);
    gsl_spline_init(spline, &sH_x[0], &sH_val[0], 50);

    for (int i=0; i<sz; i++)
        res[i] = synchH_s(x[i], acc, spline);

    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);

    return 0;
}

int synch_fun(stokes pol, double *res, int sz, double *x){

    int st = 0;

    switch (pol){
        case 0:
            synchF(res, sz, x);
            break;
        case 1:
            synchG(res, sz, x);
            break;
        case 2:
            synchH(res, sz, x);
            break;
        default:
            st = 1;
    }
    return st;

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

    if (x < SYNCHF_LO){
        return 4.*M_PI / sqrt(3) / GAMMAF(1/3.) * pow(x*0.5, 1/3.);
    } else if (x > SYNCHF_HI){
        return sqrt(M_PI*0.5) * exp(-x) * sqrt(x);
    } else {
        gsl_integration_qagiu(&F, x, epsabs, epsrel, 1000, w_integ, &result, &err);
        return x*result;
    }
}

int b_synchF(double *res, int sz, double *x){

    gsl_integration_workspace* w_integ = gsl_integration_workspace_alloc(1000);

    for (int i=0; i<sz; i++)
        res[i] = b_synchF_s(x[i], w_integ);
    
    gsl_integration_workspace_free(w_integ);

    return 0;
}

double b_synchH_integrand(double x, void* params){

    gsl_sf_result eval;
    gsl_error_handler_t *old_error_handler=gsl_set_error_handler_off (); // turn off the error handler
    int code = gsl_sf_bessel_Knu_e(1./3, x, &eval); // compute the density, see gsl/gsl_errno.h for the error code
    gsl_set_error_handler(old_error_handler); //reset the error handler 
    return (code==GSL_SUCCESS ? eval.val : 0.0);
}

double b_synchH_s(double x, gsl_integration_workspace* w_integ){
    double epsabs=0;
    double epsrel=1e-3;
    double result;
    double err;

    gsl_function F;
    F.function = &b_synchH_integrand;
    F.params = NULL;

    gsl_integration_qagiu(&F, x, epsabs, epsrel, 1000, w_integ, &result, &err);
    return result;
}

int b_synchH(double *res, int sz, double *x){

    gsl_integration_workspace* w_integ = gsl_integration_workspace_alloc(1000);

    for (int i=0; i<sz; i++)
        res[i] = b_synchH_s(x[i], w_integ) + x[i]*BESSELK(1./3, x[i]);
    
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
    stokes pol = source_t->pol;

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

    #pragma omp parallel private(nu_over_nu_crit, sFs, j_nu_int, j)
    {
    
    j_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synch_fun(pol, sFs, len_gamma, nu_over_nu_crit);
        if (pol == STOKES_V)
            for (int j=0; j<len_gamma; j++) 
                sFs[j] = sFs[j]*4/(3*gamma[j]*tan(source_t->incang));
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
    stokes pol = source_t->pol;

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

    #pragma omp parallel private(nu_over_nu_crit, sFs, a_nu_int, j)
    {
    
    a_nu_int = (double*) malloc(len_gamma*sizeof(double));
    sFs = (double*) malloc(len_gamma*sizeof(double));
    nu_over_nu_crit = (double*) malloc(len_gamma*sizeof(double));

    #pragma omp for
    for (i=0; i<sz; i++){
        for (j=0; j<len_gamma; j++){
            nu_over_nu_crit[j] = nu[i]/((3/2.)*get_nu_B(B)*POW2(gamma[j])*sin_theta);
        }
        synch_fun(pol, sFs, len_gamma, nu_over_nu_crit);
        if (pol == STOKES_V)
            for (int j=0; j<len_gamma; j++) 
                sFs[j] = sFs[j]*4/(3*gamma[j]*tan(source_t->incang));
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

    #pragma omp parallel private(j_nu_int, nu_over_nu_crit, sFs, j)
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

    #pragma omp parallel private(nu_over_nu_crit, sFs, a_nu_int, j)
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
