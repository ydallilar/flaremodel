// Licensed under a 3-clause BSD style license - see LICENSE

#include "math.h"
#include "common.h"
#include "edist.h"
#include "synch_utils.h"

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
