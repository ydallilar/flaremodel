// Licensed under a 3-clause BSD style license - see LICENSE

#include "common.h"
#include "math.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_deriv.h"
#include "stdio.h"

double P_THRESH=1e-4;

/*****************************************************************************/
// Plain electron distributions no normalization

// params : p, gamma_min, gamma_max
double powerlaw(double gamma, void *params){
    double *p = (double*) params;
    return pow(gamma, -p[0]);
}

// params : p, gamma_min, gamma_max
double powerlawexpcutoff(double gamma, void *params){
    double *p = (double*) params;
    return powerlaw(gamma, params) * exp(-gamma/p[2]);
}

// params : theta
double thermal(double gamma, void *params){
    double *p = (double*) params;
    return gamma*sqrt(gamma*gamma-1)*exp(-gamma/p[0]);
}

// params : kappa, kappa_width
double kappa(double gamma, void *params){
    double *p = (double*) params;
    return gamma*sqrt(gamma*gamma-1)*pow(1+(gamma-1)/(p[0]*p[1]),-p[0]-1);

}

//params : p1, p2, gamma_b, gamma_min, gamma_max
double bknpowerlaw(double gamma, void *params){
    double *p = (double*) params;
    double factor = pow(p[2], p[1]-p[0]);
    return (gamma < p[2]) ? pow(gamma, -p[0]) : factor*pow(gamma, -p[1]);
}

//params : p1, p2, gamma_b, gamma_min, gamma_max
double bknpowerlawexpcutoff(double gamma, void *params){
    double *p = (double*) params;
    return bknpowerlaw(gamma, params) * exp(-gamma/p[4]);
}

/*****************************************************************************/
// Electron distribution normalizations

double powerlaw_norm(void* params){
    double *p = (double*) params;
    return (P_THRESH > fabs(p[0] - 1)) ? (log(p[2])-log(p[1])) : (pow(p[1], -p[0]+1)-pow(p[2], -p[0]+1))/(p[0]-1); 
}

double bknpowerlaw_norm(void* params){
    double *p = (double*) params;
    double factor = pow(p[2], p[1]-p[0]);
    double norm1  = (P_THRESH > fabs(p[0] - 1)) ? (log(p[2])-log(p[3])) : (pow(p[3], -p[0]+1)-pow(p[2], -p[0]+1))/(p[0]-1); 
    double norm2  = (P_THRESH > fabs(p[1] - 1)) ? factor*(log(p[4])-log(p[2])) : factor*(pow(p[2], -p[1]+1)-pow(p[4], -p[1]+1))/(p[1]-1);
    return norm1+norm2;
}

double thermal_norm(void* params){
    double *p = (double*) params;
    return (p[0]*BESSELK(2, 1/p[0]));
}

// Pandya+ 2016, ApJ 822:34 / Sec. 3.3
double kappa_norm(void* params){
    double *p = (double*) params;
    double norm_low, norm_high;
    
    norm_high = (p[0]-2)*(p[0]-1)/(2* (p[0]*p[0]) * (p[1]*p[1]*p[1]));
    norm_low = pow(2/(M_PI* (p[0]*p[0]*p[0]) * (p[1]*p[1]*p[1])), 0.5);
    norm_low = norm_low * GAMMAF(p[0] + 1);
    norm_low = norm_low / GAMMAF(p[0] - 0.5);

    return pow(pow(norm_low, -0.7) + pow(norm_high, -0.7), 1/0.7);
}

/*****************************************************************************/
// Commmon interface for electron distributions - no normalization

double eDist_s(double gamma, Source* source_t){

    return source_t->d_func(gamma, (void*) source_t->params);

}

double deDistdgam_s(double gamma, Source* source_t){

    gsl_function F;

    F.function = source_t->d_func;
    F.params = (void*) source_t->params;

    double err;
    double res;
    double h = 1e-8;

    gsl_deriv_central(&F, gamma, h, &res, &err);

    return res;
}

/*****************************************************************************/
// Electron distributions from gamma array. Full math.

int deDistdgam(double *res, int sz, double *gamma, Source* source_t){

    double norm = source_t->n_func((void*) source_t->params);
	int i=0;

    #pragma omp parallel for
    for (i=0; i<sz; i++){
        res[i] = source_t->ne*deDistdgam_s(gamma[i], source_t)/norm;
    }
    return 0;
}


int eDist(double *res, int sz, double *gamma, Source* source_t){

    double norm = source_t->n_func((void*) source_t->params);
	int i=0;

    #pragma omp parallel for
    for (i=0; i<sz; i++){
        res[i] = source_t->ne*eDist_s(gamma[i], source_t)/norm;
    }
    return 0;
}

/*****************************************************************************/
