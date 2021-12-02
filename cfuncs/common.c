// Licensed under a 3-clause BSD style license - see LICENSE

#include "gsl/gsl_math.h"
#include "gsl/gsl_const_cgsm.h"

const double M_e     = GSL_CONST_CGSM_MASS_ELECTRON;
const double c       = GSL_CONST_CGSM_SPEED_OF_LIGHT;
const double e       = 4.8e-10;//GSL_CONST_CGSM_ELECTRON_CHARGE;
const double sigma_T = GSL_CONST_CGSM_THOMSON_CROSS_SECTION;
const double h       = GSL_CONST_CGSM_PLANCKS_CONSTANT_H;

double trapz(double *x, double* y, int sz){

    double sum=0;
	int i;

    for (i=0; i<sz-1; i++){
        sum = sum + (x[i+1]-x[i])*(y[i+1]+y[i])*0.5;
    }

    return sum;

}

//from gsl/deriv.c, we don't care about errors atm
double central_deriv(gsl_function *F, double x, double h){

    double fm1 = GSL_FN_EVAL (F, x - h);
    double fp1 = GSL_FN_EVAL (F, x + h);

    double fmh = GSL_FN_EVAL (F, x - h / 2);
    double fph = GSL_FN_EVAL (F, x + h / 2);

    double r3 = 0.5 * (fp1 - fm1);
    double r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;

    return r5;
}

int vec_deriv_num(double* res, double* x, double* y, int sz){

    res[0] = (y[1]-y[0])/(x[1]-x[0]);
    res[sz-1] = (y[sz-1]-y[sz-2])/(x[sz-1]-x[sz-2]);
    
	int i;

    for (i=1; i<sz-1; i++)
        res[i] = (y[i+1]-y[i])/(x[i+1]-x[i])*0.5+(y[i]-y[i-1])/(x[i]-x[i-1])*0.5;

    return 0;

}
