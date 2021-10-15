// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef COMMON_H
#define COMMON_H

#include "gsl/gsl_math.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_bessel.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_sf_result.h"

extern const double M_e;
extern const double c;
extern const double e;
extern const double sigma_T;
extern const double h;

double trapz(double *x, double* y, int sz);
double central_deriv(gsl_function *F, double x, double h);
int vec_deriv_num(double* res, double* x, double* y, int sz);

#define GAMMAF(x) gsl_sf_gamma(x)
#define BESSELK(nu, x) gsl_sf_bessel_Knu(nu, x)
#define POW2(x) gsl_pow_2(x)

struct Source {
    double B;
    double ne;
    double R;
    int d_type;
    double (*d_func) (double, void*);
    double (*n_func) (void*);
    double* params;
    double incang;
    double gamma_min;
    double gamma_max;
    double gamma_steps;
};

typedef struct Source Source;

struct Int_Params {
    Source* source_t;
    double nu;
    gsl_spline* sFsp;
    gsl_interp_accel* sFacc;
};

typedef struct Int_Params Int_Params;


#endif
