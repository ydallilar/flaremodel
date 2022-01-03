// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef FARADAY_H
#define FARADAY_H

#include "common.h"

double p0_f(double gamma);
double XA_f(double nu, double B, double theta);
double HX_f(double gamma, double nu, double B, double theta);
double HB_f(double gamma, double nu, double B, double theta);
double gX_f(double gamma, double nu, double B, double theta);
double gX_B(double gamma, double nu, double B, double theta);

int rho_nu_brute(double *res, int sz, double *nu, Source* source_t);
int rho_nu_fit_huang11(double *res, int sz, double *nu, Source* source_t);

#endif
