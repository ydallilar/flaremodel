// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef EDIST_H
#define EDIST_H

#include "common.h"

// params : p, gamma_min, gamma_max
double powerlaw(double gamma, void *params);
// params : p, gamma_min, gamma_max
double powerlawexpcutoff(double gamma, void *params);
// params : theta
double thermal(double gamma, void *params);
// params : kappa, kappa_width
double kappa(double gamma, void *params);
//params : p1, p2, gamma_b, gamma_min, gamma_max
double bknpowerlaw(double gamma, void *params);
//params : p1, p2, gamma_b, gamma_min, gamma_max
double bknpowerlawexpcutoff(double gamma, void *params);

double powerlaw_norm(void *params);
double thermal_norm(void *params);
double kappa_norm(void *params);
double bknpowerlaw_norm(void *params);

int eDist(double *res, int sz, double *gamma, Source* source_t);
int deDistdgam(double *res, int sz, double *gamma, Source* source_t);
double eDist_s(double gamma, Source* source_t);
double deDistdgam_s(double gamma, Source* source_t);

#endif