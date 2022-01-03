// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef SYNCHROTRON_H
#define SYNCHROTRON_H

#include "common.h"

int j_nu_brute(double *res, int sz, double *nu, Source* source_t);
int a_nu_brute(double *res, int sz, double *nu, Source* source_t);

int j_nu_userdist(double *res, int sz, double *nu, int len_gamma, double *gamma, double *e_dist, Source* source_t);
int a_nu_userdist(double *res, int sz, double *nu, int len_gamma, double *gamma, double *e_dist, Source* source_t);

#endif