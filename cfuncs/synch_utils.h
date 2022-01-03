// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef SYNC_UTILS_H
#define SYNC_UTILS_H

#include "common.h"

int synchF(double *res, int sz, double *x);
int synchG(double *res, int sz, double *x);
int synchH(double *res, int sz, double *x);
int b_synchF(double *res, int sz, double *x);
int b_synchH(double *res, int sz, double *x);

int synch_fun(stokes pol, double *res, int sz, double *x);
int b_synch_fun(stokes pol, double *res, int sz, double *x);

double get_nu_B(double B);

#endif
