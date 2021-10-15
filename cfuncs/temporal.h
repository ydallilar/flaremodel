// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef TEMPORAL_H
#define TEMPORAL_H

#include "common.h"

int cool_onestep(double *e_dist, int sz, double* gamma, double* e_dist_inj,
                double dt, double exp_b, double t_esc, double inj_r, Source* source_t);

#endif
