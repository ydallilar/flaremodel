// Licensed under a 3-clause BSD style license - see LICENSE

#ifndef COMPTON_H
#define COMPTON_H

void compton_emissivity(double *res, double *epsilon, double *gamma,
                        double *nu, double *n_ph, double *e_dist,
                        int eps_sz, int gamma_sz, int nu_sz);

#endif
