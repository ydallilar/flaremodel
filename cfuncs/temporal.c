
// Licensed under a 3-clause BSD style license - see LICENSE

#include "math.h"
#include "stdlib.h"
#include "edist.h"
#include "common.h"

int vec_sync_dgamdt(double *res, int sz, double *gamma, Source* source_t){

    double B = source_t->B;
    double u_B = B*B/(8*M_PI); // 1/(4*M_PI) if B is not isotropic
    double factor = 4./3.*sigma_T/M_e/c*u_B; // 2/3. angle averaged sin^2 used.

    for (int i=0; i<sz; i++)
        res[i] = -gamma[i]*gamma[i]*factor;

    return 0;

}

// exp_b : expansion speed in units of c
int vec_adiab_dgamdt(double *res, int sz, double *gamma, double exp_b, Source* source_t){

    double R = source_t->R;

    for (int i=0; i<sz; i++)
        res[i] = -gamma[i]*exp_b*c/R;

    return 0;

}

int cool_onestep(double *e_dist, int sz, double* gamma, double* e_dist_inj,
                double dt, double exp_b, double t_esc, double inj_r, Source* source_t){

    double cool_r, esc_r;

    double* sync_dgamdt = (double*) malloc(sz*sizeof(double));
    double* adiab_dgamdt = (double*) malloc(sz*sizeof(double));

    vec_sync_dgamdt(sync_dgamdt, sz, gamma, source_t);
    vec_adiab_dgamdt(adiab_dgamdt, sz, gamma, exp_b, source_t);

    for (int i=0; i<sz; i++){

        if (i == sz-1) {
            e_dist[i] = 0; // The code leads to instability propagating from the last bin if not set to 0.
        } else { 
            esc_r = -e_dist[i]/t_esc;
            cool_r = -((sync_dgamdt[i+1]+adiab_dgamdt[i+1])*e_dist[i+1]-(sync_dgamdt[i]+adiab_dgamdt[i])*e_dist[i])/(gamma[i+1]-gamma[i]);
            e_dist[i] = e_dist[i] + (cool_r + inj_r*e_dist_inj[i] + esc_r)*dt;
        }

    }

    source_t->B = source_t->B*POW2(source_t->R/(source_t->R+exp_b*c*dt)); // Assume B \propto R^-2 for the moment for flux conservation.
    source_t->R = source_t->R + exp_b*c*dt;

    free(sync_dgamdt);
    free(adiab_dgamdt);

    return 0;

}
