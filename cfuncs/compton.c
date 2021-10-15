// Licensed under a 3-clause BSD style license - see LICENSE

#include "common.h"
#include "stdlib.h"

double KN(double epsilon, double gamma, double nu){

    double epsilon1 = nu*h;
    double gamma_e = 4*epsilon*gamma/(M_e*c*c); 
    double q = epsilon1/(gamma_e*(gamma*M_e*c*c - epsilon1)); 
    double res = 3*sigma_T*c/(4*gamma*gamma*epsilon) *(
                            2*q*log(q) + 
                            (1+2*q)*(1-q) + 
                            (gamma_e*q)*(gamma_e*q)*(1-q)/
                            (2*(1 + gamma_e*q))
                            );

    if (isnan(res)) return 0;
    if (q > 1) return 0;
    if (res < 0) return 0;

    return res;

}

double compton_emissivity_s(double *epsilon, double *gamma,
                        double nu, double *n_ph, double *e_dist,
                        int eps_sz, int gamma_sz){
    
    double* tmp = (double*) malloc(gamma_sz*sizeof(double));

    double sum;
    double last, next;
    double res;

    #pragma omp parallel for private(last,next,sum)
    for (int i=0; i<gamma_sz; i++){
        sum = 0;
        last = n_ph[0]*e_dist[i]*KN(epsilon[0], gamma[i], nu);        
        for (int j=0; j<eps_sz-1; j++){
            next = n_ph[j+1]*e_dist[i]*KN(epsilon[j+1], gamma[i], nu);
            sum += (last+next)*(epsilon[j+1]-epsilon[j])*0.5;
            last = next;            
        }
        tmp[i] = sum;
    }

    res = trapz(gamma, tmp, gamma_sz);

    free(tmp);

    return POW2(h)*nu*res/(4*M_PI);

}

void compton_emissivity(double *res, double *epsilon, double *gamma,
                        double *nu, double *n_ph, double *e_dist,
                        int eps_sz, int gamma_sz, int nu_sz){

    for(int i=0; i<nu_sz; i++)
        res[i] = compton_emissivity_s(epsilon, gamma, nu[i], 
                                    n_ph, e_dist, eps_sz, gamma_sz);

}
