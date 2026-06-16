#ifndef KERNELS_H
#define KERNELS_H

// Declaración de la función wrapper que llamará al kernel
void lanzar_promedio(float *h_dataset, float *h_mean_out, int m, int n, float &tiempo_copia, float &tiempo_kernel);

#endif