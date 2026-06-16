#define cimg_display 0
#include "CImg-3.7.7_pre06112618/CImg.h" /*https://cimg.eu*/
#include <iostream>
#include <vector>
#include <string>

using namespace cimg_library;

__global__ void kernel_calcular_promedio(float *d_dataset, float *d_mean, int m, int n) {
    // Cada hilo se asigna a un componente/píxel específico 'j' de la imagen aplanada
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guardilla para evitar accesos fuera de rango
    if (j < n) {
        float suma = 0.0f;
        
        // Reducción secuencial en el eje vertical (recorre las 'm' imágenes)
        for (int k = 0; k < m; k++) {
            // Mapeo bidimensional aplanado a memoria lineal (long long evita desbordamiento)
            suma += d_dataset[(long long)k * n + j];
        }
        
        // Guarda el promedio final calculado para este componente en la VRAM
        d_mean[j] = suma / (float)m;
    }
}

int main() {
    int num_images = 100; // Ejemplo de volumen masivo
    int width, height, channels;

    // Asignar memoria paginada bloqueada (Pinned Memory) para Streams
    float *h_dataset;
    // cudaMallocHost((void**)&h_dataset, num_images * n * sizeof(float));

    // Bucle para cargar y aplanar las imagenes
    for (int k = 0; k < num_images; k++) {
        std::string filename = "DIV2K_valid_LR_bicubic_X4/0" + std::to_string(801 + k) + "x4.png";
        CImg<unsigned char> img(filename.c_str());

        if (k == 0) {
            width = img.width();
            height = img.height();
            channels = img.spectrum();
            
            // Asignación de memoria paginada bloqueada (Pinned Memory) para el dataset
            int n_local = width * height * channels;
            cudaMallocHost((void**)&h_dataset, (long long)num_images * n_local * sizeof(float));
        }

        // AQUI VA SU CODIGO:
        // 1. Convertir 'img' a escala de grises si es necesario.
        // 2. Poblar el arreglo lineal h_dataset con los valores de la imagen.
        
        int n_local = width * height * channels;
        long long offset_imagen = (long long)k * n_local;
        
        // Aplanamiento de la imagen actual
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    h_dataset[offset_imagen + idx] = (float)img(x, y, 0, c);
                    idx++;
                }
            }
        }
    }

    // AQUI VA SU CODIGO CUDA:
    // (Cálculo del promedio, centrado y Matriz de Covarianza)
    
    // inicio
    int m = num_images;
    int n = width * height * channels;

    // Declaración de variables para almacenar tiempos del experimento
    cudaEvent_t start_copia, stop_copia, start_kernel, stop_kernel;
    float tiempo_copia, tiempo_kernel;
    cudaEventCreate(&start_copia);
    cudaEventCreate(&stop_copia);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // Punteros del dispositivo (GPU)
    float *d_dataset;
    float *d_mean;

    // Asignación de memoria síncrona en el dispositivo
    cudaMalloc((void**)&d_dataset, (long long)m * n * sizeof(float));
    cudaMalloc((void**)&d_mean, n * sizeof(float));

    // MÁSTIL DE MEDICIÓN: Copia síncrona Host to Device (Stream 0)
    cudaEventRecord(start_copia, 0);
    cudaMemcpy(d_dataset, h_dataset, (long long)m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_copia, 0);
    cudaEventSynchronize(stop_copia);

    // Configuración de la topología de la grilla de hilos
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // MÁSTIL DE MEDICIÓN: Ejecución del Kernel en la GPU
    cudaEventRecord(start_kernel, 0);
    kernel_calcular_promedio<<<blocksPerGrid, threadsPerBlock>>>(d_dataset, d_mean, m, n);
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);

    // Cálculo y despliegue de métricas temporales para el informe
    cudaEventElapsedTime(&tiempo_copia, start_copia, stop_copia);
    cudaEventElapsedTime(&tiempo_kernel, start_kernel, stop_kernel);

    std::cout << ">>> [MÉTRICAS VECTOR PROMEDIO - GPU] <<<" << std::endl;
    std::cout << "Tiempo de copia hacia la GPU (H2D): " << tiempo_copia << " ms" << std::endl;
    std::cout << "Tiempo neto de cómputo (Kernel):   " << tiempo_kernel << " ms" << std::endl;

    // Liberación de recursos locales de eventos
    cudaEventDestroy(start_copia);
    cudaEventDestroy(stop_copia);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    // resto de codigos

    cudaFree(d_dataset);
    cudaFree(d_mean);
    cudaFreeHost(h_dataset);
    
    return 0;
}