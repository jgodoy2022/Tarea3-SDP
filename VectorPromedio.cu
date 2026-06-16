#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"
#include <iostream>
#include <vector>

using namespace cimg_library;

int main() {
    int num_images = 10000; // Ejemplo de volumen masivo
    int width, height, channels;

    // Asignar memoria paginada bloqueada (Pinned Memory) para Streams
    float *h_dataset;
    // cudaMallocHost((void**)&h_dataset, num_images * n * sizeof(float));

    // Bucle para cargar y aplanar las imagenes
    for (int k = 0; k < num_images; k++) {
        std::string filename = "dataset/img_" + std::to_string(k) + ".png";
        CImg<unsigned char> img(filename.c_str());

        if (k == 0) {
            width = img.width();
            height = img.height();
            channels = img.spectrum();
        }

        // AQUI VA SU CODIGO:
        // 1. Convertir 'img' a escala de grises si es necesario.
        // 2. Poblar el arreglo lineal h_dataset con los valores de la imagen.
    }

    // AQUI VA SU CODIGO CUDA:
    // (Cálculo del promedio, centrado y Matriz de Covarianza)

    // cudaFreeHost(h_dataset);
    return 0;
}