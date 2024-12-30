#ifndef INCLUDE_H
#define INCLUDE_H

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

double ortalama(const std::vector<std::vector<double>>& veriSeti); 

double standartSapma(const std::vector<std::vector<double>>& veriSeti);

double sigmoid(double x);

double sigmoid_derivative(double x);

double predict(const std::vector<std::vector<double>>& weights1, const std::vector<double>& weights2, const std::vector<double>& input_data,
               const double& ortalama_Deger, const double& standart_sapma_deger );

double roundToDecimal(double value, int decimalPlaces);

#endif


