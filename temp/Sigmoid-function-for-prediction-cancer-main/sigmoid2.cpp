#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../include/veriseti.h"
// Sigmoid fonksiyonu
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid fonksiyonunun türevi
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

int main() {
  std::vector<std::vector<double>> input;
  std::vector<std::vector<double>> sonuc;
    for (size_t j = 0; j <  inputs.size(); ++j) {
     if (j % 2 == 0) {
      input.push_back(inputs[j]);
     }
     else {
      sonuc.push_back(inputs[j]);
    }
    }
    // Ağırlıkların başlangıç değerleri (rastgele)
    std::srand(std::time(0));
    double weight1_1 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_2 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_3 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_4 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_5 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_6 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_7 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_8 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight1_9 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_1 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_2 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_3 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_4 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_5 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_6 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_7 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_8 = ((double) std::rand() / RAND_MAX) - 0.5;
    double weight2_9 = ((double) std::rand() / RAND_MAX) - 0.5;
    double bias1 = ((double) std::rand() / RAND_MAX) - 0.5;
    double bias2 = ((double) std::rand() / RAND_MAX) - 0.5;
    double output_weight1 = ((double) std::rand() / RAND_MAX) - 0.5;
    double output_weight2 = ((double) std::rand() / RAND_MAX) - 0.5;
    double output_bias = ((double) std::rand() / RAND_MAX) - 0.5;

    double learning_rate = 0.5;
    int epochs = 1000000;

    // Eğitim döngüsü
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < input.size(); i++) {
            double input1 = input[i][0];
            double input2 = input[i][1];
            double input3 = input[i][2];
            double input4 = input[i][3];
            double input5 = input[i][4];
            double input6 = input[i][5];
            double input7 = input[i][6];
            double input8 = input[i][7];
            double input9 = input[i][8];

            double target = sonuc[i][0];

        double hidden_input1 = input1 * weight1_1 + input2 * weight1_2 + input3 * weight1_3 + input4 * weight1_4 +
                             input5 * weight1_5 + input6 * weight1_6 + input7 * weight1_7 + input8 * weight1_8 + input9 * weight1_9 + bias1;
        double hidden_output1 = sigmoid(hidden_input1);

        double hidden_input2 = input1 * weight2_1 + input2 * weight2_2 + input3 * weight2_3 + input4 * weight2_4 +
                             input5 * weight2_5 + input6 * weight2_6 + input7 * weight2_7 + input8 * weight2_8 + input9 * weight2_9 + bias2;
        double hidden_output2 = sigmoid(hidden_input2);

            double final_input = hidden_output1 * output_weight1 + hidden_output2 * output_weight2 + output_bias;
            double final_output = sigmoid(final_input);

            double error = target - final_output;

            // Geri yayılım (backpropagation)
            double d_output = error * sigmoid_derivative(final_output);

            double error_hidden1 = d_output * output_weight1;
            double error_hidden2 = d_output * output_weight2;

            double d_hidden1 = error_hidden1 * sigmoid_derivative(hidden_output1);
            double d_hidden2 = error_hidden2 * sigmoid_derivative(hidden_output2);
            //ağırlıkları tekrar düzenliyor
            output_weight1 += learning_rate * d_output * hidden_output1;
            output_weight2 += learning_rate * d_output * hidden_output2;
            output_bias += learning_rate * d_output;

            weight1_1 += learning_rate * d_hidden1 * input1;
            weight1_2 += learning_rate * d_hidden1 * input2;
            bias1 += learning_rate * d_hidden1;

            weight2_1 += learning_rate * d_hidden2 * input1;
            weight2_2 += learning_rate * d_hidden2 * input2;
            bias2 += learning_rate * d_hidden2;
        }
    }
     for (size_t i = 0; i < input.size(); ++i) {
        double input1 = input[i][0];
        double input2 = input[i][1];
        double input3 = input[i][2];
        double input4 = input[i][3];
        double input5 = input[i][4];
        double input6 = input[i][5];
        double input7 = input[i][6];
        double input8 = input[i][7];
        double input9 = input[i][8];


        double hidden_input1 = input1 * weight1_1 + input2 * weight1_2+input3*weight1_3 
      + input4 * weight1_4 + input5 * weight1_5 +input6 * weight1_6+ input7 *weight1_7
      + input8 * weight1_8 + input9 * weight1_9 + bias1;
        double hidden_output1 = sigmoid(hidden_input1);

        double hidden_input2 = input1 * weight2_1 + input2 * weight2_2+input3*weight2_3 
      + input4 * weight2_4 + input5 * weight2_5 +input6 * weight2_6+ input7 *weight2_7
      + input8 * weight2_8 + input9 * weight2_9 + bias1;
        double hidden_output2 = sigmoid(hidden_input2);

        double final_input = hidden_output1 * output_weight1 + hidden_output2 * output_weight2 + output_bias;
        double final_output = sigmoid(final_input);

        std::cout << "Giriş: (" << input1 << ", " << input2 << ") -> Çıkış: " << final_output << std::endl;
    }

    return 0;
}

