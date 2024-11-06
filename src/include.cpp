#include "../include/include.h" // Gerekli başlık dosyasını dahil et

// Veri setindeki tüm değerlerin ortalamasını hesaplayan fonksiyon
double ortalama(const std::vector<std::vector<double>>& veriSeti) {
    double toplam = 0.0; // Değerlerin toplamını saklamak için değişken
    for (long unsigned int i = 0; i < veriSeti.size(); i++) { // Her satır için döngü
        for(const auto& y : veriSeti[i]) { // Her sütun için döngü
            toplam += y; // Değeri toplam değişkenine ekle
        }
    }
    return toplam / (veriSeti.size() * 30); // Toplamı tüm eleman sayısına bölerek ortalamayı hesapla
}

// Veri setinin standart sapmasını hesaplayan fonksiyon
double standartSapma(const std::vector<std::vector<double>>& veriSeti) {
    double toplam = 0.0; // Toplamı saklamak için değişken
    for(long unsigned int j = 0; j < veriSeti.size(); j++) { // Her satır için döngü
        for (const auto& x : veriSeti[j]) { // Her sütun için döngü
            toplam += pow(x - ortalama(veriSeti), 2); // Ortalama farkının karesini ekle
        }
    }
    return sqrt(toplam / (veriSeti.size() * 30)); // Toplamı eleman sayısına bölüp karekökünü alarak standart sapmayı hesapla
}

// Sigmoid aktivasyon fonksiyonu
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x)); // Sigmoid formülü
}

// Sigmoid türevi fonksiyonu (gerçekleşen değere göre)
double sigmoid_derivative(double x) {
    return x * (1.0 - x); // Sigmoid türevi hesaplaması
}

// Verilen ağırlıklar ve giriş verisine göre tahmin yapan fonksiyon
double predict(const std::vector<std::vector<double>>& weights1, const std::vector<double>& weights2, const std::vector<double>& input_data,
               const double& ortalama_Deger, const double& standart_sapma_degeri) {
    std::vector<double> layer1(4); // Gizli katman değerlerini saklamak için vektör
    for (size_t k = 0; k < 4; ++k) {
        layer1[k] = 0.0; // Gizli katmandaki her düğüm değerini sıfırla
        for (size_t l = 0; l < 30; ++l) {
            // Giriş verisini normalize et ve ağırlıklandırılmış değeri ekle
            double normalizasyon_yapilmis_deger = ((input_data[l] - ortalama_Deger) / standart_sapma_degeri);
            layer1[k] += normalizasyon_yapilmis_deger * weights1[l][k];
        }
        layer1[k] = sigmoid(layer1[k]); // Gizli katmana sigmoid aktivasyon fonksiyonu uygula
    }

    double output = 0.0; // Çıkışı başlat
    for (size_t k = 0; k < 4; ++k)
        output += layer1[k] * weights2[k]; // Çıktıyı gizli katmandan ağırlıklandırılmış verilerle hesapla

    output = sigmoid(output); // Çıkışa sigmoid fonksiyonu uygula
    return output;
}

// Belirtilen ondalık basamak sayısına yuvarlayan fonksiyon
double roundToDecimal(double value, int decimalPlaces) {
    double factor = std::pow(10.0, decimalPlaces); // Yuvarlama faktörünü belirle
    return std::round(value * factor) / factor; // Değeri yuvarlayıp faktöre bölerek sonucu döndür
}

