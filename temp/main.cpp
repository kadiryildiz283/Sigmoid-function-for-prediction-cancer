#include "../include/include.h" // Gerekli kütüphaneleri ve fonksiyonları içeren başlık dosyasını dahil et

using namespace std; // std ad alanını kullanarak kodda std:: kullanımı gereksinimini ortadan kaldır

int main() {
    double ortalama_Deger = ortalama(X); // Veri kümesindeki değerlerin ortalamasını hesapla
    double standart_sapma_degeri = standartSapma(X); // Veri kümesindeki standart sapmayı hesapla
    double learning_rate = 0.3; // Öğrenme oranını 0.3 olarak belirle
    int epochs = 80000; // Eğitim için yineleme sayısını 80000 olarak ayarla

    srand(std::time(0)); // Rastgele sayı üretimi için zaman tabanlı seed kullan
    random_device rd; // Rastgele cihaz (donanım bazlı rastgelelik)
    mt19937 gen(rd()); // Mersenne Twister rastgele sayı üreteci
    uniform_real_distribution<double> dis(0.0, 1.0); // 0.0 ile 1.0 arasında rastgele dağılım oluştur

    // İlk katmandaki ağırlıkları rastgele başlat
    for (size_t i = 0; i < 9; ++i)
        for (size_t j = 0; j < 4; ++j)
            weights1[i][j] = dis(gen);

    // Çıktı katmanı ağırlıklarını rastgele başlat
    for (size_t i = 0; i < 4; ++i)
        weights2[i] = dis(gen);

    // Eğitim döngüsü
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = 0; j < X.size(); ++j) {
            if(j >= y.size()) break; // Dizide sınır aşımlarını önlemek için kontrol

            vector<double> layer1(4); // Gizli katman değerleri için vektör oluştur
            for (size_t k = 0; k < 4; ++k) {
                layer1[k] = 0.0; // Gizli katman elemanlarını sıfırla
                for (size_t l = 0; l < 9; ++l) {
                    // Giriş değerini normalize et ve gizli katmana ağırlıklandırarak ekle
                    double normalizasyon_yapilmis_deger = ((X[j][l] - ortalama_Deger)/ standart_sapma_degeri);
                    layer1[k] += normalizasyon_yapilmis_deger * weights1[l][k];
                }
                layer1[k] = sigmoid(layer1[k]); // Aktivasyon fonksiyonunu uygula
            }

            double output = 0.0; // Çıktıyı başlat
            for (size_t k = 0; k < 4; ++k)
                output += layer1[k] * weights2[k]; // Gizli katman çıktısını ağırlıklandırarak çıktı hesapla

            output = sigmoid(output); // Çıktıya aktivasyon fonksiyonunu uygula
            
            double error = roundToDecimal(((y[j] / 10.0) - output), 1); // Normalizasyon yaparak hatayı hesapla
            std::vector<double> d_output(1); // Çıktı katmanı hata oranı
            d_output[0] = error * sigmoid_derivative(output); // Çıkış katmanı hatasını sigmoid türevi ile çarp

            std::vector<double> d_hidden_layer(4); // Gizli katman hata oranı vektörü
            for (size_t k = 0; k < 4; ++k)
                d_hidden_layer[k] = d_output[0] * weights2[k] * sigmoid_derivative(layer1[k]); // Gizli katman hatasını hesapla

            // Ağırlıkları güncelleme - girişten gizli katmana
            for (size_t l = 0; l < 9; ++l) {
                double norm_value = ((X[j][l]- ortalama_Deger)/ standart_sapma_degeri); // Giriş değerini normalize et
                weights1[l][0] += learning_rate * d_hidden_layer[0] * norm_value;
                weights1[l][1] += learning_rate * d_hidden_layer[1] * norm_value;
                weights1[l][2] += learning_rate * d_hidden_layer[2] * norm_value;
                weights1[l][3] += learning_rate * d_hidden_layer[3] * norm_value;
            }

            // Ağırlıkları güncelleme - gizli katmandan çıktıya
            for (size_t k = 0; k < 4; ++k) {
                weights2[k] += learning_rate * d_output[0] * layer1[k];
            }
        }
    }

    // İlk tahmin için giriş verisi
    vector<double> input_data = {2.0, 7.0, 5.0, 55.0, 5.0, 3.0, 4.0, 0.0, 0.0};
    cout << "Tahmin: " << predict(weights1, weights2, input_data, ortalama_Deger, standart_sapma_degeri) << endl;

    // İkinci tahmin için giriş verisi
    vector<double> input_data2 = {18, 6.8, 1.6, 3.8, 0, 53, 6.34, 21.12, 10.07};
    cout << "Tahmin: " << predict(weights1, weights2, input_data2, ortalama_Deger, standart_sapma_degeri) << endl;

    // Standart sapma ve ortalama değerlerini ekrana yazdır
    cout << "Standart sapma: " << standart_sapma_degeri << " Ortalama: " << ortalama_Deger << endl;

    return 0; // Ana programı sonlandır
}

