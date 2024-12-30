#include "./veriseti.cpp" // Gerekli veri setini sağlayan dosya
#include <algorithm> // Karıştırma (shuffle) işlemi için gerekli kütüphane
#include <ostream>
using namespace std; // Standart isim alanını kullan

int main() {
  veriler veriler1; // Veri seti ve istatistiksel bilgiler içeren 'veriler' nesnesini oluştur
  
  int k = 5; // k-fold cross-validation için k değeri
  int epochs = 500; // Eğitim döngüsü sayısı
  double learning_rate = 0.3; // Öğrenme oranı

  srand(std::time(0)); // Rastgele sayılar üretmek için başlangıç değerini zaman ile ayarla
  random_device rd; // Rastgele aygıt oluştur
  mt19937 gen(rd()); // Rastgele sayı üretici (Mersenne Twister)
  uniform_real_distribution<double> dis(0.0, 1.0); // 0.0 ile 1.0 arasında uniform dağılımlı bir rastgele sayı üreteci tanımla

  vector<pair<vector<double>, double>> veri_seti; // Giriş ve çıkış değer çiftlerinden oluşan veri setini tanımla
  for (size_t i = 0; i < veriler1.giris_Degerleri.size(); i++) {
    veri_seti.push_back(make_pair(veriler1.giris_Degerleri[i], veriler1.cikis_Degerleri[i])); // Tüm verileri veri_seti'ne ekle
  }
  shuffle(veri_seti.begin(), veri_seti.end(), gen); // Veri setini karıştır

  size_t fold_size = veri_seti.size() / k; // Her fold'un boyutunu hesapla
  double toplam_basari_orani = 0.0; // Başarı oranını toplamak için başlangıç değeri

  for (int fold = 0; fold < k; ++fold) { // k katlamalı doğrulama döngüsü
    vector<pair<vector<double>, double>> egitim_verisi, test_verisi; // Eğitim ve test verileri için vektörler

    // Veri setini eğitim ve test olarak ayır
    for (size_t i = 0; i < veri_seti.size(); ++i) {
      if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
        test_verisi.push_back(veri_seti[i]); // Test verisi
      } else {
        egitim_verisi.push_back(veri_seti[i]); // Eğitim verisi
      }
    }

    vector<vector<double>> weights1(30, vector<double>(4)); // 30 giriş ve 4 gizli nöron için ağırlık matrisi
    vector<double> weights2(4); // 4 gizli nörondan çıkış nöronuna ağırlık vektörü

    // Ağırlıkları rastgele başlat
    for (size_t i = 0; i < 30; ++i)
      for (size_t j = 0; j < 4; ++j)
        weights1[i][j] = dis(gen);

    for (size_t i = 0; i < 4; ++i)
      weights2[i] = dis(gen);

    // Epoch döngüsü
    for (int epoch = 0; epoch < epochs; ++epoch) {
      for (const auto& [giris, cikis] : egitim_verisi) {
        vector<double> layer1(4); // Gizli katman çıktısı için vektör
        for (size_t k = 0; k < 4; ++k) {
          layer1[k] = 0.0;
          for (size_t l = 0; l < 30; ++l) {
            double norm_value = (giris[l] - veriler1.ortalama_Deger) / veriler1.standart_sapma_degeri; // Normalleştir
            layer1[k] += norm_value * weights1[l][k]; // Gizli katman nöronlarına giriş
          }
          layer1[k] = sigmoid(layer1[k]); // Aktivasyon fonksiyonu uygula
        }

        double output = 0.0; // Çıkış nöronu
        for (size_t k = 0; k < 4; ++k)
          output += layer1[k] * weights2[k];
        output = sigmoid(output); // Çıkış nöronuna sigmoid uygula

        double error = cikis - output; // Hata değeri
        vector<double> d_output(1); // Çıkış katmanı delta değeri
        d_output[0] = error * sigmoid_derivative(output); // Delta hesapla

        vector<double> d_hidden_layer(4); // Gizli katman delta değerleri
        for (size_t k = 0; k < 4; ++k)
          d_hidden_layer[k] = d_output[0] * weights2[k] * sigmoid_derivative(layer1[k]); // Gizli katman deltalarını hesapla

        // Ağırlıkları güncelle
        for (size_t l = 0; l < 30; ++l) {
          double norm_value = (giris[l] - veriler1.ortalama_Deger) / veriler1.standart_sapma_degeri; // Normalleştir
          for (size_t k = 0; k < 4; ++k) {
            weights1[l][k] += learning_rate * d_hidden_layer[k] * norm_value; // Gizli katman ağırlıklarını güncelle
          }
        }

        for (size_t k = 0; k < 4; ++k) {
          weights2[k] += learning_rate * d_output[0] * layer1[k]; // Çıkış katmanı ağırlıklarını güncelle
        }
      }
    }

    int dogru_sayisi = 0; // Doğru tahmin sayısı
    for (const auto& [giris, cikis] : test_verisi) {
      double tahmin = predict(weights1, weights2, giris, veriler1.ortalama_Deger, veriler1.standart_sapma_degeri); // Tahmin et
      tahmin = round(tahmin); // Tahmini yuvarla
      double gercek_deger = cikis;
      cout << tahmin << " " << gercek_deger << endl; // Tahmin ve gerçek değeri yazdır
      if (abs(tahmin - gercek_deger) <= 0) // Yaklaşık doğru tahmin kontrolü
        dogru_sayisi++;
    }
    double basari_orani = (static_cast<double>(dogru_sayisi) / test_verisi.size()) * 100.0; // Başarı oranı
    toplam_basari_orani += basari_orani; // Toplam başarı oranına ekle
  }

  double ortalama_basari_orani = toplam_basari_orani / k; // Ortalama başarı oranını hesapla
  cout << "Ortalama Başarı Oranı: " << ortalama_basari_orani << "%" << endl; // Ortalama başarı oranını yazdır
  return 0;
}

