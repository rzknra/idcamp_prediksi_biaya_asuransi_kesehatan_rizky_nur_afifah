# Laporan Proyek Machine Learning - Prediksi Biaya Asuransi Kesehatan

## Domain Proyek
Seiring berjalannya waktu, berbagai jenis penyakit baru yang menurunkan kualitas kesehatan masyarakat bermunculan. Di sisi lain, mahalnya biaya kesehatan mengakibatkan akses ke pelayanan kesehatan pada umumnya masih rendah [[4](http://journal.unhas.ac.id/index.php/jmsk/article/view/8312)]. Oleh karena itu, perlu adanya upaya untuk mengatasi kondisi tersebut. Asuransi kesehatan  adalah  upaya  untuk  mengatasi resiko ketidakpastian akibat sakit dan biaya-biaya yang ditimbulkannya [[3](http://eprints.undip.ac.id/65328/)]. 

Peserta asuransi kesehatan (tertanggung) perlu membayarkan sejumlah biaya kepada perusahaan asuransi kesehatan (penanggung) yang  besarnya sudah  ditentukan [[1](http://journal.unhas.ac.id/index.php/jmsk/article/view/8312)]. Beberapa faktor berpengaruh terhadap biaya asuransi kesehatan, yaitu *body mass index* (bmi), umur, jumlah anak, jenis kelamin, status merokok, dan wilayah tinggal. Algoritma *machine learning* bisa dimanfaatkan untuk membantu memprediksi biaya asuransi kesehatan berdasarkan faktor-faktor tersebut, dua diantaranya yaitu Random Forest dan XGBoost [[2](https://ieeexplore.ieee.org/abstract/document/9793258?casa_token=r6h40NMURLoAAAAA:jkzdsMcrd424fZcgOSUK0tRgxuJMliYR85RPWQ-nZLLRku00-cvZFbxWNB43afvIolAdGIu-ZX2lvg)] [[5](http://journal.unuha.ac.id/index.php/JICode/article/view/3294)].

## Business Understanding
Pengembangan model prediksi biaya kesehatan bertujuan untuk membantu pengambilan keputusan oleh calon peserta asuransi kesehatan atau tertanggung, selain itu juga membantu tertanggung dan penjual asuransi kesehatan melaksanakan keputusan jual beli yang lebih bijaksana. 

### Problem Statements
- Apa saja fitur yang berpengaruh terhadap biaya asuransi kesehatan?
- Bagaimana cara mengolah *dataset* sehingga bisa digunakan untuk mengembangkan model prediksi biaya asuransi kesehatan?
- Bagaimana cara meningkatkan performa dan menyeleksi model prediksi biaya asuransi kesehatan sehingga diperoleh model terbaik?

### Goals
- Mengetahui fitur yang berpengaruh terhadap biaya asuransi kesehatan.
- Mengetahui cara mengolah *dataset* sehingga bisa digunakan untuk mengembangkan model prediksi biaya asuransi kesehatan.
- Mengetahui cara meningkatkan performa dan menyeleksi model prediksi biaya asuransi kesehatan sehingga diperoleh model terbaik.

### Solution Statements 
- Untuk mengetahui fitur yang berpengaruh terhadap biaya asuransi kesehatan, dilakukan Analisis Univariat dan Analisis Multivariat. Analsisis univariat dilakukan menggunakan *barplot* dan *histplot*, sedangkan analisis multivariat dilakukan menggunakan *catplot, pairplot* dan *correlation matrix*. 
- Untuk mengolah *dataset* sehingga bisa digunakan untuk mengembangkan model prediksi biaya asuransi kesehatan, dilakukan *Data Wragling* dan *Data Preparation*. *Data Wragling* yang dilakukan berupa *Data Assesing* dan *Data Cleaning*, sedangkan *Data Preparation* yang dilakukan berupa *Encoding* fitur kategori, pembagian dataset menjadi data latih-data uji, dan *Scaling* fitur numerik.
- Untuk meningkatkan performa model prediksi biaya asuransi kesehatan, dilakukan hyperparameter tuning terhadap baseline model. Lebih lanjut, untuk menyeleksi model prediksi biaya asuransi kesehatan, dilakukan perbandingan nilai metrik evaluasi menggunakan tabel dan *bar plot* sehingga diperoleh model terbaik.

## Data Understanding
Data yang digunakan dalam pengembangan model merupakan data sekunder yang diperoleh dari Kaggle dengan nama *dataset* yaitu 'US Health Insurance Dataset'. Data tersebut dapat diakses melalui tautan berikut: 
https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset.

Lebih lanjut, detail mengenai *dataset* tersebut diberikan sebagai berikut:
- Dataset berupa file CSV.
- Dataset terdiri dari 1338 *record* (pengamatan) dan 7 fitur. 
- Dataset terdiri dari 3 fitur kategori (*sex, smoker*, dan *region*) dan 4 fitur numerik (*bmi, charge, age*, dan *children*).

### Variabel-Variabel dari US Health Insurance Dataset
Berdasarkan informasi di Kaggle, variabel-variabel pada *dataset* adalah sebagai berikut:
1. *age*: usia tertanggung asuransi (dalam satuan tahun).
2. *sex*: jenis kelamain tertanggung asuransi (*male* atau *famale*).
3. *bmi* (*body mass index*): nilai perbandingan antara berat badan dan kuadrat dari tinggi badan (dalam satuan $kg/m^2$).
4. *children*: jumlah anak yang ditanggung oleh penyedia asuransi kesehatan.
5. *smoker*: status merokok tertanggung asuransi (*yes* atau *no*).
6. *region*: daerah pemukiman tertanggung asuransi di US (meliputi  *southwest, southeast, northwest*, dan *northeast*).
7. *charges*: besar biaya asuransi yang dibebankan kepada tertanggung asuransi (dalam satuan dolar US \$).

Lebih lanjut, dilakukan juga pengecekan deskripsi statistik dari setiap variabel/fitur. Untuk fitur kategori, diperoleh bahwa:
1. fitur 'sex' mempunyai nilai unik, yaitu *female* dan *male*.
2. fitur 'smoker' mempunyai nilai unik, yaitu *yes* dan *no*.
3. fitur 'region' mempunyai nilai unik, yaitu *southwest, southeast, northwest*, dan *northeast*.

Untuk fitur numerik, diperoleh dekripsi statistik berikut. 

Tabel 1. Deskripsi Statistik untuk Fitur Numerik

|           | age       |	bmi	    | children  |	charges |
------------|-----------|-----------|-----------|-----------|
|count	    | 1338.000000	| 1338.000000	| 1338.000000	| 1338.000000| 
|mean	    | 39.207025	| 30.663397	| 1.094918	| 13270.422265| 
|std	    | 14.049960	| 6.098187	| 1.205493	| 12110.011237| 
|min	    | 18.000000	| 15.960000	| 0.000000	| 1121.873900| 
|25%	    | 27.000000	| 26.296250	| 0.000000	| 4740.287150| 
|50%	    | 39.000000	| 30.400000	| 1.000000	| 9382.033000| 
|75%	    | 51.000000	| 34.693750	| 2.000000	| 16639.912515| 
|max	    | 64.000000	| 53.130000	| 5.000000	| 63770.428010| 

Deskripsi statistik di atas meliputi:
1. *Count* adalah jumlah sampel pada data.
2. *Mean* adalah nilai rata-rata.
3. *Std* adalah standar deviasi.
4. *Min* yaitu nilai minimum setiap kolom.
5. 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
6. 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
7. 75% adalah kuartil ketiga.
8. Max adalah nilai maksimum.

### Data Assesing
Pada tahap ini, dilakukan pengecekan terhadap:
1. Data duplikat (data yang bernilai sama dengan data lainnya).
2. *Missing value* (data yang hilang atau tidak tersedia).
3. *Outlier* (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Berdasarkan *Data Assesing* yang sudah dilakukan, diperoleh bahwa:
1. Ada 1 *record* yang terduplikat.
2. Tidak ada *missing value*.
3. Fitur 'bmi' dan 'charges' mempunyai *outlier*.

### Data Cleaning 
Berdasarkan hasil dari *Data Assesing*, maka selanjutnya dilakukan *Data Cleaning* atau pembersihan data yang meliputi:
1. Penghapusan *record* yang terduplikat.
2. Penggunaan metode *Inter Quartile Range* (IQR) untuk mengatasi *outlier*, yaitu 
$$IQR = Q_3 - Q_1,$$ 
dengan:
    - $IQR$ = *Inter Quartile Range*
    - $Q_3$ = Quartile 3
    - $Q_1$ = Quartile 1

Setelah *record*  yang terduplikat dihapus, diperoleh *dataset* baru yang terdiri dari 1337 *record*. Lebih lanjut, setelah diimplementasikan metode IQR, diperoleh *dataset* baru yang terdiri dari 1192 *record*. *Dataset* inilah yang akan digunakan dalam tahap selanjutnya.

### Analisis Univariat dan Analisis Multivariat
Data yang sudah dibersihkan selanjutnya bisa digunakan untuk analisis univariat dan analisis multivariat. Analisis univariat dilakukan untuk mengetahui dan mengindentifikasi karakterisitik dari setiap fitur, sedangkan analisis multivariat digunakan untuk mengetahui hubungan antar fitur. Lebih lanjut, analisis univariat dilakukan terhadap fitur kategori maupun fitur numerik sebagai berikut.

a. Fitur Kategori

Diperhatikan Gambar 1a berikut ini.

![univariat_kategori_sex](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/5b44dca9-fcd2-4088-97ba-44e9824d0ba4)

![univariat_kategori_smoker](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/83dfda08-3b29-4240-97b6-9c7f2e6f020b)

![univariat_kategori_region](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/0902a5ce-516e-4a52-a8f2-e700c5f3132d)

Gambar 1a. Analisis Univariat (Fitur Kategori)

Berdasarkan Gambar 1a di atas, diperoleh bahwa:
1. Banyak tertanggung asuransi laki-laki dan perempuan hampir berimbang, yaitu 50.4% laki-laki dan 49.6% perempuan.
2. Sebagaian besar tertangggug asuransi bukan perokok. Hal ini terlihat dari sebanyak 79,6% tertanggung asuransi bukan perokok, sisanya sebesar 20,6% tertanggung asuransi merupakan perokok.
4. Banyak tertanggung asuransi yang tinggal pada tiap-tiap daerah pemukiman hampir sama, yaitu 26.9% di *southeast*, 26.9% di *southwest*, 24.4% di *northwes*t, dan 24.3% di *northeast*.

b. Fitur Numerik
   
Diperhatikan Gambar 1b berikut ini.

![univariat_numerik](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/9bba4270-6d0f-4d89-8faf-b80e099eec3f)

Gambar 1b. Analisis Univariat (Fitur Numerik)

Berdasarkan Gambar 1b di atas, diperoleh bahwa:
1. Peningkatan nilai 'children' dan 'charges' sebanding dengan penurunan jumlah sampel. Hal ini terlihat dari histrogram 'children' dan 'charges' yang grafiknya mengalami penurunan seiring semakin banyaknya jumlah sampel.
2. Distribusi 'charges', 'age', dan 'children' miring ke kanan, sedangkan distribusi 'bmi' cenderung normal.
3. Rentang 'charges' cukup tinggi yaitu dari skala ratusan dolar amerika hingga sekitar \$60000.

Setelah dilakukan analisis univariat, selanjutnya dilakukan analisis multivariat. Analsis multivariat yang dilakukan di sini akan lebih berfokus untuk mengetahui hubungan antara fitur target 'charges' dengan fitur-fitur lainnya, baik fitur kategori maupun numerik. Berikut ini dilakukan analisis multivariat tersebut. 

a. Fitur Kategori

Pertama, dilakukan pengecekan rata-rata 'charges' terhadap fitur kategori untuk mengetahui pengaruh fitur kategori terhadap 'charges'. Diperhatikan Gambar 2a berikut.

![multivariat_kategori_sex](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/63deb315-f3ef-40f6-b5f2-25661284ef12)

![multivariat_kategori_smoker](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/f12ad28b-456a-419e-885b-68813b7f6e50)

![multivariat_kategori_region](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/e746debd-8f47-44a6-a863-69b335a2c2d5)

Gambar 2a. Analisis Multivariat (Fitur Kategori)

Berdasarkan Gambar 2a di atas, diperoleh bahwa:
1. Pada fitur 'sex', rata-rata 'charges' yang dikenakan terhadap 'male' dan 'female' cenderung mirip. Rentangnya berada antara  \$12000  sampai \$15000 . Dengan demikian, fitur 'sex' mempunyai pengaruh kecil terhadap rata-rata 'charges'.
2. Pada fitur 'smoker', rata-rata 'charges' untuk 'smoker' jauh lebih besar dibandingkan untuk 'non smoker'. Dengan demikian, fitur 'smoker' mempunyai pengaruh besar terhadap rata-rata 'charges'.
3. Pada fitur 'region', rata-rata 'charges' yang dikenakan terhadap masing-masing region cenderung mirip. Rentangnya berada antara  \$12000  sampai \$16000 . Dengan demikian, fitur 'region' mempunyai pengaruh kecil terhadap rata-rata 'charges'.
4. Karena fitur 'sex' dan 'region' mempunyai pengaruh kecil terhadap rata-rata 'charges', maka kedua fitur tersebut bisa dihilangkan sehingga hanya tersisa fitur 'smoker' untuk fitur kategori.

b. Fitur Numerik

Selanjutnya, dilakukan analisis multivariat terhadap fitur numerik sedemikian sehingga diperoleh Gambar 2b berikut. 

![multivariat_numerik](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/d57796c3-3023-4c0d-88aa-bdfc42943f9b)

Gambar 2b. Analisis Multivariat (Fitur Numerik)

Untuk membaca *pairplot* di atas, perhatikan fitur target 'charges' pada sumbu y. Terlihat bahwa fitur 'age' mempunyai korelasi positif dengan fitur 'price'. Sedangkan, fitur 'bmi' dan 'children' tidak mempunyai korelasi dengan fitur 'price'.

Lebih lanjut, untuk mengecek hubungan antar fitur, digunakan juga *correlation matrix* berikut ini.

![matriks korelasi](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/28070d73-7f63-4d2d-9116-c5b93695d183)

Gambar 3. Correlation Matrix untuk Fitur Numerik

Berdasarkan  *correlation matrix* di atas, terlihat bahwa fitur 'age' mempunyai korelasi yang cukup besar dengan fitur 'charges', sedangkan fitur 'bmi' dan 'children' mempunyai korelasi kecil dengan fitur 'charges'. Oleh karena itu, fitur 'bmi' dan 'children' bisa dihilangkan sehingga tidak digunakan dalam tahap selanjutnya. 

## Data Preparation 
*Data preparation* dilakukan untuk mentransformasi data sehingga menjadi bentuk yang cocok dalam proses pemodelan. Pada bagian ini, dilakukan tiga tahap persiapan data, yaitu:
1. Encoding Fitur Kategori.
Encoding dilakukan untuk mengubah fitur kategori menjadi fitur numerik. Salah satu teknik yang digunakan untuk encoding yaitu *label encoding* dengan menggunakan fungsi LabelEncoder dari library ScikitLearn. Fitur kategori dari *dataset* yang dikenakan encoding yaitu 'sex' sehingga diperoleh .
2. Pembagian Dataset.
Pembagian *dataset* menjadi data latih (train) dan data uji (test) perlu dilakukan sebelum membangun model. Pembagian *dataset* ini bisa dilakukan menggunakan fungsi train_test_split dari library ScikitLearn. Adapun proporsi pembagian data latih dan data uji yang digunakan dalam tahap ini, yaitu 80:20, yang merupakan proporsi umum yang sering digunakan. Dengan proporsi pembagian 80:20, maka diperoleh:
    - data latih terdiri dari 953 *record*. 
    - data uji terdiri dari 239 *record*. 
3. Scaling Fitur Numerik.
Scaling atau penyekelaan ialah proses mengubah data sehingga mempunyai nilai dalam rentang tertentu. Untuk fitur numerik, salah satu teknik yang umum untuk digunakan yaitu MinMaxScaler dari library Scikitlearn. MinMaxScaler mentransformasi data sehingga nilainya berada dalam rentang 0 hingga 1. MinMaxScaler melakukan proses penyekalaan fitur dengan mengurangkan nilai minimal fitur (min) kemudian membaginya dengan selisih dari nilai minimal fitur (min) dan nilai maksimal fitur(max), yaitu
$$x_{scaled} = \frac{x - min}{max - min},$$
dengan:
    - $x_{scaled}$ = hasil scaling data
    - $x$ = nilai data pada fitur yang di scaling
    - $min$ = nilai minimal fitur
    - $max$ = nilai maksimal fitur 

## Modeling
Pada tahap ini dikembangkan model *machine learning* dengan dua algoritma, yaitu:
1. Random Forest.
2. XGBoost.

Kedua model tersebut merupakan model *ensemble* (*group*), yaitu model prediksi yang terdiri dari beberap model dan bekerja secara bersama-sama. Ada dua teknik pendekatan dari model *ensemble*, yaitu *bagging* dan *boosting*. Random forest merupakan versi *bagging* dari algoritma Decision Tree, sedangkan XGBoost merupakan versi *boosting* dari algoritma Decision Tree.

Algoritma Random Forest dan XGBoost mempunyai kelebihan dan kekurangan masing-masing. Kelebihan dan kekurangan algoritma Random Forest, yaitu:
1. Kelebihan 
    - Mengurangi varians dalam model dan meningkatkan generalisasi data baru sehingga mengurangi overfitting.
    - Cenderung stabil.
    - Bekerja dengan baik pada data yang tidak seimbang. 
    - Bekerja dengan baik pada data kecil.
2. Kekurangan
    - Cenderung bias terhadap data kategori. 
    - Waktu komputasi pada dataset berskala besar cenderung lambat. 

Di sisi lain, kelebihan dan kekurangan algoritma XGBoost, yaitu:
1. Kelebihan 
    - Mengurangi bias dalam model sehingga mengurangi underfitting.
    - Dirancang untuk pelatihan model yang efisien dan *scalable* sehingga cocok untuk bekerja pada data besar.
    - Bekerja dengan baik pada data dengan *misssing value*.
2. Kekurangan
    - Lebih rentan terhadap overfitting. 
    - Menggunakan banyak ruang penyimpanan dan kompleksitas komputasai yang tinggi terutama ketika berkerja pada data yang besar sehingga tidak cocok untuk sistem dengan sumber daya terbatas.

Performa model yang dibangun dengan algoritma Random Forest dan Decision Tree dioptimalkan dengan menerapkan *hyperparameter tuning* pada *baseline* model Random Forest dan XGBoost sedemikian sehingga terdapat empat model *machine learning* yang dikembangkan, yaitu:
1. RF1: Random Forest
2. RF2: Random Forest with Hyperparamter Tuning
3. XGB1: XGBoost 
4. XGB2: XGBoost with Hyperparameter Tuning

*Hyperparameter* dari algoritma Random Forest yang di-*tuning*, yaitu:
1. max_depth: kedalaman atau panjang pohon, yang berarti ukuran seberapa banyak pohon dapat membelah (*splitting*) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
2. min_samples_leaf: jumlah minimum sampel yang diperlukan untuk berada di *leafs* (daun).
3. mis_samples split: jumlah minimum sampel yang diperlukan untuk membagi node internal.
4. n_estimator: jumlah *trees* (pohon) di forest.

Di sisi lain, *hyperparameter* dari algoritma XGBoost yang di-*tuning*, yaitu:
1. n_estimator: jumlah trees (pohon) di forest.
2. learning_rate: bobot yang diterapkan pada setiap *regresso*r di masing-masing proses iterasi *boosting*.
2. max_depth: kedalaman atau panjang pohon, yang berarti ukuran seberapa banyak pohon dapat membelah (*splitting*) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.

*Hyperparameter tuning* dilakukan dengan menggunakan fungsi GridSearchCV dari library ScikitLearn sedemikian sehingga diperoleh nilai *hyparameter* terbaik, yaitu:
1. Random Forest
```
{'max_depth': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 2,
 'n_estimators': 100}
 ```
 2. XGBoost
 ```
 {'learning_rate': 0.01,
 'max_depth': 3,
 'n_estimators': 300}
 ```
 Nilai *hyperparameter* tersebut yang digunakan untuk mengembangkan model RF2 dan XGB2.

## Evaluation
Model *machine learning* yang dibangun di atas merupakan model regeresi, akibatnya untuk mengevaluasi performa model tersebut bisa digunakan tiga metrik evaluasi, yaitu:
1. MSE (*Mean Squarred Error*).
2. MAE (*Mean Absolute Error*).
3. $R^2$ (*R-Squarred*).

Berikut ini diberikan rumus dari ketiga metrik evaluasi di atas, yaitu:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^2,$$
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}|,$$
$$R^2 = 1 - (MSE/Var(y)),$$
dengan:
- $n$ = ukuran sampel
- $y_i$ = nilai data aktual
- $\hat{y}$ = nilai data hasil prediksi
- $var(y)$ = variansi dari data aktual 

Adapun kegunaan dari setiap metrik evaluasi di atas, yaitu:
- MSE untuk menghitung rata-rata dari selisih kuadrat dari nilai hasil data hasil prediksi dan nilai data aktual. Semakin kecil nilai MSE, maka semakin baik kualitas model tersebut. 
- MAE untuk menghitung selisih absolut dari nilai data hasil prediksi dan nilai data aktual. Semakin kecil nilai MAE, maka semakin baik kualitas model tersebut. 
- $R^2$ untuk mengetahui seberapa besar pengaruh variabel independen tertentu terhadap variabel dependen. 

Selanjutnya, dibandingkan nilai ketiga metrik evaluasi dari masing-masing model. Nilai metrik evaluasi MSE terhadap data latih (train_mse) dan data uji (test_mse) dibandingkan menggunakan *barplot* berikut ini.

![perbandingan_mse](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/0864012e-b013-44ff-bf9a-0c43f9d6a68d)

Gambar 4. Perbandingan Nilai MSE terhadap Data Latih dan Data Uji

Berdasarkan Gambar 4 (*barplot*) tersebut diperoleh bahwa model RF2 dan XGB2 mempunyai nilai 'test_mse' yang lebih rendah dibandingkan kedua model lainnya. Dilain hal, selisih nilai 'test_mse' dan 'train_mse' pada model RF2 dan XGB2 lebih kecil dibandingkan dua model lainnya, yang berarti kedua model tidak terlalu *overfitting*. Dengan demikian, bedasarkan perbandingan nilai metrik evaluasi MSE terhadap data latih (train_mse) dan data uji (test_mse) tersebut diperoleh model RF2 dan XGB2 lebih baik dari kedua model lainnya.

Lebih lanjut, dibandingkan nilai ketiga metrik evaluasi yaitu nilai MSE terhadap data uji (test_MSE), MAE, dan $R^2$ dari setiap model *machine learning* yang sudah dibangun menggunakan Tabel 2 berikut.

Tabel 2. Nilai Metrik Evaluasi Setiap Model
|     | MSE | MAE | $R^2$ | 
|-----|-----|-----|-----|
| RF1 | 25282.763885 | 2922.9584113729475 | 0.5660674252523771 |
| RF2 | 24324.410399 | 2897.073605878178 | 0.5825158166395445 |
| XGB1 | 25967.960248 | 2948.621166460513 | 0.5543072781685672 |
| XGB2 | 23633.954472 | 2848.303577468292 | 0.594366234561500 |

Berdasarkan Tabel 2 di atas, diperoleh bahwa model RF2 dan XGB2 mempunyai nilai MSE dan MAE terhadap data uji (test_mse) yang lebih rendah dibandingkan kedua model lainnya. Di sisi lain, nilai $R^2$ model RF2 dan XGB2 yang lebih tinggi dibandingkan dua model lainnya. Dengan demikian, berdasarkan perbandingan ketiga nilai metrik evaluasi tersebut diperoleh model RF2 dan XGB2 lebih baik daripada kedua model lainnya. 

Oleh karena itu, berdasarkan hasil perbandingan nilai metrik evaluasi MSE (train_MSE dan test_MSE), MAE, dan $R^2$ menggunakan Gambar 4 (*barplot*) dan Tabel 2 di atas, diperoleh bahwa model RF2 (Random Forest with Hyperparameter Tuning) dan XGB2 (XGBoost with Hyperparameter Tuning) adalah model terbaik untuk prediksi biaya asuransi kesehatan.

Berikut ini berikan hasil pengujian masing-masing model dengan menggunakan salah satu *record* data dari data uji.

Tabel 3. Hasil Prediksi Setiap Model
| y_true	| prediksi_RF1	|prediksi_RF2 |	prediksi_XGB1 |	prediksi_XGB2 |
|----|---|---|---|---|
| 6877.9801  | 8472.5 | 7024.5 | 8458.5 | 7274.700195 |

Terlihat bahwa model RF2 dan XGB2 memberikan hasil yang paling mendekati y_true (nilai asli).


## Referensi 
[1] Iriana, N., & Nasution, Y. N. (2020). Penentuan Cadangan Premi Asuransi Jiwa Seumur Hidup Menggunakan Metode Zillmer. Jurnal Matematika, Statistika Dan Komputasi, 16(2), 219-225.

[2] Jyothsna, C., Srinivas, K., Bhargavi, B., Sravanth, A. E., Kumar, A. T., & Kumar, J. S. (2022). Health Insurance Premium Prediction using XGboost Regressor. 2022 International Conference on Applied Artificial Intelligence and Computing (ICAAIC) IEEE, 1645-1652.

[3] Munadi  S.  (2009).  Asuransi  Kesehatan  Kumpulan untuk Perawatan Rumah Sakit. Jurnal Matematika. 12(2), 61-69.

[4] Setyawan, F. E. B. (2015). Sistem Pembiayaan Kesehatan. Saintika Medika, 11(2), 119-126.

[5] Sumantiawan, D. I. (2024). METODE ANALASIS MENGGUNAKAN ALGORITMA RANDOM FOREST UNTUK PREDIKSI BIAYA ASURANSI KESEHATAN. JICode: Jurnal Informatika dan Komputer, 1(1), 1-8.
