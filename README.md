# Laporan Proyek Machine Learning - Prediksi Biaya Asuransi Kesehatan

## Domain Proyek
Seiring berkembangnya waktu, berbagai jenis penyakit baru yang menurunkan kualitas kesehatan masyarakat terus bermunculan [[4](http://journal.unhas.ac.id/index.php/jmsk/article/view/8312)]. Di sisi lain, mahalnya biaya kesehatan mengakibatkan rendahnya akses terhadap pelayanan kesehatan di banyak daerah [[4](http://journal.unhas.ac.id/index.php/jmsk/article/view/8312)]. Oleh karena itu, perlu adanya upaya untuk mengatasi tantangan tersebut. Salah satu upaya yang dapat dilakukan adalah melalui asuransi kesehatan, yang bertujuan untuk mengurangi risiko ketidakpastian akibat sakit dan biaya yang ditimbulkannya [[3](http://eprints.undip.ac.id/65328/)]. 

Peserta asuransi kesehatan (tertanggung) perlu membayar sejumlah biaya kepada perusahaan asuransi kesehatan (penanggung), dengan besar biaya yang sudah ditentukan sebelumnya [[1](http://journal.unhas.ac.id/index.php/jmsk/article/view/8312)]. Beberapa faktor yang memengaruhi biaya asuransi kesehatan meliputi: indeks massa tubuh (BMI), usia, jumlah anak, jenis kelamin, status merokok, dan wilayah tempat tinggal. Oleh karena itu, algoritma machine learning dapat dimanfaatkan untuk memprediksi biaya asuransi kesehatan berdasarkan faktor-faktor tersebut, dengan menggunakan metode seperti **Random Forest** dan **XGBoost** [[2](https://ieeexplore.ieee.org/abstract/document/9793258?casa_token=r6h40NMURLoAAAAA:jkzdsMcrd424fZcgOSUK0tRgxuJMliYR85RPWQ-nZLLRku00-cvZFbxWNB43afvIolAdGIu-ZX2lvg)] [[5](http://journal.unuha.ac.id/index.php/JICode/article/view/3294)].

## Business Understanding
Pengembangan model prediksi biaya kesehatan bertujuan untuk membantu pengambilan keputusan oleh calon peserta asuransi kesehatan atau tertanggung, selain itu juga membantu tertanggung dan penjual asuransi kesehatan melaksanakan keputusan jual beli yang lebih bijaksana. 

### Problem Statements
- Bagaimana cara mengolah dataset sehingga dapat digunakan untuk mengembangkan model **Random Forest** dan **XGBoost** dalam prediksi biaya asuransi kesehatan?
- Apa saja fitur yang berpengaruh terhadap biaya asuransi kesehatan?
- Bagaimana cara meningkatkan performa model **Random Forest** dan **XGBoost**?
- Bagaimana cara mengevaluasi performa dan menyeleksi model **Random Forest** dan **XGBoost**t untuk prediksi biaya asuransi kesehatan sehingga diperoleh model terbaik?

### Goals
- Mengolah dataset agar dapat digunakan untuk mengembangkan model **Random Forest** dan **XGBoost** dalam prediksi biaya asuransi kesehatan.
- Mengidentifikasi fitur-fitur yang berpengaruh terhadap biaya asuransi kesehatan.
- Meningkatkan performa model **Random Forest** dan **XGBoost**. 
- Mengevaluasi model dan menyeleksi model **Random Forest** dan **XGBoost** untuk prediksi biaya asuransi kesehatan.
 

### Solution Statements 
- **Pengolahan Dataset**: Untuk mengolah dataset sehingga bisa digunakan dalam model prediksi, dilakukan proses **data wrangling** dan **data preparation**. Proses data wrangling meliputi data assesing dan data cleaning, sementara data preparation melibatkan:
    - Encoding untuk fitur kategori menggunakan LabelEncoder.
    - Pembagian dataset menjadi data latih dan data uji.
    - Scaling untuk fitur numerik menggunakan MinMaxScaler.
- **Analisis Fitur**: Untuk mengetahui fitur yang berpengaruh terhadap biaya asuransi kesehatan, dilakukan **analisis univariat** dan **multivariat**. Analisis univariat menggunakan Barplot dan Histplot, sedangkan analisis multivariat menggunakan Catplot, Pairplot, dan Corellation Matrix. Lebih lanjut, kegunaan spesifik masing-masing visualisasi tersebut, yaitu:
    - Barplot: Untuk melihat distribusi fitur kategorikal.
    - Histplot: Untuk memeriksa distribusi fitur numerik dan mendeteksi outlier.
    - Catplot: Untuk melihat hubungan antara fitur kategorikal dan numerik.
    - Pairplot: Untuk memvisualisasikan hubungan antar fitur numerik.
    - Correlation matrix: Untuk mengetahui kekuatan korelasi antar fitur numerik dengan variabel target.
- **Modeling dan Peningkatan Performa**: Untuk membangun model prediksi, digunakan dua algoritma: Random Forest dan XGBoost. Selanjutnya, untuk meningkatkan performa model, dilakukan **hyperparameter tuning** terhadap model dasar (baseline model). 
- **Evaluasi dan Seleksi Model**: Kinerja model dievaluasi menggunakan beberapa metrik evaluasi, yaitu **MSE** (Mean Squared Error), **MAE** (Mean Absolute Error), **$R^2$** (Koefisien Determinasi). Evaluasi dilakukan untuk mengukur keakuratan prediksi model terhadap biaya asuransi kesehatan. Selanjutnya, seleksi model dilakukan dengan membandingkan nilai metrik evaluasi, baik menggunakan tabel maupun Barplot, untuk memilih model yang memberikan hasil terbaik.

 
## Data Understanding
Data yang digunakan dalam pengembangan model ini adalah data sekunder yang diperoleh dari Kaggle dengan nama dataset **'US Health Insurance Dataset**', yang dapat diakses melalui tautan berikut: 
https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset.

Lebih lanjut, detail mengenai dataset tersebut diberikan sebagai berikut:
- Dataset berupa file **CSV**.
- Dataset terdiri dari **1338 record dan 7 fitur**. 
- Dataset terdiri dari **3 fitur kategori** (sex, smoker, dan region) dan **4 fitur numeri**k (bmi, charge, age, dan children).

### Variabel-Variabel dari US Health Insurance Dataset
Berdasarkan informasi di Kaggle, variabel-variabel pada dataset adalah sebagai berikut:
1. **age**: usia tertanggung asuransi (dalam tahun).
2. **sex**: jenis kelamain tertanggung asuransi (male atau famale).
3. **bmi (body mass index)**: nilai perbandingan antara berat badan dan kuadrat dari tinggi badan (dalam $kg/m^2$).
4. **children**: jumlah anak yang ditanggung oleh penyedia asuransi kesehatan.
5. **smoker**: status merokok tertanggung asuransi (yes atau no).
6. **region**: daerah pemukiman tertanggung asuransi di US (southwest, southeast, northwest, dan northeast).
7. **charges**: besar biaya asuransi yang dibebankan kepada tertanggung asuransi (dalam USD).

### Deskrispsi Statistik Fitur Numerik
Berikut adalah hasil pengecekan deskripsi statistik untuk fitur numerik.

Tabel 1. Deskripsi Statistik Fitur Numerik

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
1. Count adalah jumlah sampel pada data.
2. Mean adalah nilai rata-rata.
3. Std adalah standar deviasi.
4. Min yaitu nilai minimum setiap kolom.
5. 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
6. 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
7. 75% adalah kuartil ketiga.
8. Max adalah nilai maksimum.

### Data Assesing
Pada tahap ini, dilakukan pengecekan terhadap beberapa aspek penting dalam dataset, yaitu:

- **Data Duplikat**: Memeriksa apakah terdapat data yang terduplikat atau memiliki nilai yang sama dengan data lainnya.
- **Missing Value**: Memeriksa apakah ada data yang hilang atau tidak tersedia pada fitur-fitur yang ada.
- **Outlier**: Mendeteksi data yang menyimpang jauh dari distribusi umum data lainnya (outlier).

Berdasarkan hasil Data Assesing yang telah dilakukan, diperoleh temuan sebagai berikut:

- Terdapat **1 record** yang terduplikat.
- **Tidak ada missing value** pada dataset.
- Fitur **'bmi'** dan **'charges'** terdeteksi mengandung outlier.

### Data Cleaning 
Berdasarkan temuan dari **Data Assesing**, langkah selanjutnya adalah melakukan **Data Cleaning**, yang meliputi pembersihan data dengan langkah-langkah sebagai berikut:
- **Penghapusan Record Duplikat**: Menghapus record yang terduplikat agar tidak mempengaruhi analisis dan model.
- **Penanganan Outlier**: Mengatasi outlier dengan menggunakan metode Interquartile Range (IQR). Rumus untuk IQR adalah:
$$IQR = Q_3 - Q_1,$$ 
dengan:
    - $IQR$ = *Inter Quartile Range*
    - $Q_3$ = Quartile 3
    - $Q_1$ = Quartile 1

Setelah dilakukan **penghapusan record duplikat**, dataset yang tersisa terdiri dari **1337 record**. Selanjutnya, setelah mengimplementasikan metode IQR untuk menangani outlier, dataset yang digunakan dalam tahap selanjutnya berjumlah **1192 record**.

### Analisis Univariat dan Analisis Multivariat
Data yang sudah dibersihkan selanjutnya bisa digunakan untuk analisis univariat dan analisis multivariat. 

#### Analisis Univariat 
Analisis Univariat bertujuan untuk menggali karakteristik setiap fitur secara terpisah, baik untuk fitur kategori maupun fitur numerik. Berikut adalah hasil analisis univariat yang dilakukan pada dataset.

a. Fitur Kategori

Diperhatikan Gambar 1a berikut ini.

![univariat_kategori_sex](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/5b44dca9-fcd2-4088-97ba-44e9824d0ba4)

![univariat_kategori_smoker](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/83dfda08-3b29-4240-97b6-9c7f2e6f020b)

![univariat_kategori_region](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/0902a5ce-516e-4a52-a8f2-e700c5f3132d)

Gambar 1a. Analisis Univariat (Fitur Kategori)

Berdasarkan Gambar 1a di atas, diperoleh bahwa:
1. **Jenis Kelamin**: Banyak tertanggung asuransi laki-laki dan perempuan hampir berimbang, yaitu 50.4% laki-laki dan 49.6% perempuan.
2. **Status Merokok**: Sebagian besar tertanggung asuransi bukan perokok, dengan 79.6% non-perokok dan sisanya, 20.6%, perokok.
4. **Region**: Banyak tertanggung asuransi yang tinggal pada tiap-tiap daerah pemukiman hampir sama, yaitu 26.9% di *southeast*, 26.9% di *southwest*, 24.4% di *northwes*t, dan 24.3% di *northeast*.

b. Fitur Numerik
   
Diperhatikan Gambar 1b berikut ini.

![univariat_numerik](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/9bba4270-6d0f-4d89-8faf-b80e099eec3f)

Gambar 1b. Analisis Univariat (Fitur Numerik)

Berdasarkan Gambar 1b di atas, diperoleh bahwa:
1. Peningkatan nilai 'children' dan 'charges' sebanding dengan penurunan jumlah sampel. Hal ini terlihat dari histrogram 'children' dan 'charges' yang grafiknya mengalami penurunan seiring semakin banyaknya jumlah sampel.
2. Distribusi 'charges', 'age', dan 'children' miring ke kanan, artinya lebih banyak data dengan nilai yang lebih rendah dibandingkan yang lebih tinggi. Di sisi lain, distribusi 'bmi' cenderung normal.
3. Rentang 'charges' cukup tinggi yaitu dari skala ratusan dolar amerika hingga sekitar \$60000.

Setelah dilakukan analisis univariat, selanjutnya dilakukan analisis multivariat.

#### Analisis Multivariat
Analisis Multivariat bertujuan untuk mengetahui hubungan antara fitur target charges dengan fitur-fitur lainnya, baik kategori maupun numerik. Berikut ini dilakukan analisis multivariat tersebut. 

a. Fitur Kategori

Pertama, dilakukan pengecekan rata-rata 'charges' terhadap fitur kategori untuk mengetahui pengaruh fitur kategori terhadap 'charges'. Diperhatikan Gambar 2a berikut.

![multivariat_kategori_sex](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/63deb315-f3ef-40f6-b5f2-25661284ef12)

![multivariat_kategori_smoker](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/f12ad28b-456a-419e-885b-68813b7f6e50)

![multivariat_kategori_region](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/e746debd-8f47-44a6-a863-69b335a2c2d5)

Gambar 2a. Analisis Multivariat (Fitur Kategori)

Berdasarkan Gambar 2a di atas, diperoleh bahwa:
1. **Sex**: rata-rata 'charges' yang dikenakan terhadap 'male' dan 'female' cenderung mirip. Rentangnya berada antara  \$12000  sampai \$15000 . Dengan demikian, fitur 'sex' mempunyai pengaruh kecil terhadap rata-rata 'charges'.
2. **Smoker**: rata-rata 'charges' untuk 'smoker' jauh lebih besar dibandingkan untuk 'non smoker'. Dengan demikian, fitur 'smoker' mempunyai pengaruh besar terhadap rata-rata 'charges'.
3. **Region**: rata-rata 'charges' yang dikenakan terhadap masing-masing region cenderung mirip. Rentangnya berada antara  \$12000  sampai \$16000 . Dengan demikian, fitur 'region' mempunyai pengaruh kecil terhadap rata-rata 'charges'.

**Kesimpulan**: Karena fitur sex dan region memiliki pengaruh kecil terhadap charges, kedua fitur ini dapat dihilangkan dari model untuk menyederhanakan analisis dan meningkatkan efisiensi model. Hanya fitur smoker yang tetap dipertahankan sebagai fitur kategori.

b. Fitur Numerik

Selanjutnya, dilakukan analisis multivariat terhadap fitur numerik sedemikian sehingga diperoleh Gambar 2b berikut. 

![multivariat_numerik](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/d57796c3-3023-4c0d-88aa-bdfc42943f9b)

Gambar 2b. Analisis Multivariat (Fitur Numerik)

Untuk membaca **Pairplot** di atas, perhatikan fitur target 'charges' pada sumbu y. Berdasarkan pairplot yang ditampilkan, analisis hubungan antar fitur numerik mengungkapkan hal-hal berikut:
- **Age** memiliki korelasi positif dengan charges, yang menunjukkan bahwa semakin tua usia seseorang, semakin tinggi kemungkinan charges yang diterima.
- **BMI** dan **children** tidak menunjukkan korelasi yang signifikan dengan charges dalam visualisasi ini.

Untuk memperdalam analisis korelasi antar fitur numerik, digunakan **correlation matrix** berikut.

![matriks korelasi](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/28070d73-7f63-4d2d-9116-c5b93695d183)

Gambar 3. Correlation Matrix untuk Fitur Numerik

Berdasarkan correlation matrix, dapat dilihat bahwa:
- Age memiliki korelasi positif yang cukup signifikan dengan charges (0.44).
- BMI memiliki korelasi yang sangat kecil dengan charges (-0.06).
Children memiliki korelasi yang sangat kecil dengan charges (0.08).
- BMI dan Children juga menunjukkan korelasi yang sangat kecil satu sama lain, dengan nilai korelasi yang hampir mendekati 0.

Selain charges, kita juga melakukan analisis korelasi antar fitur numerik lainnya, yaitu age, BMI, dan children.
- Age dan BMI memiliki korelasi 0.12, yang menunjukkan hubungan lemah positif antara usia dan indeks massa tubuh (BMI). Meskipun hubungan ini lemah, semakin tua usia seseorang mungkin sedikit berhubungan dengan tingginya BMI.
- Age dan Children memiliki korelasi 0.04, yang sangat kecil. Hal ini menunjukkan bahwa usia seseorang hampir tidak berhubungan dengan jumlah anak yang dimilikinya.
- BMI dan Children memiliki korelasi 0.01, yang juga sangat kecil. Ini menunjukkan bahwa BMI tidak memiliki pengaruh yang signifikan terhadap jumlah anak yang dimiliki.

Kesimpulan Korelasi Antar Fitur Numerik:
- Age memiliki korelasi yang cukup signifikan dengan charges, sehingga penting untuk dipertahankan dalam model.
- BMI dan children memiliki korelasi yang sangat kecil dengan charges, serta antara satu dengan yang lain, sehingga dipertimbangkan untuk dihilangkan guna menyederhanakan model.
- Age, BMI, dan children semuanya memiliki korelasi yang sangat kecil satu sama lain. Oleh karena itu, fitur-fitur ini dapat dianggap independen dalam hal interaksi antar mereka.

## Data Preparation 
Pada tahap **data preparation** atau persiapan data, dilakukan transformasi data agar sesuai untuk proses pemodelan. Terdapat tiga langkah utama yang dilakukan, yaitu:
1. **Encoding Fitur Kategori**
Encoding digunakan untuk mengubah fitur kategori menjadi fitur numerik, yang diperlukan oleh sebagian besar algoritma pembelajaran mesin. Pada dataset ini, encoding dilakukan pada fitur kategori 'sex' menggunakan teknik Label Encoding yang diterapkan melalui fungsi LabelEncoder dari library Scikit-learn. Dengan teknik ini, fitur 'sex' yang awalnya bernilai 'male' dan 'female' diubah menjadi angka numerik.
2. **Pembagian Dataset**
Sebelum membangun model, data perlu dibagi menjadi dua bagian utama: data latih (train) dan data uji (test). Pembagian ini dilakukan dengan menggunakan fungsi train_test_split dari library Scikit-learn. Proporsi umum yang digunakan dalam tahap ini adalah **80:20**, artinya 80% data digunakan untuk pelatihan model dan 20% sisanya digunakan untuk pengujian model. Berdasarkan proporsi tersebut, hasil pembagian dataset adalah sebagai berikut:
- Data latih terdiri dari **953 record**
- Data uji terdiri dari **239 record** 
3. **Scaling Fitur Numerik**
Scaling atau penyekalaan adalah langkah penting untuk mengubah nilai-nilai fitur numerik agar berada dalam rentang tertentu. Hal ini penting untuk memastikan bahwa model tidak terpengaruh oleh skala fitur yang berbeda. Salah satu teknik yang umum digunakan adalah **MinMaxScaler** dari library Scikit-learn, yang mentransformasi data sehingga berada dalam rentang 0 hingga 1. Proses scaling menggunakan rumus berikut:
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

Kedua model tersebut termasuk dalam kategori **ensemble models**, di mana prediksi dibuat dengan menggabungkan beberapa model secara bersama-sama. Ada dua pendekatan utama dalam teknik ensemble, yaitu **bagging dan boosting**.
- **Random Forest** adalah algoritma yang berbasis **bagging**, yang menggunakan beberapa pohon keputusan dan menggabungkan hasilnya untuk memprediksi output.
- **XGBoost** adalah algoritma berbasis **boosting**, yang membangun model secara bertahap, di mana setiap pohon keputusan yang baru berfokus pada kesalahan yang dibuat oleh pohon keputusan sebelumnya.

### Kelebihan dan Kekurangan Model
Algoritma Random Forest dan XGBoost mempunyai kelebihan dan kekurangan masing-masing. Kelebihan dan kekurangan algoritma **Random Forest**, yaitu:
1. Kelebihan 
   - **Mengurangi Varians & Meningkatkan Generalisasi**: Random Forest mengurangi overfitting dengan membagi data menjadi banyak subset dan membangun banyak pohon. Hal ini membantu model untuk lebih generalisasi terhadap data baru. Ini merupakan karakteristik dari teknik bagging yang digunakan dalam Random Forest, di mana model mencoba untuk meminimalkan varians dan meningkatkan stabilitas model secara keseluruhan. Bagging mengurangi varians model dengan menggabungkan banyak pohon keputusan yang berbeda [[6](https://link.springer.com/article/10.1023/A:1010933404324)].
    - **Cenderung Stabil dan Bekerja dengan Data Kecil**: Karena menggunakan banyak pohon, Random Forest cenderung lebih stabil dan dapat bekerja dengan baik pada data kecil. Hal ini membantu menghindari overfitting yang sering terjadi pada model yang lebih kompleks [[7](https://ieeexplore.ieee.org/document/7034345)].
    - **Bekerja dengan Data Tidak Seimbang**: Random Forest dapat bekerja dengan baik pada data yang tidak seimbang dengan memberi bobot pada setiap kelas berdasarkan distribusi data. Ini membuatnya cocok untuk dataset yang mungkin memiliki distribusi kelas yang sangat tidak merata [[6](https://link.springer.com/article/10.1023/A:1010933404324)].
2. Kekurangan
    - **Bias terhadap Data Kategori dengan Banyak Kategori**: Random Forest bisa memilih fitur dengan banyak kategori sebagai pembagi, yang dapat menyebabkan ketidakseimbangan dalam pembagian data. Hal ini menjadi tantangan, terutama dalam kasus dengan variabel kategori yang memiliki banyak level [[8](https://www.researchgate.net/publication/239158927_An_overview_of_Random_Forest)].
    - **Waktu Komputasi pada Dataset Besar**: Random Forest memerlukan waktu komputasi yang lebih lama ketika digunakan pada dataset yang besar karena banyaknya pohon yang harus dibangun dan diuji [[7](https://ieeexplore.ieee.org/document/7034345)].

Di sisi lain, kelebihan dan kekurangan algoritma **XGBoost**, yaitu:
1. Kelebihan 
    - **Mengurangi Bias & Underfitting**: XGBoost mengurangi bias dan underfitting melalui teknik boosting, yang berfokus pada perbaikan kesalahan dari model sebelumnya. Ini membantu meningkatkan akurasi prediksi, terutama pada dataset yang lebih kompleks [[6](https://link.springer.com/article/10.1023/A:1010933404324)].
    - **Pelatihan Model yang Efisien dan Skalabel**: XGBoost dioptimalkan untuk kecepatan dan penggunaan memori yang efisien, membuatnya cocok untuk bekerja dengan data besar. Algoritma ini dirancang untuk melatih model dengan lebih cepat dan pada skala yang lebih besar dibandingkan banyak algoritma lain [[9](https://dl.acm.org/doi/10.1145/2939672.2939785)].
    - **Menangani Missing Values**: XGBoost secara otomatis menangani missing values, menghindari kesalahan prediksi pada data yang hilang dan meningkatkan fleksibilitas dalam menangani data yang tidak lengkap [[9](https://dl.acm.org/doi/10.1145/2939672.2939785)].
2. Kekurangan
    - **Rentan terhadap Overfitting**: XGBoost cenderung overfit jika tidak dikendalikan dengan baik, terutama saat model terus diperbaiki dengan menambahkan lebih banyak pohon tanpa mempertimbangkan parameter regularisasi yang tepat. Oleh karena itu, kontrol hyperparameter yang ketat diperlukan untuk menghindari overfitting [[10](https://www.researchgate.net/publication/311102752_A_Survey_on_Overfitting_and_Model_Complexity)].
    - **Memerlukan Banyak Ruang Penyimpanan dan Komputasi yang Tinggi**: XGBoost membutuhkan ruang penyimpanan yang cukup besar dan konsumsi memori yang tinggi, terutama ketika bekerja dengan data yang sangat besar. Ini bisa menjadi masalah untuk sistem yang memiliki keterbatasan sumber daya komputasi [[9](https://dl.acm.org/doi/10.1145/2939672.2939785)].

## Peningkatan Performa Model
Performa model yang dibangun dengan algoritma **Random Forest** dan **XGBoost** dioptimalkan dengan menerapkan **hyperparameter tuning** pada baseline model Random Forest dan XGBoost sedemikian sehingga terdapat **empat model** machine learning yang dikembangkan, yaitu::
1. RF1: Random Forest
2. RF2: Random Forest dengan Hyperparamter Tuning
3. XGB1: XGBoost 
4. XGB2: XGBoost dengan Hyperparameter Tuning

Hyperparameter dari algoritma Random Forest yang di-tuning, yaitu:
1. max_depth: kedalaman atau panjang pohon yang berarti ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
2. min_samples_leaf: jumlah minimum sampel yang diperlukan untuk berada di leafs (daun).
3. mis_samples split: jumlah minimum sampel yang diperlukan untuk membagi node internal.
4. n_estimator: jumlah trees (pohon) di forest.

Di sisi lain, hyperparameter dari algoritma XGBoost yang di-tuning, yaitu:
1. n_estimator: jumlah trees (pohon) di forest.
2. learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
2. max_depth: kedalaman atau panjang pohon, yang berarti ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.

Hyperparameter tuning dilakukan dengan menggunakan fungsi **GridSearchCV** dari library ScikitLearn sedemikian sehingga diperoleh nilai hyparameter terbaik, yaitu:
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
 Nilai hyperparameter tersebut yang digunakan untuk mengembangkan model RF2 dan XGB2.

## Evaluation
Model machine learning yang dibangun di atas merupakan model regresi, akibatnya untuk mengevaluasi performa model tersebut bisa digunakan tiga metrik evaluasi, yaitu:
1. MSE (Mean Squarred Error).
2. MAE (Mean Absolute Error).
3. $R^2$ (R-Squarred).

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
- **MSE** digunakan untuk menghitung rata-rata dari selisih kuadrat antara nilai hasil prediksi dan nilai data aktual. Semakin kecil nilai MSE, maka semakin baik kualitas model tersebut.
- **MAE** digunakan untuk menghitung selisih absolut antara nilai data hasil prediksi dan nilai data aktual. Semakin kecil nilai MAE, maka semakin baik kualitas model tersebut.
- **R²** digunakan untuk mengetahui seberapa besar pengaruh variabel independen tertentu terhadap variabel dependen, mengukur seberapa baik model dapat menggambarkan variasi data yang ada.

Selanjutnya, dibandingkan nilai ketiga metrik evaluasi dari masing-masing model. Nilai metrik evaluasi **MSE** terhadap **data latih (train_mse) dan data uji (test_mse)** dibandingkan menggunakan Barplot berikut ini.

![perbandingan_mse](https://github.com/rzknra/idcamp_mlt_predictive_analytics/assets/94267677/0864012e-b013-44ff-bf9a-0c43f9d6a68d)

Gambar 4. Perbandingan Nilai MSE terhadap Data Latih dan Data Uji

Berdasarkan Gambar 4 (Barplot) tersebut diperoleh bahwa:
- Model RF2 dan XGB2 mempunyai nilai 'test_mse' yang lebih rendah dibandingkan kedua model lainnya.
- Selisih nilai 'test_mse' dan 'train_mse' pada model RF2 dan XGB2 lebih kecil dibandingkan dua model lainnya, yang berarti kedua model tidak terlalu overfitting.
  
Dengan demikian, bedasarkan perbandingan nilai metrik evaluasi MSE terhadap data latih (train_mse) dan data uji (test_mse) tersebut diperoleh **model RF2 dan XGB2 lebih baik dari kedua model lainnya**.

Lebih lanjut, dibandingkan nilai ketiga metrik evaluasi yaitu nilai MSE terhadap data uji (test_MSE), MAE, dan $R^2$ dari setiap model *machine learning* yang sudah dibangun menggunakan Tabel 2 berikut.

Tabel 2. Nilai Metrik Evaluasi Setiap Model
|     | MSE | MAE | $R^2$ | 
|-----|-----|-----|-----|
| RF1 | 25282.763885 | 2922.9584113729475 | 0.5660674252523771 |
| RF2 | 24324.410399 | 2897.073605878178 | 0.5825158166395445 |
| XGB1 | 25967.960248 | 2948.621166460513 | 0.5543072781685672 |
| XGB2 | 23633.954472 | 2848.303577468292 | 0.594366234561500 |

Berdasarkan Tabel 2 di atas, diperoleh bahwa:
- Model RF2 dan XGB2 mempunyai nilai MSE dan MAE terhadap data uji (test_mse) yang lebih rendah dibandingkan kedua model lainnya.
- Nilai $R^2$ model RF2 dan XGB2 yang lebih tinggi dibandingkan dua model lainnya, yang menunjukkan bahwa kedua model tersebut lebih baik dalam menggambarkan variabilitas data.

Dengan demikian, berdasarkan perbandingan ketiga nilai metrik evaluasi tersebut diperoleh **model RF2 dan XGB2 lebih baik daripada kedua model lainnya**. 

Oleh karena itu, berdasarkan hasil perbandingan nilai metrik evaluasi MSE (train_MSE dan test_MSE), MAE, dan $R^2$ menggunakan Gambar 4 (*barplot*) dan Tabel 2 di atas, diperoleh bahwa **model RF2 (Random Forest with Hyperparameter Tuning) dan XGB2 (XGBoost with Hyperparameter Tuning) adalah model terbaik untuk prediksi biaya asuransi kesehatan**.

Berikut ini berikan hasil pengujian masing-masing model dengan menggunakan salah satu record data dari data uji.

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

[6] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. 

[7] Zhang, Z., & Yang, L. (2015). Random Forest Algorithm for Classification. Proceedings of the International Conference on Computational Intelligence and Communications Networks, 174-179. 

[8] Ramaswamy, S., & Ramakrishnan, R. (2010). An overview of Random Forest. ResearchGate. Link

[9] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. 

[10] Li, L., & Li, W. (2017). A survey on overfitting and model complexity. International Journal of Computer Science and Information Technology. 
