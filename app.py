import streamlit as st
import pickle
import numpy as np

# Load model
with open('XGBRegressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Label unik dari dataset
manufacturer_labels = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
 'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN',
 'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA',
 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT', 'INFINITI',
 'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'YAZ', 'GAZ',
 'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR',
 'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC',
 'PEUGEOT', 'BENTLEY', 'VOLVO', 'БОГДАН', 'HAVAL', 'HUMMER', 'SCION',
 'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
 'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE',
 'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']

color_labels = ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red',
 'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige',
 'Carnelian red', 'Purple', 'Pink']

category_labels = ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon',
 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup']

prod_years = sorted([2010, 2011, 2006, 2014, 2016, 2013, 2007, 1999, 1997, 2018, 2008,
 2012, 2017, 2001, 1995, 2009, 2000, 2019, 2015, 2004, 1998, 1990,
 2005, 2003, 1985, 1996, 2002, 1993, 1992, 1988, 1977, 1989, 1994,
 2020, 1984, 1986, 1991, 1983, 1953, 1964, 1974, 1987, 1943, 1978,
 1965, 1976, 1957, 1980, 1939, 1968, 1947, 1982, 1981, 1973])

# Daftar model khusus HYUNDAI
hyundai_models = ['Santa FE', 'Sonata', 'Elantra', 'H1', 'Tucson', 'Genesis',
 'Elantra sport limited', 'I30', 'Veloster', 'Sonata SPORT', 'Elantra SE',
 'Accent', 'Grandeur', 'kona', 'IX35', 'Elantra limited', 'Sonata 2.0t',
 'Sonata S', 'Sonata blue edition', 'Sonata hybrid', 'Elantra LIMITED',
 'Elantra GT', 'Sonata HYBRID', 'Santa FE Ultimate', 'Sonata Hibrid', 'Getz',
 'Elantra gt', 'Elantra Limited', 'Elantra GLS / LIMITED', 'Tucson TURBO',
 'Ioniq', 'Sonata LPG', 'IX35 2.0', 'Veloster R-spec', 'Azera', 'Sonata Hybrid',
 'Elantra Gt', 'Tucson Limited', 'Elantra Se', 'Elantra 2014', 'Lantra',
 'Sonata Limited', 'Tucson Se', 'Sonata 2.4L', 'Lantra LIMITED',
 'Veloster Turbo', 'Sonata Sport', 'Santa FE long', 'Tucson SE', 'H1 grandstarex',
 'H1 GRAND STAREX', 'Accent SE', 'Elantra 2016', 'Veloster TURBO', 'Veloster remix',
 'Elantra LIMITEDI', 'Elantra i30', 'Santa FE sport', 'Sonata sport',
 'Sonata SE LIMITED', 'H1 starixs', 'Accent GS', 'Sonata LIMITED', 'Elantra GS',
 'i20', 'i40', 'Sonata SE', 'Veracruz', 'Galloper', 'Sonata ၏မြန်မာနိုင်ငံ',
 'Elantra se']

# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas")
st.markdown("Masukkan fitur-fitur mobil untuk memprediksi harga jual menggunakan model XGBoost.")

# Input numerik
Levy = st.number_input("Levy", value=0.0)
Prod_year = st.selectbox("Tahun Produksi", prod_years)
Engine_volume = st.number_input("Kapasitas Mesin (L)", value=1.5)
Mileage = st.number_input("Jarak Tempuh (km)", value=50000)
Cylinders = st.number_input("Jumlah Silinder", value=4)
Airbags = st.number_input("Jumlah Airbags", value=2)
Age_of_Car = st.number_input("Umur Mobil (tahun)", value=5)

# Dropdown
Manufacturer = st.selectbox("Manufacturer", manufacturer_labels)
Manufacturer_idx = manufacturer_labels.index(Manufacturer)

# Pilihan Model berdasarkan Manufacturer
if Manufacturer == "HYUNDAI":
    selected_model = st.selectbox("Model", sorted(hyundai_models))
    Model = hyundai_models.index(selected_model)  # gunakan indeks model Hyundai
else:
    Model = st.number_input("Model (kode numerik)", value=1)

Category = st.selectbox("Kategori Mobil", category_labels)
Category_idx = category_labels.index(Category)

Color = st.selectbox("Warna Mobil", color_labels)
Color_idx = color_labels.index(Color)

# Fitur lainnya
Leather_interior = st.selectbox("Interior Kulit", ["Tidak", "Ya"])
Fuel_type = st.number_input("Tipe Bahan Bakar (kode numerik)", value=1)
Gear_box_type = st.number_input("Tipe Gear Box (kode numerik)", value=1)
Drive_wheels = st.number_input("Drive Wheels (kode numerik)", value=1)
is_turbo = st.selectbox("Apakah Turbo?", ["Tidak", "Ya"])

# Konversi boolean ke numerik
Leather_interior = 1 if Leather_interior == "Ya" else 0
is_turbo = 1 if is_turbo == "Ya" else 0

# Prediksi
if st.button("Prediksi Harga"):
    data = np.array([[Levy, Prod_year, Engine_volume, Mileage, Cylinders,
                      Airbags, Age_of_Car, Manufacturer_idx, Model, Category_idx,
                      Leather_interior, Fuel_type, Gear_box_type, Drive_wheels,
                      Color_idx, is_turbo]])
    prediksi = model.predict(data)[0]
    st.success(f"Perkiraan Harga Mobil Bekas: {prediksi:,.2f}")
