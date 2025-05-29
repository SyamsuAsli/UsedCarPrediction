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

# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas")
st.markdown("Masukkan fitur-fitur mobil untuk memprediksi harga jual menggunakan model XGBoost.")

# Input numerik
Levy = st.number_input("Levy", value=0.0)
Prod_year = st.number_input("Tahun Produksi", value=2010)
Engine_volume = st.number_input("Kapasitas Mesin (L)", value=1.5)
Mileage = st.number_input("Jarak Tempuh (km)", value=50000)
Cylinders = st.number_input("Jumlah Silinder", value=4)
Airbags = st.number_input("Jumlah Airbags", value=2)
Age_of_Car = st.number_input("Umur Mobil (tahun)", value=5)

# Dropdown
Manufacturer = st.selectbox("Manufacturer", manufacturer_labels)
Manufacturer_idx = manufacturer_labels.index(Manufacturer)

Category = st.selectbox("Kategori Mobil", category_labels)
Category_idx = category_labels.index(Category)

Color = st.selectbox("Warna Mobil", color_labels)
Color_idx = color_labels.index(Color)

# Fitur lainnya
Model = st.number_input("Model (kode numerik)", value=1)
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
