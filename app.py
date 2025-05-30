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

prod_years = sorted([2010, 2011, 2006, 2014, 2016, 2013, 2007, 1999, 1997, 2018, 2008,
 2012, 2017, 2001, 1995, 2009, 2000, 2019, 2015, 2004, 1998, 1990,
 2005, 2003, 1985, 1996, 2002, 1993, 1992, 1988, 1977, 1989, 1994,
 2020, 1984, 1986, 1991, 1983, 1953, 1964, 1974, 1987, 1943, 1978,
 1965, 1976, 1957, 1980, 1939, 1968, 1947, 1982, 1981, 1973])

cylinders_labels = sorted([
    6,  4,  8,  1, 12,  3,  2, 16,  5,  7,  9, 10, 14
])

Engine_volume_labels = sorted([ 3.5,  3. ,  1.3,  2.5,  2. ,  1.8,  2.4,  3.3,  1.6,  2.2,  4.7,
        1.5,  4.4,  1.4,  3.6,  4. ,  2.3,  5.5,  3.2,  3.8,  4.6,  1.2,
        5. ,  1.7,  2.9,  0.5,  1.9,  2.7,  4.8,  5.3,  0.4,  2.8,  2.1,
        0.7,  5.4,  3.7,  1. ,  2.6,  0.8,  0.2,  5.7,  6.7,  6.2,  3.4,
        6.3,  4.2,  0. , 20. ,  1.1,  5.6,  6. ,  0.6,  6.8,  4.5,  7.3,
        0.1,  4.3,  6.4,  3.9,  5.9,  0.3,  5.2,  5.8])

airbags_labels =sorted([12,  8,  2,  0,  4,  6, 10,  3,  1, 16,  7,  9,  5, 11, 14, 15, 13])

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

# Daftar model khusus LEXUS
lexus_models = ['RX 450', 'RX 350', 'RX 400', 'GX 470', 'GX 460', 'NX 300',
 'CT 200h', 'GS 350', 'NX 200', 'RX 300', 'RX 400 HYBRID',
 'HS 250h Hybrid', 'ES 350', 'IS 200', 'ES 300', 'IS 250', 'LS 460',
 'IS 350', 'HS 250h', 'CT 200h F-sport', 'LX 570', 'LX 470',
 'RX 350 F sport', 'RX 450 H', 'CT 200h F SPORT', 'RC F',
 'IS 250 ᲒᲐᲛᲝᲪᲓᲘᲚᲘ', 'RX 400 RESTAILING', 'GX 470 470', 'IS 300',
 'GS 300', 'IS 250 TURBO', 'IS-F', 'GS 450', 'RX 450 HYBRID',
 'GX 470 SUV 4D (4.7L V8 S)', 'RC F F SPORT', 'ES 300 hybrid',
 'IS 350 C', 'CT 200h 1.8', 'RX 400 hybrid', 'CT 200h F sport',
 'RX 400 H', 'RX 450 F SPORT']

chevrolet_models = ['Equinox', 'Cruze LT', 'Captiva', 'Cruze', 'Orlando', 'Volt',
       'Avalanche', 'Malibu', 'Lacetti', 'Aveo', 'Matiz', 'Spark',
       'Impala', 'Cruze ltz', 'Cruze LTZ', 'Camaro', 'Cruze strocna',
       'Volt premier', 'Traverse', 'Cruze Premier', '1500', 'Equinox LT',
       'Cruze RS', 'Sonic', 'Cruze LS', 'Trailblazer', 'Cruze sonic',
       'Nubira', 'Cruze L T', 'Malibu LT', 'Malibu eco', 'HHR',
       'Suburban', 'Cruze Cruze', 'Camaro LS', 'Silverado',
       'Malibu Hybrid', 'Trax', 'Volt Full Packet', 'Volt PREMIER',
       'Sonic LT', 'Corvette', 'Cruze PREMIER', 'Niva', 'Volt Premier',
       'Camaro RS', 'Colorado', 'Cruze LT RS', 'Kalos', 'Cruze S']

honda_models = ['FIT', 'Civic', 'Cr-v', 'Hr-v EX', 'Insight', 'Stream', 'Shuttle',
       'Element', 'Odyssey', 'Insight EX', 'Hr-v', 'Accord', 'FIT Sport',
       'Step Wagon Pada', 'FIT SPORT', 'Civic EX', 'FIT Hbrid',
       'Passport', 'Fit Aria', 'FIT S', 'Inspire', 'Fred', 'FIT HIBRID',
       'Pilot', 'Elysion', 'Accord CL9 type S', 'FIT HYBRYD', 'FIT fit',
       'Edix', 'Elysion 3.0', 'Civic Ferio', 'CR-Z', 'FIT Premiym',
       'Cr-v Cr-v', 'FIT RS MODELI', 'CR-Z ჰიბრიდი', 'FIT RS',
       'Fred HIBRIDI', 'FIT RS MUGEN', 'CRX', 'Step Wagon', 'Edix FR-v',
       'FIT Modulo', 'FIT GP-5', 'FIT "S"- PAKETI.', 'Cr-v LX', 'FIT ex',
       'Crossroad', '400X', 'Insight LX', 'FIT GP-6', 'Hr-v EXL',
       'FIT Hybrid', 'FIT PREMIUMI', 'Crosstour', 'Legend FULL',
       'FIT PREMIUM PAKETI', 'Integra', 'Step Wagon RG2 SPADA',
       'FIT NAVI PREMIUM', 'Civic Hybrid', 'Ridgeline', 'FIT LX']

ford_models = ['Escape', 'Transit', 'Escape Hybrid', 'Fusion', 'Mustang',
       'Focus SE', 'Explorer', 'C-MAX', 'Fusion Titanium', 'Taurus',
       'Galaxy', 'Fiesta', 'Fiesta 1.6', 'Focus', 'Fusion TITANIUM',
       'Transit Connect', 'Tourneo Connect', 'Transit 350T',
       'Transit Connect ბენზინი', 'Edge', 'Mondeo', 'Escort', 'Sierra',
       'Fusion phev', 'Escape Titanium', 'Ranger', 'Transit CL',
       'Escape 3.0', 'Transit Fff', 'Transit S', 'C-MAX HYBRID',
       'Transit 135', 'Expedition', 'Taurus interceptor',
       'Focus TITANIUM', 'F150', 'KA', 'Focus ST', 'Maverick', 'S-max',
       'C-MAX SEL', 'Fusion SE', 'B-MAX', 'Focus Fokusi',
       'Transit Custom', 'Focus Flexfuel', 'Transit Connect Prastoi',
       'Transit T330', 'Escape SE', 'Escape მერკური მერინერი',
       'EcoSport SE', 'Focus Titanium', 'Cougar', 'C-MAX C-MAX',
       'Focus se', 'C-MAX PREMIUM', 'Mustang cabrio', 'Transit Tourneo',
       'C-MAX SE', 'Transit 100LD', 'Transit პერეგაროტკა',
       'Explorer Turbo japan', 'Fusion Bybrid', 'Fusion 2015',
       'Fusion HIBRID', 'Fiesta SE', 'Explorer XLT', 'Fusion 1.6',
       'Transit ford', 'Escape HYBRID', 'EcoSport', 'Mustang ecoboost',
       'Scorpio', 'Fusion hybrid', 'Courier', 'Transit 2.4',
       'Fusion HYBRID', 'Sierra DIZEL', 'Fusion Hybrid', 'Taurus X',
       'Ranger Wildtrak', 'Escape სასწრაფოდ', 'Escape escape',
       'Focus SEL', 'Fusion HYBRID SE']

fuel_type_labels=['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG',
       'Hydrogen']

gear_box_labels=['Automatic', 'Tiptronic', 'Variator', 'Manual']

drive_wheels_labels=['4x4', 'Front', 'Rear']

# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas")
st.markdown("Masukkan fitur-fitur mobil untuk memprediksi harga jual menggunakan model XGBoost.")

# Input numerik
Levy = st.number_input("Levy", value=0.0)
Prod_year = st.selectbox("Tahun Produksi", prod_years)
Engine_volume = st.number_input("Kapasitas Mesin (L)", Engine_volume_labels)
Mileage = st.number_input("Jarak Tempuh (km)", value=50000)
Cylinders = st.number_input("Jumlah Silinder", cylinders_labels)
Airbags = st.number_input("Jumlah Airbags", airbags_labels)

# Dropdown
Manufacturer = st.selectbox("Manufacturer", manufacturer_labels)
Manufacturer_idx = manufacturer_labels.index(Manufacturer)

# Pilihan Model berdasarkan Manufacturer
if Manufacturer == "HYUNDAI":
    selected_model = st.selectbox("Model", sorted(hyundai_models))
    Model = hyundai_models.index(selected_model)
elif Manufacturer == "LEXUS":
    selected_model = st.selectbox("Model", sorted(lexus_models))
    Model = lexus_models.index(selected_model)
elif Manufacturer == "CHEVROLET":
    selected_model = st.selectbox("Model", sorted(chevrolet_models))
    Model = chevrolet_models.index(selected_model)
elif Manufacturer == "HONDA":
    selected_model = st.selectbox("Model", sorted(honda_models))
    Model = honda_models.index(selected_model)
elif Manufacturer == "FORD":
    selected_model = st.selectbox("Model", sorted(ford_models))
    Model = ford_models.index(selected_model)
else:
    Model = st.number_input("Model (kode numerik)", value=1)


Color = st.selectbox("Warna Mobil", color_labels)
Color_idx = color_labels.index(Color)

Fuel_type=st.selectbox("Tipe Bahan Bakar",fuel_type_labels)
Fuel_type_idx=fuel_type_labels.index(Fuel_type)

Gear_box_type =st.selectbox("Tipe Gear Box",gear_box_labels)
Gear_box_type_idx=gear_box_labels.index(Gear_box_type)

Drive_wheels =st.selectbox("Tipe Gear Box",drive_wheels_labels)
Drive_wheels_idx=drive_wheels_labels.index(Drive_wheels)

# Fitur lainnya
Leather_interior = st.selectbox("Interior Kulit", ["Tidak", "Ya"])
is_turbo = st.selectbox("Apakah Turbo?", ["Tidak", "Ya"])

# Konversi boolean ke numerik
Leather_interior = 1 if Leather_interior == "Ya" else 0
is_turbo = 1 if is_turbo == "Ya" else 0

# Prediksi
if st.button("Prediksi Harga"):
    data = np.array([[Levy, Prod_year, Engine_volume, Mileage, Cylinders,
                      Airbags, Manufacturer_idx, Model,
                      Leather_interior, Fuel_type_idx, Gear_box_type_idx, Drive_wheels,
                      Color_idx, is_turbo]])
    prediksi = model.predict(data)[0]
    st.success(f"Perkiraan Harga Mobil Bekas: {prediksi:,.2f}")
