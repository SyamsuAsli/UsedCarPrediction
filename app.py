import streamlit as st
import pickle
import numpy as np
import joblib
    
# Load model
model = joblib.load('RandomForestRegresso_compressed.pkl')

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

cylinders_labels = sorted([
    6,  4,  8,  1, 12,  3,  2, 16,  5,  7,  9, 10, 14
])

Engine_vol_labels = sorted([ 3.5,  3. ,  1.3,  2.5,  2. ,  1.8,  2.4,  3.3,  1.6,  2.2,  4.7,
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

toyota_models=['Prius', 'Camry', 'CHR', 'Highlander', 'CHR Limited', 'Tacoma',
       'Prius C', 'Aqua', 'VOXY', 'Vitz', 'Yaris', 'RAV 4', 'Sienna',
       'Vitz funkargo', 'Camry Se', 'RAV 4 XLE Sport', 'Sienta',
       'Avalon LIMITED', 'Ist', 'Corolla', 'Tundra', 'RAV 4 Le', 'Avalon',
       'Camry SE', 'RAV 4 s p o r t', 'Aqua S', 'Land Cruiser Prado',
       'Corolla IM', 'Corolla verso', 'Auris', 'FJ Cruiser', 'Ipsum',
       'Corolla 04', 'Prius 2014', 'Aqua g soft leather sele', 'Prius V',
       'Passo', 'ISIS', 'Ist 1.5', 'Camry se', 'Sequoia', 'Corolla LE',
       'Camry S', 'Ipsum S', 'Prius plugin', 'Verso', 'Alphard',
       'Camry XLE', 'Venza', 'Corolla S', 'Altezza', 'Camry SPORT',
       'Hilux', 'Camry sport', 'Land Cruiser', 'Caldina', 'Aqua s',
       'Prius C Navigation', 'Prius V ALPINA', 'Highlander sport',
       'Aqua L paketi', 'Avensis', 'Prius s', 'Will Vs', 'Prius BLUG-IN',
       'Yaris IA', 'Estima', 'Camry sporti', 'Camry HYBRID', 'Camry LE',
       'Fortuner', 'Avalon limited', 'Wish', 'Vitz RS', 'Century',
       'Fun Cargo', 'Aqua G klas', '4Runner', 'Corolla Im',
       'Highlander 2.4 lit', 'Camry Le', 'Camry Hybrid', 'Prius ფლაგინი',
       'Prius 1.5I', 'Mark X', 'Yaris iA', 'Aqua G', 'Camry sport se',
       'Highlander LIMITED', 'Camry XV50', 'Prius ჰიბრიდი', 'Aqua HIBRID',
       'Celica', 'RAV 4 SPORT', 'Prius TSS LIMITED', 'Prius S',
       'Vitz i.ll', 'RAV 4 XLE', 'Prius C 2013', 'Corolla ECO', 'Hiace',
       '4Runner LIMITED', 'Prius V HYBRID', 'RAV 4 LIMITED', 'Camry sel',
       'Belta', 'Prius 9', 'Grand HIACE', 'Yaris RS', 'Prius prius',
       'Camry XLEi', 'Prius 11', 'Land Cruiser 105', 'Land Cruiser 100',
       'Matrix XR', 'RAV 4 se', 'Prius plug-in', 'RAV 4 Dizel',
       'Prius personna', 'Prius C ჰიბრიდი', 'Noah', 'Land Cruiser 11',
       'Camry SPORT PAKET', 'Will Chypa', 'Sai', 'VOXY 2003', 'Ractis',
       'Aqua სასწრაფოდ', 'Sienta LE', 'Avalon Limited', 'Highlander 2,4',
       'Prius C 80 original', 'Land Cruiser 200', 'Highlander limited',
       'Prius Plug IN', 'Mark X Zio', 'Aqua sport', 'Yaris SE',
       'Highlander XLE', 'Prius 1.8', 'Corolla se', 'Prius C hybrid',
       'Camry XSE', 'Prius plagin', 'Prius 3', 'RAV 4 L', 'BB',
       'Camry ჰიბრიდი', 'Camry SE HIBRYD', 'Prius Plug in',
       'Tacoma TRD Off Road', 'Land Cruiser PRADO', 'Corolla 140',
       'RAV 4 SUPER!!!', 'Prius C Hybrid', 'Cami', 'Prius C YARIS IA',
       'Prius C 1.5I', 'Prius V HIBRID', 'Prius C aqua']

marcedes_benz_models= ['E 350', 'E 220', 'Vito', 'E 300', 'C 180', 'GLA 250', 'A 160',
       'ML 350', 'GL 63 AMG', 'E 320', 'Sprinter', 'E 500 AMG', 'A 170',
       'C 250', 'Vaneo', 'CLS 550', 'C 300', 'C 350', 'CLS 500', 'S 350',
       'C 200', 'E 350 ამგ', 'CLK 320', 'ML 250', 'GLE 350', 'GLC 300',
       'C 240', 'C 200 2.0', 'ML 550 4.7', 'CLK 240', 'E 270',
       'S 350 W2222', 'A 190', 'E 250', 'GL 320', 'CLK 320 AMG', 'S 550',
       'Sprinter 411', 'E 500', 'S 550 ჰიბრიდი', 'CLS 350', 'G 350',
       'CLA 250', 'G 65 AMG 63AMG', 'GL 550', 'Smart', 'Sprinter MAX',
       'A 140', 'E 280', 'GL 450', 'B 170', 'CLS 55 AMG', 'Sprinter VAN',
       'Viano', 'CLK 200', 'E 550', 'CLK 230 .', 'R 350', 'E 240',
       'ML 350 4 MATIC', 'CLK 55 AMG', 'C 220', 'E 200', 'C 230', 'S 500',
       'C 320 CDI', 'C 200 7G-TRONIC', 'X 250', 'ML 280 სასწრაფოდ', '320',
       'CLA 45 AMG', 'GLE 63 AMG', 'Smart Fortwo', 'CLS 550 550',
       'E 270 AVANGARDI', 'E 350 212', 'E 55', 'S 63 AMG', 'C 63 AMG',
       'ML 500', 'G 55 AMG', 'ML 270', 'CLK 200 Kompressor',
       'CLS 350 AMG', 'CLK 270', 'E 350 w211', 'ML 320',
       'Sprinter 315CDI', 'Sprinter 311', 'E 280 CDI', 'E 36 AMG',
       'Vito 2.2', '416', 'E 300 AVANTGARDE-LTD', 'R 320', 'CL 550',
       'E 350 AMG', 'C 220 CDI', 'GLS 450', 'S 320',
       'E 350 4 Matic AMG Packag', 'GL 320 bluetec', 'C 320', 'E 220 cdi',
       'GL 350', 'CLS 63 AMG', 'R 350 BLUETEC', 'S 350 Longia',
       'C 230 2.5', 'Sprinter Maxi-ს Max', 'ML 350 BLUETEC', 'CLK 430',
       'ML 270 CDI', 'C 200 KOMPRESSOR', '300', 'ML 280', 'GLE 400 A M G',
       'E 270 CDI', 'CLK 230', 'E 50', 'X 250 ევროპული', 'ML 350 3.7',
       'C 400', 'Sprinter სატვირთო', 'GLK 300', '270', 'C 250 luxury',
       'E 320 4×4', 'GLC 300 GLC coupe', 'G 550', 'GLK 350',
       'C 300 4matic', 'C 250 1.8 ტურბო', 'E 400', 'ML 350 4matic',
       'GL 350 BLUETEC', 'G 65 AMG G63 AMG', 'E 350 4 MATIC',
       'Sprinter VIP CLASS', 'CLK 280', 'C 230 kompresor', 'S 430',
       'Citan', 'E 320 4matic', 'S 55 5.5', 'GLE 450', 'E 270 4', '250',
       'ML 320 cdi', 'GL 350 BLUTEC', 'GL 450 3.0', 'C 250 A.M.G',
       'S 400', 'C 250 AMG', 'C 180 komp', 'CL 500', 'AMG GT S',
       'Sprinter 313CDI', 'ML 350 370', 'E 320 bluetec', 'Vito 115 CDI',
       '220', 'ML 350 ML350', 'A 200', 'CLS 450 CLS 400',
       'GL 350 Bluetec', 'G 63 AMG', 'ML 350 sport', 'SLK 32 AMG',
       'C 32 AMG', 'S 550 LONG', 'Sprinter 316', 'Sprinter 314',
       'CLS 350 JAPAN', 'C 320 AMG', 'E 220 W210...CDI', 'GLK 250',
       'Sprinter 316 CDI', 'ML 63 AMG', 'ML 55 AMG', 'Sprinter 313',
       'Vito long115', 'GL 350 დიზელი', 'Vito 113', 'A 170 CDI',
       'E 220 CDI', 'E 280 3.0', 'E 350 4 matic', 'C 270', 'Vito 111 CDI',
       'ML 500 AMG', 'S 350 CDI 320', '230', 'GLE 400 Coupe, AMG Kit',
       'ML 300', 'B 200', 'C 240 w203', 'CLA 250 AMG', 'Sprinter 516',
       'CLK 320 avangarde', 'GLC 250', 'GLS 63 AMG', 'Sprinter 315CDI-XL',
       'GL 500', 'GLA 200', 'B 200 Turbo', 'C 300 sport', 'C 280',
       'Vito 111', 'Sprinter 308 CDI', 'Sprinter EURO4', 'Viano Ambiente',
       'B 180', 'C 250 1,8 turbo', 'C 200 Kompressor', 'Vito 115',
       'GLE 400', 'C 250 AMG-PAKET-1,8', 'B 170 B Class', 'E 220 211',
       'SL 55 AMG', 'C 250 1.8', 'E 220 ELEGANCE',
       'ML 350 SPECIAL EDITION', 'E 430', 'CLK 350', 'B 170 Edition One',
       'GLE 43 AMG', 'C 230 2.0 kompresor', 'Vito Extra Long', '200',
       'C 300 6.3 AMG Package', 'E 200 CGI', 'CL 55 AMG KOMPRESSOR',
       'S 600', 'ML 320 AMG', '280', 'CL550 AMG', 'A 170 Avangard',
       'ML 550', 'SLK 350 300', 'Vito Exstralong', 'C 240 W 203',
       'Vito Extralong', 'E 500 AVG', 'CL 600']

porsche_models = ['Cayenne', '911', 'Panamera', 'Panamera S', 'macan', 'Cayman',
       'Panamera 4', 'macan S', 'Cayenne S', 'Cayenne s', 'Panamera GTS']

bmw_models = ['X5', '535', '428 Sport Line', '328', '325', '530', '330', '750',
       'X6', '328 Xdrive', '535 M PAKET', '525', 'M6', '520', '320',
       '528 i', 'M3', 'X5 x5', '114', '320 DIESEL', '318 სასწრაფოდ',
       '335', '730 3.0', 'X5 M', '435', 'X6 M', '325 CI', '428', 'X5 3.5',
       '320 M', '528', '120', '135', '650', '550', 'X5 XDRIVE', '535 i',
       '640 GRAN-COUPE', '740', '523', '320 i', '5.30E+62', 'X3',
       'X5 e53', '435 CUPE', 'M5 Japan', 'X1', '130', 'X5 XDRIVE 35D',
       '535 Twinturbo', '550 GT', 'X5 X-Drive', '128 M tech', 'X3 3.5i',
       '550 F10', '545', 'X5 4,4i', '320 2.2', '645 CI', '225', '525 i',
       '320 2.0', '328 DIZEL', '320 320', '535 comfort-sport', '525 525',
       'X5 3.0', '745 i', '320 I', '116', '328 sulev', 'M5',
       'X1 28Xdrive', 'X5 restilling', '335 335i', '750 4.8', '318',
       'X5 rest', 'X1 X-Drive', 'X5 3.0i', '3.25E+48', '645',
       '535 Diesel', 'X1 4X4', '530 GT', '325 i', 'i3', '535 I', '328 i',
       '630', '745', '530 G30', 'M5 Машина в максимально', '525 ///M',
       '318 რესტაილინგი', 'X3 SDRIVE', 'X5 Japan', '535 535',
       '550 M Packet', '640 M', '118', '118 2,0', '335 D', '535 XI',
       '640', 'X5 4.8is', 'X4', 'X5 M packet', '320 Diesel', '420',
       'M550', '535 i xDrive', '323', '128', 'Z4', '530 M', 'M4',
       'X6 GERMANY', 'M4 Competition', 'M6 Gran cupe', '530 525i',
       'X5 DIESEL', '520 d xDrive Luxury', '730', '118 M-sport LCI',
       '316', 'X5 35d', '635', '335 M paket', 'X5 E70', 'X6 Limited',
       '320 Gran Turismo', '740 i', '535 M', '335 ტურბო', '428 i',
       '650 450 HP', '530 I', 'Z4 3,0 SI', '325 XI', 'X5 Sport', 'X6 40D',
       '528 3.0', '530 i']

jeep_models= ['Grand Cherokee', 'Wrangler', 'Compass', 'Cherokee', 'Liberty',
       'Renegade', 'Patriot', 'Grand Cherokee LAREDO', 'Wrangler ARB',
       'Grand Cherokee special e', 'Wrangler sport',
       'Patriot 70th anniversary', 'Patriot Latitude']

volkswagen_models = ['Jetta', 'Passat', 'Sharan', 'GTI', 'Golf', 'Polo', 'Jetta სპორტ',
       'Crafter', 'Tiguan', 'Golf 6', 'Jetta GLI', 'UP', 'Passat sel',
       'CC R line', 'CC 2.0 T', 'Touareg', 'Golf TDI', 'Passat R-line',
       'Caddy', 'NEW Beetle', 'CC', 'Jetta SEL', 'Passat SE', 'Touran',
       'Golf 4', 'Scirocco', 'Jetta TDI', 'Passat Se',
       '1500,1600 Schtufenheck', 'Passat sport', 'Jetta se', 'Eos',
       'Jetta SE', 'Jetta 2.0', 'Passat RLAINI', 'Phaeton',
       'Passat 2.0 tfsi', 'Jetta SPORT', 'CC sport', 'Vento',
       'Lupo iaponuri', 'Golf 1.8', 'Routan SEL', 'Golf GTI',
       'Jetta სასწაფოდ', 'Jetta sel', 'Passat se', 'Passat B7', 'Lupo',
       'Passat B5', 'Polo GTI 16V', 'Jetta sport', 'Bora', 'Jetta sei',
       'Crafter 2.5 TDI', 'Passat tdi sel', 'Crafter 2,5TDI',
       'Passat SEL', 'Passat tsi-se', 'Caddy cadi', 'Jetta s',
       'Golf GOLF 5', 'Jetta Hybrid', 'Jetta 1.4 TURBO', 'T5',
       'Tiguan SE']

audi_models = ['Q7', 'Q5', '50', 'A6', 'A7', 'A7 Prestige', 'A4 premium', 'A4',
       'Q3', 'A5', 'A3 PREMIUM', 'A8', 'A4 S line', 'A4 premium plius',
       'Allroad', 'Q5 S-line', '100', 'RS7', 'A3', 'Q7 sport', 'A3 4X4',
       'A6 С6', 'A4 B6', 'A6 UNIVERSAL', 'Q5 Prestige', 'A6 QUATTRO',
       'A4 B5', 'RS6', 'S3', 'A4 S4', 'A4 B7', 'TT', 'A6 premium plus',
       'Allroad Allroad', 'A5 Sportback', 'S6', 'A6 C7', 'A6 evropuli',
       'A4 Sline']

renault_models = ['Megane 1.5CDI', 'Twingo', 'Scenic', 'Laguna', 'Clio',
       'Megane GT Line', 'Megane', 'Kangoo', 'Duster',
       'Captur QM3 Samsung', 'Kangoo Waggon', 'Megane 1.9CDI', 'Megane 5',
       'Megane 1.9 CDI']

nissan_models = ['Juke', 'Serena', 'Maxima', 'Pathfinder', 'Rogue', 'Tiida',
       'Altima', 'Elgrand', 'X-Trail', 'Teana', 'March', 'Rogue Sport',
       'X-Terra', 'Rogue SL', 'Frontier', 'Versa', 'Note', 'Tiida 2008',
       'Sentra', 'Fuga', 'Tiida AXIS', 'Bluebird', 'Murano', 'Patrol',
       'Skyline', 'Wingroad', 'Serena Serea', 'X-Trail X-trail',
       'Presage RIDER', 'Micra', 'Quest 2016', 'Kicks', 'X-Trail gt',
       '100 NX', 'Kicks SR', 'Cefiro', 'March 231212', 'Tiida 15 M',
       'Vanette', 'Quest', 'Juke Nismo RS', 'March Rafeet', 'Armada',
       'Juke Nismo', 'Gloria', 'X-Trail NISSAN X TRAIL R', 'LATIO',
       '370Z', 'X-Trail NISMO', 'Pathfinder SE', 'Presage', 'Navara',
       'Skyline 4WD', 'Tiida Latio', 'Qashqai Advance CVT', 'Rasheen',
       'Juke nismo', 'Leaf', 'Qashqai SPORT', 'Moco', 'Almera dci',
       'Skyline GT250', 'Primera', 'LAFESTA', 'Rogue SPORT', 'Juke juke',
       'Juke Juke', 'Micra <DIESEL>', 'Juke Turbo', 'Versa s',
       'Juke NISMO', 'Versa SE']

subaru_models = ['Forester', 'Legacy', 'Outback', 'Stella', 'XV',
       'Legacy B4 twin turbo', 'XV LIMITED', 'Impreza', 'BRZ',
       'Forester SH', 'Impreza G4', 'Crosstrek', 'Forester stb',
       'Legacy Bl5', 'Forester XT', 'Forester 4x4', 'Forester CrossSport',
       'Outback Limited', 'Forester cross sport', 'Legacy bl5', 'R2',
       'Legacy B4', 'XV HYBRID', 'Impreza Sport', 'B9 Tribeca',
       'Impreza WRX/STI LIMITED', 'Legacy b4', 'Forester L.L.BEAN',
       'Outback 3.0', 'Outback 2007']

daewoo_models = ['Lacetti', 'Gentra', 'Matiz']

kia_models = ['Picanto', 'Carnival grand', 'Cerato K3', 'Optima', 'RIO', 'SOUL',
       'Sorento', 'Sportage', 'Carnival', 'Ceed', 'Cadenza', 'Cerato',
       'RIO lx', 'Avella', 'Forte', 'RIO lX', 'Optima X', 'Optima Hybrid',
       'Sorento SX', 'Optima ECO', 'Niro', 'Optima HYBRID', 'Sportage SX',
       'Sportage EX', 'Optima EX', 'Sorento EX', 'Sportage PRESTIGE',
       'Optima SXL', 'Carens', 'Optima hybrid', 'Optima hybid',
       'Optima ex', 'Optima k5']

mitsubishi_models = ['Airtrek', 'Delica', 'Colt Lancer', 'Outlander', 'I', 'Colt',
       'Pajero Mini', 'L 200', 'Montero', 'Pajero IO', 'Outlander 2.0',
       'Pajero', 'Mirage', 'Pajero Sport', 'ColtPlus', 'Grandis', 'RVR',
       'Outlander SPORT', 'Outlander SE', 'Outlander Sport',
       'Outlander sport', 'Lancer', 'Montero Sport', 'Carisma',
       'ColtPlus Plus', 'Outlander სპორტ', 'Lancer GT', 'Pajero MONTERO',
       'Minica', 'Outlander xl', 'Pajero Mini 2008 წლიანი', 'Eclipse ES',
       'Space Runner', 'Airtrek turbo', 'Eclipse', 'Galant', 'Galant GTS',
       'Lancer GTS', 'Delica 5', 'Pajero Mini 2010 წლიანი']

drive_wheels_labels=['4x4', 'Front', 'Rear']


# Judul Aplikasi
st.title("Prediksi Harga Mobil Bekas")
st.markdown("Masukkan fitur-fitur mobil untuk memprediksi harga jual menggunakan model XGBoost.")

# Input numerik
Levy = st.number_input("Levy", value=0.0)
Mileage = st.number_input("Jarak Tempuh (km)", value=50000)
Age_of_Car = st.number_input("Umur Mobil (tahun)", value=5)

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
elif Manufacturer == "TOYOTA":
    selected_model = st.selectbox("Model", sorted(toyota_models))
    Model = toyota_models.index(selected_model)
elif Manufacturer == "MARCEDEZ_BENZ":
    selected_model = st.selectbox("Model", sorted(marcedes_benz_models))
    Model = marcedes_benz_models.index(selected_model)
elif Manufacturer == "PORSCHE":
    selected_model = st.selectbox("Model", sorted(porsche_models))
    Model = porsche_models.index(selected_model)
elif Manufacturer == "BMW":
    selected_model = st.selectbox("Model", sorted(bmw_models))
    Model = bmw_models.index(selected_model)
elif Manufacturer == "JEEP":
    selected_model = st.selectbox("Model", sorted(jeep_models))
    Model = jeep_models.index(selected_model)
elif Manufacturer == "VOLKSWAGEN":
    selected_model = st.selectbox("Model", sorted(volkswagen_models))
    Model = volkswagen_models.index(selected_model)
elif Manufacturer == "AUDI":
    selected_model = st.selectbox("Model", sorted(audi_models))
    Model = audi_models.index(selected_model)
elif Manufacturer == "RENAULT":
    selected_model = st.selectbox("Model", sorted(renault_models))
    Model = renault_models.index(selected_model)
elif Manufacturer == "NISSAN":
    selected_model = st.selectbox("Model", sorted(nissan_models))
    Model = nissan_models.index(selected_model)
elif Manufacturer == "SUBARU":
    selected_model = st.selectbox("Model", sorted(subaru_models))
    Model = subaru_models.index(selected_model)
elif Manufacturer == "DAEWOO":
    selected_model = st.selectbox("Model", sorted(daewoo_models))
    Model = daewoo_models.index(selected_model)
elif Manufacturer == "KIA":
    selected_model = st.selectbox("Model", sorted(kia_models))
    Model = kia_models.index(selected_model)
elif Manufacturer == "MITSUBISHI":
    selected_model = st.selectbox("Model", sorted(mitsubishi_models))
    Model = mitsubishi_models.index(selected_model)
else:
    Model = st.number_input("Model (kode numerik)", value=1)


Leather_interior = st.selectbox("Interior Kulit", ["Tidak", "Ya"])

Fuel_type=st.selectbox("Tipe Bahan Bakar",fuel_type_labels)
Fuel_type_idx=fuel_type_labels.index(Fuel_type)

Gear_box_type =st.selectbox("Tipe Gear Box",gear_box_labels)
Gear_box_type_idx=gear_box_labels.index(Gear_box_type)

Drive_wheels =st.selectbox("Tipe Penggerak",drive_wheels_labels)
Drive_wheels_idx=drive_wheels_labels.index(Drive_wheels)

Color = st.selectbox("Warna Mobil", color_labels)
Color_idx = color_labels.index(Color)


Engine_volume = st.selectbox("Kapasitas Mesin (L)", Engine_vol_labels)
Engine_volume_idx=Engine_vol_labels.index(Engine_volume)

Cylinders = st.selectbox("Jumlah Silinder", cylinders_labels)
cylinders_idx=cylinders_labels.index(Cylinders)

Airbags = st.selectbox("Jumlah Airbags", airbags_labels)
airbags_idx=airbags_labels.index(Airbags)

is_turbo = st.selectbox("Apakah Turbo?", ["Tidak", "Ya"])

# Fitur lainnya

# Konversi boolean ke numerik
Leather_interior = 1 if Leather_interior == "Ya" else 0
is_turbo = 1 if is_turbo == "Ya" else 0

# Prediksi
if st.button("Prediksi Harga"):
    data = np.array([[Levy, Engine_volume_idx, Mileage, cylinders_idx,
                      airbags_idx, Age_of_Car, Manufacturer_idx, Model,
                      Leather_interior, Fuel_type_idx, Gear_box_type_idx, Drive_wheels_idx,
                      Color_idx, is_turbo]])
    prediksi = model.predict(data)[0]
    st.success(f"Perkiraan Harga Mobil Bekas: {prediksi:,.2f}")
