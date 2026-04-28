1. run draw_plates.py --> prepares data and saves it in draw_plate_data_prepared/license_plates_data.npy (an .npy file). transforms images into 28x28 pixels (as DRAW requires // double check it doesn't support other formats)
THIS DID NOT WORK SINCE IT RESIZES TO 28X28 WHICH MAKES ALL PLATES BLURRY
2. optimized version for license plates (keeping within cpu limits but may adapt to gpu later with colab, changing the size): result of our analysis:
Analyzed 201 license plate images
Average dimensions: 118×38
Min dimensions: 11×6
Max dimensions: 429×175
3. data_prep_plates.py: changed the dimensions to: TARGET_SIZE = (96, 32)  # (width, height) much less blurry. may try with better pixel quality next time
4. draw_plates_rectangle.py: runs on environment 3.7
5. changed draw_plates_rectangle.py: changes applied
============================================================
iter     0: Lx = 2131.2114, Lz = 94.8082, Total = 2226.0195
iter   100: Lx = 1968.7957, Lz = 104.1250, Total = 2072.9207
iter   200: Lx = 1923.8368, Lz = 106.3082, Total = 2030.1450
iter   300: Lx = 1967.6365, Lz = 100.2645, Total = 2067.9009
iter   400: Lx = 1941.1833, Lz = 99.0738, Total = 2040.2572
iter   500: Lx = 1927.7474, Lz = 103.2378, Total = 2030.9852
iter   600: Lx = 1937.6284, Lz = 108.6080, Total = 2046.2365
iter   700: Lx = 1879.0745, Lz = 103.6310, Total = 1982.7054
iter   800: Lx = 1908.5554, Lz = 101.9012, Total = 2010.4565
iter   900: Lx = 1798.1694, Lz = 102.8646, Total = 1901.0339
iter  1000: Lx = 1874.2662, Lz = 100.2056, Total = 1974.4718
iter  1100: Lx = 1876.9250, Lz = 100.4832, Total = 1977.4082
iter  1200: Lx = 1827.4382, Lz = 99.4838, Total = 1926.9220
iter  1300: Lx = 1836.4379, Lz = 99.7494, Total = 1936.1873
iter  1400: Lx = 1854.8152, Lz = 99.5323, Total = 1954.3475
6. Approach was giving loss around 2K, then I modified the following to get lower losses:

![alt text](image.png)

8. 

