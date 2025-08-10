import numpy as np

def wbi(r, g, b, nir, swir1, swir2,
        MNDWI_threshold=0.42, NDWI_threshold=0.4,
        filter_UABS=True, filter_SSI=False):
    ws = np.zeros_like(r, dtype=np.uint8)

    # To avoid division by zero warnings, use np.errstate
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - r) / (nir + r)
        mndwi = (g - swir1) / (g + swir1)
        ndwi = (g - nir) / (g + nir)
        ndwi_leaves = (nir - swir1) / (nir + swir1)
        aweish = b + 2.5 * g - 1.5 * (nir + swir1) - 0.25 * swir2
        aweinsh = 4 * (g - swir1) - (0.25 * nir + 2.75 * swir1)
        dbsi = ((swir1 - g) / (swir1 + g)) - ndvi
        wii = (nir ** 2) / r
        wri = (g + r) / (nir + swir1)
        puwi = 5.83 * g - 6.57 * r - 30.32 * nir + 2.25
        uwi = (g - 1.1 * r - 5.2 * nir + 0.4) / np.abs(g - 1.1 * r - 5.2 * nir)
        usi = 0.25 * (g / r) - 0.57 * (nir / g) - 0.83 * (b / g) + 1

    cond1 = (mndwi > MNDWI_threshold) | (ndwi > NDWI_threshold) | (aweinsh > 0.1879) | \
            (aweish > 0.1112) | (ndvi < -0.2) | (ndwi_leaves > 1)
    ws[cond1] = 1

    if filter_UABS:
        cond_filter = (ws == 1) & ((aweinsh <= -0.03) | (dbsi > 0))
        ws[cond_filter] = 0

    return ws


def FAI(a, b, c):
    # Wavelength constants 665, 783, 865 in nm
    return b - a - (c - a) * (783 - 665) / (865 - 665)


def NDCI(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (b - a) / (b + a)


def cyanobacteria_index(B02, B03, B04, B05, B07, B08, B8A, B11, B12):
    water = wbi(B04, B03, B02, B08, B11, B12)
    FAIv = FAI(B04, B07, B8A)
    NDCIv = NDCI(B04, B05)
    chl = 826.57 * NDCIv**3 - 176.43 * NDCIv**2 + 19 * NDCIv + 4.071

    # True color scaled by 3
    true_color = np.stack([3 * B04, 3 * B03, 3 * B02], axis=0)

    # Initialize output image as float arrays with shape (3, height, width)
    out = np.zeros_like(true_color, dtype=np.float32)

    # Water pixels: return true color
    mask_water = (water == 0)
    out[:, mask_water] = true_color[:, mask_water]

    # For water pixels, check other conditions and assign colors
    mask_nonwater = ~mask_water

    # Define color mapping helper
    def rgb(r, g, b):
        return np.array([r, g, b], dtype=np.float32)

    # Define colors scaled 0-1
    colors = {
        "orange": rgb(233/255, 72/255, 21/255),
        "blue1": rgb(0, 0, 1.0),
        "blue2": rgb(0, 59/255, 1),
        "blue3": rgb(0, 98/255, 1),
        "cyan1": rgb(15/255, 113/255, 141/255),
        "cyan2": rgb(14/255, 141/255, 120/255),
        "cyan3": rgb(13/255, 141/255, 103/255),
        "green1": rgb(30/255, 226/255, 28/255),
        "green2": rgb(42/255, 226/255, 28/255),
        "green3": rgb(68/255, 226/255, 28/255),
        "green4": rgb(134/255, 247/255, 0),
        "yellow1": rgb(205/255, 237/255, 0),
        "yellow2": rgb(251/255, 210/255, 3/255),
        "yellow3": rgb(248/255, 207/255, 2/255),
        "yellow4": rgb(245/255, 164/255, 9/255),
        "orange2": rgb(240/255, 159/255, 8/255),
        "orange3": rgb(237/255, 157/255, 7/255),
        "orange4": rgb(239/255, 118/255, 15/255),
        "orange5": rgb(239/255, 101/255, 15/255),
        "orange6": rgb(239/255, 100/255, 14/255),
    }

    # Assign colors based on conditions for water pixels only
    def assign_color(cond, color_name):
        mask = mask_nonwater & cond
        for i in range(3):
            out[i][mask] = colors[color_name][i]

    assign_color(FAIv > 0.08, "orange")
    assign_color(chl < 0.5, "blue1")
    assign_color((chl >= 0.5) & (chl < 1), "blue1")
    assign_color((chl >= 1) & (chl < 2.5), "blue2")
    assign_color((chl >= 2.5) & (chl < 3.5), "blue3")
    assign_color((chl >= 3.5) & (chl < 5), "cyan1")
    assign_color((chl >= 5) & (chl < 7), "cyan2")
    assign_color((chl >= 7) & (chl < 8), "cyan3")
    assign_color((chl >= 8) & (chl < 10), "green1")
    assign_color((chl >= 10) & (chl < 14), "green2")
    assign_color((chl >= 14) & (chl < 18), "green3")
    assign_color((chl >= 18) & (chl < 20), "green3")
    assign_color((chl >= 20) & (chl < 24), "green4")
    assign_color((chl >= 24) & (chl < 28), "green4")
    assign_color((chl >= 28) & (chl < 30), "yellow1")
    assign_color((chl >= 30) & (chl < 38), "yellow1")
    assign_color((chl >= 38) & (chl < 45), "yellow1")
    assign_color((chl >= 45) & (chl < 50), "yellow2")
    assign_color((chl >= 50) & (chl < 75), "yellow3")
    assign_color((chl >= 75) & (chl < 90), "green4")
    assign_color((chl >= 90) & (chl < 100), "yellow4")
    assign_color((chl >= 100) & (chl < 150), "orange2")
    assign_color((chl >= 150) & (chl < 250), "orange3")
    assign_color((chl >= 250) & (chl < 300), "orange4")
    assign_color((chl >= 300) & (chl < 350), "orange5")
    assign_color((chl >= 350) & (chl < 450), "orange6")
    assign_color(chl >= 450, "orange")

    return out