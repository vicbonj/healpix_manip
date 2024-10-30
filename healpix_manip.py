import numpy as np
import healpy as hp
from astropy import wcs
from tqdm import tqdm
from skimage.transform import resize


def make_wcs(size, glon, glat, reso, rot_rand):
    w = wcs.WCS(naxis = 2)
    w.wcs.crpix = [(size-1)/2., (size-1)/2.]
    w.wcs.crval = [glon, glat]
    w.wcs.cdelt = [-reso, reso]
    w.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
    w.wcs.crota = [-rot_rand, -rot_rand]
    return w

def map_proj(w):
    size = int(w.wcs.crpix[0]*2+1)
    yy, xx = np.indices((size, size))
    patch_GLON, patch_GLAT = w.wcs_pix2world(xx, yy, 1)
    return patch_GLON, patch_GLAT

def extract_proj(maps, patch_GLON, patch_GLAT, nside, interp):
    p_PHI = patch_GLON*np.pi/180.
    p_TH = patch_GLAT*np.pi/180.
    if interp is True:
        patch = [hp.get_interp_val(m, np.pi/2.-p_TH, p_PHI) for m in maps]
    else:
        pix = hp.ang2pix(nside, np.pi/2-p_TH, p_PHI)
        patch = [m[pix] for m in maps]
    return patch

def from_healpix_to_maps(maps, nside=2048, nside_cut=16, npix=256+16*2, reso=1.5/60, interp=False):
    patches = []
    for i in tqdm(range(hp.nside2npix(nside_cut))):
        lon, lat = hp.pix2ang(nside_cut, i, lonlat=True)
        if abs(lat) <= 45:
            rot = 45
        elif (abs(lat) > 45) & (abs(lat) <= 60):
            rot = 45
        elif (abs(lat) > 60) & (abs(lat) <= 84):
            rot = 45# * (i%2)
        else:
            rot = 45
        #if cut is not False:
        #    pixels = npix+cut*2
        #else:
        pixels = npix
        w = make_wcs(pixels, lon, lat, reso, rot)
        p_GLON, p_GLAT = map_proj(w)
        patch = extract_proj(maps, p_GLON, p_GLAT, nside, interp=interp)
        patches.append(patch)
    return patches

def from_maps_to_healpix(patches, nside=2048, nside_cut=16, npix=256+16*2, reso=1.5/60, cut=24, fact=2):
    rec_map = np.zeros((len(patches[0]), hp.nside2npix(nside)))
    counts = np.zeros(hp.nside2npix(nside))
    
    remove = int(fact/2)
    if cut is not False:
        pixels = npix
        rem = remove + cut*fact
    else:
        pixels = npix
        rem = remove
    for i in tqdm(range(hp.nside2npix(nside_cut))):
        lon, lat = hp.pix2ang(nside_cut, i, lonlat=True)
        if abs(lat) <= 45:
            rot = 45
        elif (abs(lat) > 45) & (abs(lat) <= 60):
            rot = 45
        elif (abs(lat) > 60) & (abs(lat) <= 84):
            rot = 45# * (i%2)
        else:
            rot = 45
        w = make_wcs(pixels, lon, lat, reso, rot)
        patch_GLON, patch_GLAT = map_proj(w)
        
        if (np.min(patch_GLON) < 1) & (np.max(patch_GLON) > 359):
            patch_GLON[patch_GLON < 180] += 360
            
        if fact > 1:
            patch_GLON_ok = resize(patch_GLON, (npix*fact, npix*fact))[rem:-rem,rem:-rem]%360
            patch_GLAT_ok = resize(patch_GLAT, (npix*fact, npix*fact))[rem:-rem,rem:-rem]
            patch_ok = [resize(p, (len(p[0])*fact, len(p[0])*fact), order=0)[remove:-remove,remove:-remove] for p in patches[i]]
        else:
            if rem > 0:
                patch_GLON_ok = patch_GLON[rem:-rem,rem:-rem]%360
                patch_GLAT_ok = patch_GLAT[rem:-rem,rem:-rem]
            else:
                patch_GLON_ok = patch_GLON%360
                patch_GLAT_ok = patch_GLAT
                patch_ok = patches[i]
            
        pix = hp.ang2pix(nside, patch_GLON_ok.ravel(), patch_GLAT_ok.ravel(), lonlat=True)

        for j in range(len(rec_map)):
            its = patch_ok[j].ravel()
            for k in range(len(pix)):
                rec_map[j][pix[k]] += its[k]
                if j == 0:
                    counts[pix[k]] += 1

    for j in range(len(rec_map)):
        rec_map[j] = rec_map[j]/counts

    return rec_map, counts

def populate_diagonal_by_diagonal(to_populate):
    n = int(np.sqrt(len(to_populate)))
    new_array = np.zeros((n, n))
    first = [to_populate[np.sum(np.arange(col+1)):np.sum(np.arange(col+1))+col+1] for col in range(n)]
    second = [to_populate[::-1][np.sum(np.arange(col+1)):np.sum(np.arange(col+1))+col+1] for col in range(n-1)]
    for col in range(n):
        i, j = 0, col
        while i < n and j >= 0:
            new_array[i][j] = first[col][i]
            if col < n-1:
                new_array[::-1,::-1][i][j] = second[col][i]
            i += 1
            j -= 1
    return new_array

def from_healpix_to_maps_new(healpix_map, nside_cut=4):
    nside_map = hp.npix2nside(len(healpix_map))
    conv = hp.ang2pix(nside_cut, *hp.pix2ang(nside_map, np.arange(hp.nside2npix(nside_map))))
    big_pixels = np.zeros(hp.nside2npix(nside_map))
    for i in range(hp.nside2npix(nside_cut)):
        big_pixels[conv == i] = i

    nb_pix = int(np.sqrt(hp.nside2npix(nside_map)/hp.nside2npix(nside_cut)))
    patches = np.zeros((hp.nside2npix(nside_cut), nb_pix, nb_pix))

    print("Cutting HEALPIX map with Nside={} onto {} patches of {}x{} pixels...".format(nside_map, len(patches), nb_pix, nb_pix))
    for i in tqdm(range(hp.nside2npix(nside_cut))):
        patch1d = healpix_map[big_pixels == i]
        patch2d = populate_diagonal_by_diagonal(patch1d)
        patches[i] = patch2d

    return patches

def from_2d_to_diagonal(patch):
    n = len(patch)
    first = [np.diag(patch[:i+1,:i+1][::,::-1]) for i in range(n)]
    second = [np.diag(patch[::-1,::-1][:i+1,:i+1][::,::-1])[::-1] for i in range(n-1)][::-1]
    
    total = []
    for f in first:
        total.extend(f)
    for s in second:
        total.extend(s)

    return np.array(total)

def from_maps_to_healpix_new(patches, nside_out=256):
    healpix = np.zeros(hp.nside2npix(nside_out))
    nside_cut = hp.npix2nside(len(patches))
    conv = hp.ang2pix(nside_cut, *hp.pix2ang(nside_out, np.arange(hp.nside2npix(nside_out))))
    big_pixels = np.zeros(hp.nside2npix(nside_out))
    for i in range(hp.nside2npix(nside_cut)):
        big_pixels[conv == i] = i

    print("Reconstructing onto an HEALPIX map of Nside={}".format(nside_out))
    for i in tqdm(range(len(patches))):
        patch1d = from_2d_to_diagonal(patches[i])
        healpix[big_pixels == i] = patch1d

    return healpix
