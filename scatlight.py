#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import lmfit
import argh


def get_waves(hdr, vacuum=False):
    """
    Get wavelengths from the header.
    """
    waves = hdr['CRVAL1'] - (hdr['CRPIX1']-1.0)*hdr['CDELT1'] + \
        (np.arange(0., hdr['NAXIS1'])) * hdr['CDELT1']

    if vacuum:
        waves = waves / (1.0 + 2.735182E-4 + 131.4182 /
                         waves**2 + 2.76249E8 / waves**4)
    ran = [waves[0], waves[-1]]
    return ran, waves


@argh.arg('file', type=str,
          help="""Fits file with long-slit spectrum of the standard star. It has
                to be night sky removed.""")
@argh.arg('--ranges', nargs='+', type=float,
          help="""Wavelength intervals of the bins where stellar profile will be
          analysed. Has to be even. [left1, right1, left2, right2, ...]""")
def fit(file, ranges=[5500, 5600, 6060, 6200], verbose=True, plot=False):
    """
    Estimate scattering light profile using long-slit spectra of standard star.
    """
    hdr = fits.getheader(file)
    spec = fits.getdata(file)
    wrange, waves = get_waves(hdr)

    # some checks
    assert (len(ranges) % 2) == 0, (
        "Ranges argument has to be even! See syntax. `scatlight fit --help`"
        )
    nbins = len(ranges) // 2

    print(spec.shape)
    for i in range(nbins):
        idxbin = (waves >= ranges[2*i]) & (waves <= ranges[2*i+1])
        prf = np.nansum(spec[:, idxbin], axis=1)
        plt.plot(prf)
        plt.show()


parser = argh.ArghParser()
parser.add_commands([fit])

if __name__ == '__main__':
    parser.dispatch()
