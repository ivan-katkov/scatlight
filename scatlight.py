#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, CustomKernel
import lmfit
from lmfit.models import GaussianModel, MoffatModel
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


def decompose_profile(x, x0, prf, prf_core, ngaus=4, nexp=4, win=100,
                      cocentered=False):
    "Model scattering function"

    def resids(p, x, prf, prf_core, ngaus=None, nexp=None, mode='fit'):
        nxx = int(prf.size/2.2)
        xx = np.arange(-nxx, nxx+1)
        comps = np.zeros((xx.size, ngaus+nexp))
        for igau in range(ngaus):
            pref = "gau{}".format(igau+1)
            gint, gsig, gpos = p[pref+'_int'], p[pref+'_sig'], p[pref+'_pos']
            gyy = (xx -gpos) / gsig
            comps[:, igau] = gint * np.exp(-0.5*gyy**2)

        for iexp in range(nexp):
            pref = "exp{}".format(iexp+1)
            eint, esig, epos = p[pref+'_int'], p[pref+'_sig'], p[pref+'_pos']
            comps[:, ngaus + iexp] = eint * np.exp(-np.abs((xx-epos)/esig))
        scat = np.sum(comps, axis=1)

        model = prf_core * (1 - p['alpha']) + \
            convolve(prf_core, scat, normalize_kernel=False)

        if mode == 'fit':
            return prf - model
        elif mode == 'model':
            return model, comps, xx, scat

    ymax = np.nanmax(prf)

    xmsk = np.argwhere(np.abs(x - x0) <= win)
    # check does xmsk is even or not
    if len(prf[xmsk]) % 2 == 0:
        xmsk = xmsk[0:-1].ravel()

    params = lmfit.Parameters()
    params.add("alpha", value=0.2, min=0.0, max=1.0)
    for igau in range(ngaus):
        pref = "gau{}".format(igau+1)
        params.add(pref+"_int", value=0.01/(igau+1), min=0, max=1.5*ymax, vary=True)
        params.add(pref+"_sig", value=2.0*(igau+1), min=1.1, max=10.0, vary=True)
        params.add(pref+"_pos", value=0.0, min=-10, max=10.0, vary=True)
    for iexp in range(nexp):
        pref = "exp{}".format(iexp+1)
        params.add(pref+"_int", value=0.0015/(iexp+1), min=0, max=1.5*ymax, vary=True)
        params.add(pref+"_sig", value=16*(iexp+1), min=1.0, max=100.0, vary=True)
        params.add(pref+"_pos", value=0.0, min=-10, max=10.0, vary=True)
    if cocentered:
        for par in params:
            if ('_pos' in par) & (par != 'gau1_pos'):
                params[par].expr = 'gau1_pos'

    args = (x[xmsk], prf[xmsk], prf_core[xmsk])
    kws = dict(ngaus=ngaus, nexp=nexp)
    res = lmfit.minimize(resids, params, method='powell', args=args,
                         kws=kws)
    model, comps, xx, scat = resids(res.params, x, prf, prf_core, **kws, mode='model')
    lmfit.report_fit(res)
    return model, comps, xx, scat


def model_profile(prf, maxfrac=0.10, win=100, mode='gaussian', cocentered=False):
    "Model profile and determine scattering function."

    xmax = np.argmax(prf)
    prf /= np.nanmax(prf)
    x = np.arange(prf.size)
    xmsk = prf >= maxfrac

    # fit central part of the peak
    kws = dict(center=xmax, amplitude=5.0, sigma=5.0)
    if mode == 'gaussian':
        model_peak = GaussianModel()
    elif mode == 'moffat':
        model_peak = MoffatModel()
        kws['beta'] = 3
    else:
        raise NameError("Incorrect mode name which has to be `gaussian` or "
                        "`moffat`")

    model_res = model_peak.fit(prf[xmsk], x=x[xmsk], **kws)
    print(model_res.fit_report())
    model_peak = model_res.eval(x=x)

    prf_core = model_peak / np.sum(model_peak) * np.nansum(prf)
    model, comps, xscat, scat = \
        decompose_profile(x, model_res.params['center'].value, prf, prf_core,
                          ngaus=2, nexp=3, win=win, cocentered=cocentered)

    # print(comps.shape)
    resid = prf - model
    rms = np.nanstd(resid)
    plt.close()
    f = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for ax in ax1, ax2:
        ax.plot(x, prf, color='k')
        ax.plot(x, model_peak, color='C1')

        for k in range(comps.shape[1]):
            ax.plot(xscat+xmax, comps[:, k], color='red', lw=0.5)
        ax.plot(xscat+xmax, scat, color='C0')
        ax.plot(x, model, color='red')
        ax.axvline(xmax, linestyle=':')
        ax.axhline(maxfrac, linestyle=':', color='k')
        ax.secondary_xaxis('top', functions=(
            lambda x: x - xmax, lambda x: x + xmax))

    
    ax2.plot(x, resid - 5*rms)
    ax2.axhline(-5*rms)
    ax1.set_ylim(1e-6, 2)
    ax1.set_xlim(xmax-win, xmax+win)
    ax2.set_xlim(xmax-10, xmax+10)
    # ax.xlim(xmax-10, xmax+10)
    ax1.set_yscale('log')
    
    plt.show()


@argh.arg('file', type=str, help="Fits file with long-slit spectrum of the "
          "standard star. It has to be night sky removed.")
@argh.arg('--ranges', nargs='+', type=float, help="Wavelength intervals of the "
          "bins where stellar profile will be analysed. Has to be even. "
          "[left1, right1, left2, right2, ...]")
@argh.arg('--win', type=int, help="Window (+/- value) around the peak center "
          "where stellar profile is modelled.")
@argh.arg('-mf', '--maxfrac', type=float, help="Defines which part of the peak "
          "approximated to model non-scattered stellar profile. maxfrac=0.2 "
          "means that the peak profile where prf > 0.2*max is fitted.")
@argh.arg('--mode', type=str, help="Defines model which used for non-scattered "
          "profile modeling. Default is `gaussian`. `moffat` is another option.")
@argh.arg('-c', '--cocentered', help="Whether all component represented "
          "scattering function have the same central position.")
def fit(file, ranges=[5500, 5600, 6060, 6200], win=100, maxfrac=0.10,
        mode='gaussian', cocentered=False, verbose=True, plot=False):
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
        model_profile(prf, maxfrac=maxfrac, mode=mode, win=win,
                      cocentered=cocentered)


parser = argh.ArghParser()
parser.add_commands([fit])

if __name__ == '__main__':
    parser.dispatch()
