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


def decompose_profile(x, x0, prf, prf_core, ngaus=2, nexp=3, win=100,
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


def write_output(data, bin_wranges, bin_wcenters, ofile=None, verbose=False):
    "Write stellar profile modeling results to the binary fits table."
    nbins = len(data)
    data = np.array(data)

    xprfs = np.vstack(data[:, 0])
    prfs = np.vstack(data[:, 1])
    models = np.vstack(data[:, 2])
    comps = np.stack(data[:,3])
    xscat = np.vstack(data[:, 4])
    scat = np.vstack(data[:, 5])
    xmaxs = np.vstack(data[:, 6])
    maxfracs = np.vstack(data[:, 7])
    wins = np.vstack(data[:, 8])
    bin_wranges = np.array(bin_wranges)
    bin_wcenters = np.array(bin_wcenters)

    columns = [
        fits.Column(name='PRF_X', format="{}D".format(xprfs.shape[1]),
                    array=xprfs),
        fits.Column(name='PRF', format="{}D".format(prfs.shape[1]),
                    array=prfs),
        fits.Column(name='FIT', format="{}D".format(models.shape[1]),
                    array=models),
        fits.Column(name='FIT_COMPS', format="{}D".format(comps.shape[1]*comps.shape[2]),
                    dim="({},{})".format(comps.shape[2], comps.shape[1]), array=comps),
        fits.Column(name='PSF_SCAT_X', format="{}D".format(xscat.shape[1]),
                    array=xscat),
        fits.Column(name='PSF_SCAT_Y', format="{}D".format(scat.shape[1]),
                    array=scat),
        fits.Column(name='PARS_WAVE_RANGE', format="{}D".format(bin_wranges.shape[1]),
                    array=bin_wranges),
        fits.Column(name='PARS_WAVE_CENTER', format="D", array=bin_wcenters),
        fits.Column(name='PARS_MAX_POS', format="I", array=xmaxs),
        fits.Column(name='PARS_MAXFRAC', format="D", array=maxfracs),
        fits.Column(name='PARS_WIN', format="D", array=wins),
    ]
    htbl = fits.BinTableHDU.from_columns(columns)
    htbl.name = 'PSF_SCATTERING'
    if verbose:
        print("Output binary table: {}".format(ofile))
    htbl.writeto(ofile, overwrite=True)


@argh.arg('-b', '--binnum', type=str, help="Which spectral bins should be "
          "plotted. Using default value `all` will generate plots for all "
          "bins. More than one bins might be specified like `0,2,5` (without "
          "spaces!)")
@argh.arg('-p', '--pdf', type=str, help="Output multi-page pdf file.")
@argh.arg('-y', '--yrange', type=float, nargs=2,
          help="Y-axis range of the plots.")
def plot(file, binnum='all', pdf=None, yrange=[1e-6, 2]):
    "Plot figures of stellar profile analysis."

    d = fits.getdata(file, 'PSF_SCATTERING')
    nbins = len(d)
    if binnum == 'all':
        bins = np.arange(nbins)
    elif ',' in binnum:
        bins = np.array(binnum.split(',')).astype(int)
    else:
        bins = np.array([binnum])

    print("Make plots for spectral bins: {}".format(bins))

    plt.close()
    if pdf is not None:
        print("Write multi-page pdf: {}".format(pdf))
        from matplotlib.backends.backend_pgf import PdfPages
        p = PdfPages(pdf)

    for bn in bins:
        xmax = d[bn]['PARS_MAX_POS']
        wcenter = d[bn]['PARS_WAVE_CENTER']
        wrange = d[bn]['PARS_WAVE_RANGE']
        maxfrac = d[bn]['PARS_MAXFRAC']
        win = d[bn]['PARS_WIN']

        f = plt.figure()
        ax = f.add_subplot()

        ax.plot(d[bn]['PRF_X']-xmax, d[bn]['PRF'], color='k')
        comps = d[bn]['FIT_COMPS']
        for k in range(comps.shape[1]):
            ax.plot(d[bn]['PSF_SCAT_X'], comps[:, k], color='C1', lw=0.5)
        ax.plot(d[bn]['PSF_SCAT_X'], d[bn]['PSF_SCAT_Y'], color='C0')
        ax.plot(d[bn]['PRF_X']-xmax, d[bn]['FIT'], color='red')

        ax.axvline(0, linestyle=':')
        ax.axhline(maxfrac, linestyle=':')
        ax.set_yscale('log')
        ax.set_xlim(-win, win)
        ax.set_ylim(yrange)
        ax.set_title(r"Bin {}  $\lambda$ {:.1f}  $\Delta \lambda$ "
                      "{:.1f}-{:.1f}".format(bn, wcenter, wrange[0], wrange[1]))
        ax.secondary_xaxis('top', functions=(lambda x: x + xmax, 
                                             lambda x: x - xmax))

        if pdf is not None:
            p.savefig(f)
    p.close()
    plt.show()


def model_profile(prf, maxfrac=0.10, win=100, mode='gaussian', cocentered=False,
                  ngaus=2, nexp=3, plot=False):
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
                          ngaus=ngaus, nexp=nexp, win=win, cocentered=cocentered)

    if plot:
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

    return x, prf, model, comps, xscat, scat, xmax, maxfrac, win


@argh.arg('infile', type=str, help="Fits file with long-slit spectrum of the "
          "standard star. It has to be night sky removed.")
@argh.arg('-o', '--ofile', type=str, help="Fits file name where binary table "
          "with output parameters will be stored. If not specified than "
          "`_psfscat` will be added to the fits file.")
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
@argh.arg('--ngaus', type=int, help="Number of gaussian components described "
          "scattering function.")
@argh.arg('--nexp', type=int, help="Number of exponential components described "
          "scattering function.")
@argh.arg('-c', '--cocentered', help="Whether all component represented "
          "scattering function have the same central position.")
def fit(infile, ofile=None, ranges=[5500, 5600, 6060, 6200], win=100, maxfrac=0.10,
        mode='gaussian', cocentered=False, ngaus=2, nexp=3, verbose=True,
        plot=False):
    """
    Estimate scattering light profile using long-slit spectra of standard star.
    """
    hdr = fits.getheader(infile)
    spec = fits.getdata(infile)
    wrange, waves = get_waves(hdr)

    # some checks
    assert (len(ranges) % 2) == 0, (
        "Ranges argument has to be even! See syntax. `scatlight fit --help`"
        )
    nbins = len(ranges) // 2

    outdata = []
    bin_wranges = []
    bin_wcenters = []
    for i in range(nbins):
        idxbin = (waves >= ranges[2*i]) & (waves <= ranges[2*i+1])
        prf = np.nansum(spec[:, idxbin], axis=1)
        out = model_profile(
            prf, maxfrac=maxfrac, mode=mode, win=win, cocentered=cocentered,
            ngaus=ngaus, nexp=nexp, plot=plot)
        outdata.append(out)
        bin_wranges.append([waves[idxbin][0], waves[idxbin][-1]])
        bin_wcenters.append(0.5 * (waves[idxbin][0] + waves[idxbin][-1]))

    if ofile is None:
        ofile = infile.replace('.fits', '_psfscat.fits')
    write_output(outdata, bin_wranges, bin_wcenters, ofile=ofile, verbose=True)


parser = argh.ArghParser()
parser.add_commands([fit, plot])

if __name__ == '__main__':
    parser.dispatch()
