# scatlight
Calculation of light scattering profile in the astronomical long-slit spectroscopic data.


This approach has been developed and tested for long-slit data taken with RSS spectrograph mounted at SALT telescope. See detailed description in the paper Katkov, Kniazev et al..
[Katkov, Kniazev, Kasparova, Sil'chenko, MNRAS 483, 2413 (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.2413K/abstract)

---
**NOTE**

To estimate scattering light profile one needs reference star spectrum and good knowledge of the PSF at the observation moment.

---


## Example

<img src="https://github.com/ivan-katkov/scatlight/blob/master/pic_example.png" width="30%" height="30%" alt="Example of scattered light calculation">

The top panel shows the light profile of a reference star along the slit (black colour) which was used for the scattered light component calculation.
The red line represents the result of the convolution of an atmospheric PSF (green lines) with the model of the instrumental PSF component fscat (blue lines).
Two different parameterizations for the atmospheric PSF have been used.
The Gaussian one is shown with a solid line and the Moffat one with a dashed line.
Both parameterizations provide a different estimation of an additive component of the instrumental scattering function fscat.
The middle panel shows the galaxy light profile of NGC 1380A including the night sky background along the slit at a wavelength 5096 Å.
The estimated scattered light component as shown by the blue lines corresponding
to both Gaussian and Moffat shapes of the atmospheric PSF.
The lower panel indicates the fraction of light scattered by the telescope and instrument with respect to the total light distribution.
See Section 2.3 in the [Paper](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.2413K/abstract) for a detailed description.


