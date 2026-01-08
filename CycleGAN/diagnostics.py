import numpy as np
import scipy.stats
import torch 
def get_radial_profile(image):
    """
    Calculates the 1D Azimuthally Averaged Power Spectrum of a 2D image.
    Standard diagnostic for astrophysical turbulence/structure.
    """
    y, x = np.indices(image.shape)
    center = np.array(image.shape) / 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    # Sum over variance (power)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    # Return only up to the Nyquist frequency (half image size)
    return radialprofile[:image.shape[0] // 2]

def calculate_physics_metrics(generator, dataloader, device, n_samples=50):
    """
    Compares the Power Spectrum and Histogram of Real vs Fake data.
    Returns:
        psd_error: Low value means structure sizes are correct.
        hist_error: Low value means flux/intensity values are correct.
    """
    generator.eval()
    
    real_psds = []
    fake_psds = []
    real_hists = []
    fake_hists = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * batch['A'].shape[0] >= n_samples: break
            
            real_A = batch['A'].to(device).cpu().numpy()
            
            # Generate Fake B (using G_AB) or Fake A (using G_BA)
            # Usually we want to compare Domain B (Real) vs Domain B (Fake generated from A)
            # But since we don't have paired data, we compare statistics of the population.
            fake_B = generator(batch['A'].to(device)).cpu().numpy()

            # Iterate over batch
            for j in range(real_A.shape[0]):
                # --- Metric 1: Power Spectrum (on Channel 0 for simplicity) ---
                # We take FFT, then magnitude squared
                real_fft = np.fft.fftshift(np.fft.fft2(real_A[j, 0, :, :]))
                real_power = np.abs(real_fft)**2
                real_psds.append(get_radial_profile(real_power))

                fake_fft = np.fft.fftshift(np.fft.fft2(fake_B[j, 0, :, :]))
                fake_power = np.abs(fake_fft)**2
                fake_psds.append(get_radial_profile(fake_power))

                # --- Metric 2: Histogram / Flux Distribution ---
                # We normalize histograms to treat them as probability distributions
                rh, _ = np.histogram(real_A[j, 0, :, :], bins=50, range=(-1, 1), density=True)
                fh, _ = np.histogram(fake_B[j, 0, :, :], bins=50, range=(-1, 1), density=True)
                real_hists.append(rh)
                fake_hists.append(fh)

    # Average over all samples
    avg_real_psd = np.mean(real_psds, axis=0)
    avg_fake_psd = np.mean(fake_psds, axis=0)
    
    avg_real_hist = np.mean(real_hists, axis=0)
    avg_fake_hist = np.mean(fake_hists, axis=0)

    # Log Difference (Mean Squared Log Error for PSD because scales vary wildly)
    psd_error = np.mean((np.log10(avg_real_psd + 1e-8) - np.log10(avg_fake_psd + 1e-8))**2)
    
    # Histogram Difference (MSE)
    hist_error = np.mean((avg_real_hist - avg_fake_hist)**2)

    return psd_error, hist_error