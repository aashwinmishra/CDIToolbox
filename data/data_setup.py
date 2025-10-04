import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera


def img_to_transmission(img, invert_contrast=True):    
    if invert_contrast:
        amplitude = (255 - img) / 255.0
    else:
        amplitude = img / 255.0
    return amplitude + 0j


def gaussian_probe(size, sigma, wavelength: float=1.0):
  y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
  r2 = x**2 + y**2
  amplitude = np.exp(-r2 / (2 * sigma**2))
  phase = np.zeros_like(amplitude)  
  return amplitude * np.exp(1j * phase)


def generate_scan_positions(object_size, 
                            probe_size, 
                            overlap_fraction: float=0.65):
  step_size = int(probe_size * (1 - overlap_fraction))
  positions = []
  for y in range(0, object_size[0] - probe_size, step_size):
    for x in range(0, object_size[1] - probe_size, step_size):
      positions.append((y, x))
  return np.array(positions)


def generate_diffraction_data(object_array: np.array,
                              probe_array: np.array,
                              scan_positions: np.array,
                              add_noise: bool=False,
                              noise_level: float=0.1) -> np.array:
  """
  Generates simulated Ptychography diffraction patterns
  Parameters:
    object_array: 2D, complex array describing the object transmission function
    probe_array: 2D, complex array describing the probe
    scan_positions: list of positions for the scan in pixels
    add_noise: If to add noise
    noise_level: Scaling factor for the noise
  Returns:
    intensities : 3D array of shape [n_positions, detector_y, detector_x]
    with the measured diffraction intensity fields
  """
  intensities = []
  probe_height, probe_width = probe_array.shape
  for i in range(scan_positions.shape[0]):
    y, x = scan_positions[i, 0], scan_positions[i, 1]
    patch = object_array[y:y + probe_height, x:x + probe_width] * probe_array
    fhat = np.fft.fft2(patch)
    intensity = np.abs(fhat)**2
    if add_noise:
      intensity = np.random.poisson(intensity * noise_level) / noise_level
    intensities.append(intensity)
  return np.array(intensities)


