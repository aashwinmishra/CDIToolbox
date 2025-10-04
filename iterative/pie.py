import numpy as np

def extract_patch(arr, idx, scan_positions, probe_size):
  y, x = scan_positions[idx, 0], scan_positions[idx, 1]
  return arr[y:y + probe_size, x:x + probe_size]


def apply_fourier_constraint(exit_wave, measured_intensity):
  phase = np.angle(exit_wave)
  return np.sqrt(measured_intensity) * np.exp(1j * phase)


def update_object_patch(object_patch, 
                        exit_wave_guessed, 
                        exit_wave_corrected, 
                        probe_arr, 
                        alpha, 
                        beta):
  illumination_weight = np.abs(probe_arr) / np.max(np.abs(probe_arr))
  division_term = np.conjugate(probe_arr) / (np.abs(probe_arr)**2 + alpha)
  difference_term = exit_wave_corrected - exit_wave_guessed 
  return object_patch + illumination_weight * division_term * beta * difference_term


def update_object(object_arr, new_patch, position, scan_positions, probe_size):
  y, x = scan_positions[position, 0], scan_positions[position, 1]
  object_arr[y:y + probe_size, x:x + probe_size] = new_patch


def pie_algorithm(diffraction_data: np.array, 
                  scan_positions: np.array, 
                  probe: np.array, 
                  initial_object: np.array,
                  true_object: np.array, 
                  alpha: float=0.0001, 
                  beta: float=0.5,
                  num_iter: int=100) -> np.array:
  num_patches = scan_positions.shape[0]
  probe_size = probe.shape[0] #assume square probe
  object_current = initial_object.copy()
  history = []
  for iter in range(num_iter):
    for position in range(num_patches):
      current_object_patch = extract_patch(object_current, position, scan_positions, probe_size)
      patch_exit_wave = current_object_patch * probe 
      fourier_patch_exit_wave = np.fft.fft2(patch_exit_wave)
      fourier_patch_exit_wave_corr = apply_fourier_constraint(fourier_patch_exit_wave, diffraction_data[position])
      patch_exit_wave_corr = np.fft.ifft2(fourier_patch_exit_wave_corr)
      new_object_patch = update_object_patch(current_object_patch, patch_exit_wave, patch_exit_wave_corr, probe, alpha, beta)
      update_object(object_current, new_object_patch, position, scan_positions, probe_size)
    loss = np.mean(np.abs(object_current - true_object))
    print(f"Iter: {iter} Loss: {loss}")
    history.append(loss)
  plt.plot(history)
  return object_current
