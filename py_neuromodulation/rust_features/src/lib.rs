use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{IntoPyArray, PyArrayDyn};
use ndarray::{Array3, Axis};
use num_complex::Complex;
use realfft::RealFftPlanner;
use rayon::prelude::*;

#[pyfunction]
fn calculate_bispectra(py: Python, data: &PyArrayDyn<f64>) -> PyResult<PyObject> {
    let data_array = unsafe{data.as_array()};
    let result = bispectra_calculation(data_array)?;
    // Convert the result back to NumPy array TODO: This also copies the data, can we avoid this?
    Ok(result.into_pyarray_bound(py).into_py(py))
}

// Implement the actual bispectra calculation
fn bispectra_calculation(data: ndarray::ArrayViewD<f64>) -> PyResult<Array3<Complex<f64>>> {
    assert_eq!(data.ndim(), 2, "Input data must be a 2D array");

    let num_channels = data.shape()[0];
    let signal_length = data.shape()[1];
    let half_N = signal_length / 2;

    //Real fftplanner is a bit faster than the complex one -> asumption is that the input is real like thomas did
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(signal_length);

    // Hardcoded freq range -> might need to be passed as an argument when really implementing this
    let freq_start = 5;
    let freq_end = 35;
    let num_freqs = freq_end - freq_start + 1;

    // Init bispec array with hardcoded sizes still
    let mut bispectrum = Array3::<Complex<f64>>::zeros((num_channels, num_freqs, num_freqs));

    // Parallel channel processing!
    bispectrum
        .axis_iter_mut(Axis(0))
        .into_par_iter() // par_iter does give me a boost in performance when parallelizing channels!! 
        .enumerate()
        .try_for_each(|(s, mut bispectrum_s)| -> Result<(), PyErr> {
            // Get the signal for the current channel TODO: Check if this is is possible to do without copying! -> potentially boost
            let signal = data.index_axis(ndarray::Axis(0), s);

            // input and output buffers for the FFT
            let mut input = signal.to_owned(); // Copy the signal since FFT modifies it
            let mut output = r2c.make_output_vec();

            r2c.process(input.as_slice_mut().unwrap(), &mut output).unwrap();

            // Now output has length half_N + 1
            let fft_result = &output;

            // Compute bispectrum only for the relevant frequencies like thomas did
            for (i1, f1) in (freq_start..=freq_end).enumerate() {
                let X_f1 = fft_result[f1];
                for (i2, f2) in (freq_start..=freq_end).enumerate() {
                    let X_f2 = fft_result[f2];
                    let f3 = f1 + f2;
                    let X_f3_conj = if f3 <= half_N {
                        fft_result[f3].conj()
                    } else if f3 < signal_length {
                        fft_result[signal_length - f3].conj()
                    } else {
                        fft_result[f3 - signal_length].conj()
                    };
                    let triple_product = X_f1 * X_f2 * X_f3_conj;
                    bispectrum_s[[i1, i2]] = triple_product;
                }
            }
            Ok(())
        })?;

    Ok(bispectrum)
}

#[pymodule]
fn rust_features(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_bispectra, m)?)?;
    Ok(())
}
