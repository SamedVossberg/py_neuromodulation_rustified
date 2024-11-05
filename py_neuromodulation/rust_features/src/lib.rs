use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{IntoPyArray, PyArrayDyn};
use ndarray::{Array3, Axis};
use num_complex::Complex;
use realfft::RealFftPlanner;
use rayon::prelude::*;

#[pyfunction]
fn calculate_bispectra(py: Python, data: &PyArrayDyn<f64>) -> PyResult<PyObject> {
    let data_array = unsafe { data.as_array() };

    // Process data
    let result = bispectra_calculation(data_array)?;

    // Convert the result back to NumPy array
    Ok(result.into_pyarray_bound(py).into_py(py))
}

// Implement the actual bispectra calculation
fn bispectra_calculation(data: ndarray::ArrayViewD<f64>) -> PyResult<Array3<Complex<f64>>> {
    //assert_eq!(data.ndim(), 2, "Input data must be a 2D array");

    println!("Data shape: {:?}", data.shape());

    let num_channels = data.shape()[0];
    let signal_length = data.shape()[1];
    let half_N = signal_length / 2;

    // Initialize Real FFT planner
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(signal_length);

    // Prepare the bispectrum array
    let mut bispectrum = Array3::<Complex<f64>>::zeros((num_channels, half_N + 1, half_N + 1));

    // Process each channel in parallel
    bispectrum
        .axis_iter_mut(Axis(0))
        .into_par_iter() // par iter is faster than iter!!!
        .enumerate()
        .try_for_each(|(s, mut bispectrum_s)| -> Result<(), PyErr> {
            // Get the signal for the current channel
            let signal = data.index_axis(Axis(0), s);

            // Prepare input and output buffers for the FFT
            let mut input = signal.to_owned();
            let mut output = r2c.make_output_vec();
            let mut input_slice = input.as_slice_mut().unwrap();

            // Perform FFT
            r2c.process(&mut input_slice, &mut output).unwrap();

            // Now output has length half_N + 1
            let fft_result = &output;

            // Compute bispectrum
            for f1 in 5..=35 { //hardcoded frequency ranges -> f1s
                let X_f1 = fft_result[f1];
                for f2 in 5..=35 { //hardcoded frequency ranges -> f2s
                    let X_f2 = fft_result[f2];
                    let f3 = f1 + f2;
                    let X_f3_conj = if f3 <= half_N {
                        fft_result[f3].conj()
                    } else if f3 < signal_length {
                        // Use symmetry for frequencies beyond Nyquist
                        fft_result[signal_length - f3].conj()
                    } else {
                        // f3 wraps around due to modulo operation
                        fft_result[f3 - signal_length].conj()
                    };
                    let triple_product = X_f1 * X_f2 * X_f3_conj;
                    bispectrum_s[[f1, f2]] = triple_product;
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
