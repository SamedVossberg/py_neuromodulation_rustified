use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::{PyArrayDyn, ToPyArray};

use ndarray::{ArrayD, Array2, Array3, Axis};

use num_complex::Complex;

use rustfft::FftPlanner;

use rayon::prelude::*;

#[pyfunction]
fn calculate_bispectra(
    data: &PyArrayDyn<f64>,
) -> PyResult<PyObject> {
    let data_array: ArrayD<f64> = data.readonly().as_array().to_owned();

    // Process data
    let result: ArrayD<f64> = bispectra_calculation(&data_array);

    // Convert the result back to PyObject (as NumPy array)
    let py_result: PyObject = Python::with_gil(|py| result.to_pyarray(py).into_py(py));

    Ok(py_result)
}

// Implement the actual bispectra calculation
fn bispectra_calculation(data: &ArrayD<f64>) -> ArrayD<f64> {
    assert_eq!(data.ndim(), 2, "Input data must be a 2D array");

    let shape = data.shape();
    let num_channels = shape[0];
    let signal_length = shape[1];

    // Convert data to Complex<f64>
    let mut data_complex = Array2::<Complex<f64>>::zeros((num_channels, signal_length));
    data_complex
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(data.axis_iter(Axis(0)))
        .for_each(|(mut row_complex, row_real)| {
            row_complex
                .iter_mut()
                .zip(row_real)
                .for_each(|(elem_complex, &elem_real)| {
                    *elem_complex = Complex::new(elem_real, 0.0);
                });
        });

    // Perform FFT on each signal
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal_length);

    data_complex
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut signal| {
            fft.process(signal.as_slice_mut().unwrap());
        });

    // Compute bispectrum for each channel
    let N = signal_length;
    let half_N = N / 2;

    let mut bispectrum = Array3::<f64>::zeros((num_channels, half_N, half_N));

    bispectrum
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(s, mut bispectrum_s)| {
            for f1 in 0..half_N {
                for f2 in 0..half_N {
                    let f3 = (f1 + f2) % N;
                    let X_f1 = data_complex[[s, f1]];
                    let X_f2 = data_complex[[s, f2]];
                    let X_f3_conj = data_complex[[s, f3]].conj();
                    let triple_product = X_f1 * X_f2 * X_f3_conj;
                    bispectrum_s[[f1, f2]] = triple_product.norm();
                }
            }
        });

    bispectrum.into_dyn()
}

#[pymodule]
fn rust_features(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_bispectra, m)?)?;
    Ok(())
}
