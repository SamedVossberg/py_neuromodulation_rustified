from collections.abc import Iterable
from pydantic import field_validator
from typing import TYPE_CHECKING, Callable

import numpy as np
import time

from py_neuromodulation.utils.types import (
    NMBaseModel,
    NMFeature,
    BoolSelector,
    FrequencyRange,
)

import rust_features  # Import the Rust module

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class BispectraComponents(BoolSelector):
    absolute: bool = True
    real: bool = True
    imag: bool = True
    phase: bool = True


class BispectraFeatures(BoolSelector):
    mean: bool = True
    sum: bool = True
    var: bool = True


class BispectraSettings(NMBaseModel):
    f1s: FrequencyRange = FrequencyRange(5, 35)
    f2s: FrequencyRange = FrequencyRange(5, 35)
    compute_features_for_whole_fband_range: bool = True
    frequency_bands: list[str] = ["theta", "alpha", "low_beta", "high_beta"]

    components: BispectraComponents = BispectraComponents()
    bispectrum_features: BispectraFeatures = BispectraFeatures()

    @field_validator("f1s", "f2s")
    def test_range(cls, filter_range):
        assert (
            filter_range[1] > filter_range[0]
        ), f"Second frequency range value must be higher than the first one, got {filter_range}"
        return filter_range

    @field_validator("frequency_bands")
    def fbands_spaces_to_underscores(cls, frequency_bands):
        return [f.replace(" ", "_") for f in frequency_bands]


FEATURE_DICT: dict[str, Callable] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "var": np.nanvar,
}

COMPONENT_DICT: dict[str, Callable] = {
    "real": lambda obj: getattr(obj, "real"),
    "imag": lambda obj: getattr(obj, "imag"),
    "absolute": np.abs,
    "phase": np.angle,
}


class Bispectra(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.frequency_ranges_hz = settings.frequency_ranges_hz
        self.settings: BispectraSettings = settings.bispectrum_settings

        assert all(
            f_band_bispectrum in settings.frequency_ranges_hz
            for f_band_bispectrum in self.settings.frequency_bands
        ), (
            "Bispectrum selected frequency bands don't match those specified in settings['frequency_ranges_hz']."
            f" Bispectrum frequency bands: {self.settings.frequency_bands},"
            f" specified frequency_ranges_hz: {settings.frequency_ranges_hz}"
        )

        self.used_features = self.settings.bispectrum_features.get_enabled()

        self.min_freq = min(
            self.settings.f1s.frequency_low_hz, self.settings.f2s.frequency_low_hz
        )
        self.max_freq = max(
            self.settings.f1s.frequency_high_hz, self.settings.f2s.frequency_high_hz
        )

    def calc_feature(self, data: np.ndarray) -> dict:
        """Calculate bispectrum features using the Rust implementation.
        Detecting the quadratic phase coupling between distinct frequency components in neural signals."""
        overall_start_time = time.time()
        results = self.calc_feature_rust(data)
        overall_end_time = time.time()
        print(f"Batch calculation took {overall_end_time - overall_start_time:.5f} seconds")
        return results

    def calc_feature_python(self, data: np.ndarray) -> dict:
        """Calculate bispectrum features using the Python implementation."""
        from pybispectra import compute_fft, WaveShape

        # start_time = time.time()

        fft_coeffs, freqs = compute_fft(
            data=np.expand_dims(data, axis=0),
            sampling_freq=self.sfreq,
            n_points=data.shape[1],
            verbose=False,
        )

        f_spectrum_range = freqs[
            np.logical_and(freqs >= self.min_freq, freqs <= self.max_freq)
        ]

        waveshape = WaveShape(
            data=fft_coeffs,
            freqs=freqs,
            sampling_freq=self.sfreq,
            verbose=False,
        )

        waveshape.compute(
            f1s=tuple(self.settings.f1s),  # type: ignore
            f2s=tuple(self.settings.f2s),  # type: ignore
        )
        bispectrum_results = waveshape.results.get_results(copy=False)

        feature_results = {}
        for ch_idx, ch_name in enumerate(self.ch_names):
            bispectrum = bispectrum_results[ch_idx]

            for component in self.settings.components.get_enabled():
                spectrum_ch = COMPONENT_DICT[component](bispectrum)

                for fb in self.settings.frequency_bands:
                    range_ = (f_spectrum_range >= self.frequency_ranges_hz[fb][0]) & (
                        f_spectrum_range <= self.frequency_ranges_hz[fb][1]
                    )
                    data_bs = spectrum_ch[np.ix_(range_, range_)]

                    for bispectrum_feature in self.used_features:
                        feature_results[
                            f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_{fb}"
                        ] = FEATURE_DICT[bispectrum_feature](data_bs)

                if self.settings.compute_features_for_whole_fband_range:
                    for bispectrum_feature in self.used_features:
                        feature_results[
                            f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_whole_fband_range"
                        ] = FEATURE_DICT[bispectrum_feature](spectrum_ch)

        # print(f"Python calculation took {end_time - start_time:.4f} seconds")
        return feature_results

    def calc_feature_rust(self, data: np.ndarray) -> dict:
        """Calculate bispectrum features using the Rust implementation."""
        start_time = time.time()

        N = data.shape[1]
        freqs = np.fft.rfftfreq(N, d=1 / self.sfreq)
        # Adjust frequency indices to match the frequencies of interest
        freq_start = 5
        freq_end = 35
        freqs_of_interest = freqs[freq_start : freq_end + 1]

        bispectrum = rust_features.calculate_bispectra(data)
        bispectrum = bispectrum.astype(np.complex64)

        feature_results = {}
        for ch_idx, ch_name in enumerate(self.ch_names):
            bispectrum_ch = bispectrum[ch_idx]

            for component in self.settings.components.get_enabled():
                spectrum_ch = COMPONENT_DICT[component](bispectrum_ch)

                for fb in self.settings.frequency_bands:
                    fb_range = self.frequency_ranges_hz[fb]
                    range_ = (freqs_of_interest >= fb_range[0]) & (freqs_of_interest <= fb_range[1])

                    data_bs = spectrum_ch[np.ix_(range_, range_)]

                    for bispectrum_feature in self.used_features:
                        feature_value = FEATURE_DICT[bispectrum_feature](data_bs)
                        feature_results[
                            f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_{fb}"
                        ] = feature_value

            if self.settings.compute_features_for_whole_fband_range:
                for bispectrum_feature in self.used_features:
                    feature_value = FEATURE_DICT[bispectrum_feature](spectrum_ch)
                    feature_results[
                        f"{ch_name}_Bispectrum_{component}_{bispectrum_feature}_whole_fband_range"
                    ] = feature_value

        end_time = time.time()
        # print(f"Rust calculation took {end_time - start_time:.4f} seconds")
        return feature_results



    def compare_results(self, data: np.ndarray):
        """Compare the results of the Python and Rust implementations."""
        # Calculate features using both implementations
        results_python = self.calc_feature_python(data)
        results_rust = self.calc_feature_rust(data)

        keys_python = set(results_python.keys())
        keys_rust = set(results_rust.keys())

        if keys_python != keys_rust:
            print("Mismatch in feature keys:")
            print("Keys only in Python results:", keys_python - keys_rust)
            print("Keys only in Rust results:", keys_rust - keys_python)
            return

        all_close = True
        for key in keys_python:
            value_python = results_python[key]
            value_rust = results_rust[key]
            if not np.allclose(value_python, value_rust, atol=1e-6):
                print(f"Mismatch in {key}:")
                print(f"  Python value: {value_python}")
                print(f"  Rust value:   {value_rust}")
                all_close = False

        if all_close:
            print("All feature results match between Python and Rust implementations.")
        else:
            print("Some feature results do not match between Python and Rust implementations.")
