import multiprocessing
import multiprocessing.synchronize
import pathlib
import pickle
import queue
import tkinter
import tkinter.filedialog
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pylsl

import realtime_decoding

from .helpers import _PathLike

_timezone = timezone.utc


class Decoder(multiprocessing.Process):
    """Decode motor intention in real time."""

    def __init__(
        self,
        queue_decoding: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
        interval: float,
        out_dir: _PathLike,
        verbose: bool,
    ) -> None:
        super().__init__(name="DecodingThread")
        self.queue_decoding = queue_decoding
        self.queue_feat = queue_features
        self.interval = interval
        self.verbose = verbose
        self.out_dir = pathlib.Path(out_dir)

        self._threshold: float = 0.5

        filename = tkinter.filedialog.askopenfilename(
            title="Select model",
            filetypes=(
                ("pickle files", ["*.p", "*.pkl", "*.pickle"]),
                ("All files", "*.*"),
            ),
        )
        self.filename = pathlib.Path(filename)

        # self._model = sklearn.dummy.DummyClassifier(strategy="stratified")
        with open(self.filename, "rb") as file:
            self._model = pickle.load(file)
        self._save_model()

    def _save_model(self) -> None:
        with open(self.out_dir / self.filename.name, "wb") as file:
            pickle.dump(self._model, file)

    def clear_queue(self) -> None:
        for q in (self.queue_feat, self.queue_decoding):
            realtime_decoding.clear_queue(q)

    def run(self) -> None:
        labels = ["Prediction", "Probability", "Threshold"]

        def _predict(data) -> None:
            y = self._model.predict(np.expand_dims(data.to_numpy(), axis=0))

            timestamp = np.datetime64(datetime.now(_timezone), "ns")
            output = pd.DataFrame(
                [[y >= self._threshold, y, self._threshold]],
                columns=labels,
                index=[timestamp],
            )
            self.outlet.push_sample(
                x=list(output.to_numpy().squeeze()),
                timestamp=timestamp.astype(float),
            )
            # try:
            #     self.queue_decoding.put(output, timeout=self.interval)
            # except queue.Full:
            #     print("Decoding queue Full. Skipping sample.")
            # try:
            #     self.queue_decoding.get(block=False)
            # except queue.Empty:
            #     print("Decoding queue empty. Skipping sample.")

        info = pylsl.StreamInfo(
            name="Decoding",
            type="EEG",
            channel_count=3,
            channel_format="double64",
            source_id="decoding_1",
        )
        channels = info.desc().append_child("channels")
        for label in labels:
            channels.append_child("channel").append_child_value("label", label)
        self.outlet = pylsl.StreamOutlet(info)
        while True:
            try:
                sample = self.queue_feat.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                if self.verbose:
                    print("Got features.")
                if sample is None:
                    break
                _predict(sample)
        try:
            self.queue_decoding.put(None, timeout=3.0)
        except queue.Full:
            pass
        self.clear_queue()
        print(f"Terminating: {self.name}")
