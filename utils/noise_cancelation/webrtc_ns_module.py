from __future__ import annotations

import audioop
import logging
from dataclasses import dataclass
from typing import Optional

from livekit import rtc


logger = logging.getLogger(__name__)


class NoiseCancellationError(RuntimeError):
    pass


@dataclass
class WebRTCNoiseGainConfig:
    enabled: bool = True
    sample_rate: int = 16000
    auto_gain_dbfs: int = 0
    noise_suppression_level: int = 2
    preserve_channels: bool = True
    passthrough_on_error: bool = True
    passthrough_on_unsupported_frame: bool = True


class WebRTCNoiseGainCanceller(rtc.FrameProcessor[rtc.AudioFrame]):
    """LiveKit audio frame processor backed by ``webrtc-noise-gain``.

    The underlying package only supports 16 kHz, 16-bit PCM, mono audio in
    10 ms chunks (320 bytes). This adapter buffers incoming LiveKit frames,
    converts stereo to mono when needed, and returns the processed audio in the
    original frame shape.
    """

    _BYTES_PER_SAMPLE = 2

    def __init__(self, config: Optional[WebRTCNoiseGainConfig] = None) -> None:
        self.config = config or WebRTCNoiseGainConfig()
        self._enabled = self.config.enabled
        self._closed = False
        self._processor = self._build_processor()
        self._pending_input = bytearray()
        self._pending_output = bytearray()
        self._logged_unsupported = False

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _build_processor(self):
        try:
            from webrtc_noise_gain import AudioProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise NoiseCancellationError(
                "webrtc-noise-gain is not installed. "
                "Add `webrtc-noise-gain` to the environment to use the custom noise canceller."
            ) from exc

        return AudioProcessor(
            int(self.config.auto_gain_dbfs),
            int(self.config.noise_suppression_level),
        )

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self.enabled:
            return frame

        if frame.sample_rate != self.config.sample_rate:
            return self._handle_unsupported_frame(
                frame,
                f"Unsupported sample rate {frame.sample_rate}. "
                f"webrtc-noise-gain requires {self.config.sample_rate} Hz.",
            )

        if frame.num_channels not in (1, 2):
            return self._handle_unsupported_frame(
                frame,
                f"Unsupported channel count {frame.num_channels}. Only mono or stereo are supported.",
            )
        if frame.num_channels == 2 and not self.config.preserve_channels:
            return self._handle_unsupported_frame(
                frame,
                "Stereo input requires preserve_channels=True so the returned frame shape remains stable.",
            )

        frame_bytes = bytes(frame.data)
        expected_output_size = len(frame_bytes)
        chunk_size = self._chunk_size_bytes(frame.num_channels)

        self._pending_input.extend(frame_bytes)
        while len(self._pending_input) >= chunk_size:
            chunk = bytes(self._pending_input[:chunk_size])
            del self._pending_input[:chunk_size]
            processed = self._process_chunk(chunk, frame.num_channels)
            self._pending_output.extend(processed)

        if len(self._pending_output) < expected_output_size:
            missing = expected_output_size - len(self._pending_output)
            fallback = frame_bytes[-missing:]
            self._pending_output.extend(fallback)
            if self._pending_input[:missing] == fallback:
                del self._pending_input[:missing]

        out_bytes = bytes(self._pending_output[:expected_output_size])
        del self._pending_output[:expected_output_size]

        return rtc.AudioFrame(
            data=out_bytes,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _process_chunk(self, chunk: bytes, num_channels: int) -> bytes:
        mono_chunk = chunk
        if num_channels == 2:
            mono_chunk = audioop.tomono(chunk, self._BYTES_PER_SAMPLE, 0.5, 0.5)

        try:
            result = self._processor.Process10ms(mono_chunk)
            cleaned = bytes(result.audio)
        except Exception as exc:  # pragma: no cover - depends on external runtime
            if not self.config.passthrough_on_error:
                raise NoiseCancellationError("webrtc-noise-gain processing failed.") from exc
            logger.warning("Noise cancellation failed, returning original audio: %s", exc)
            return chunk

        if num_channels == 2 and self.config.preserve_channels:
            return audioop.tostereo(cleaned, self._BYTES_PER_SAMPLE, 1.0, 1.0)
        if num_channels == 2:
            return cleaned
        return cleaned

    def _handle_unsupported_frame(self, frame: rtc.AudioFrame, message: str) -> rtc.AudioFrame:
        if not self.config.passthrough_on_unsupported_frame:
            raise NoiseCancellationError(message)
        if not self._logged_unsupported:
            logger.warning("%s Passing audio through unchanged.", message)
            self._logged_unsupported = True
        return frame

    def _chunk_size_bytes(self, num_channels: int) -> int:
        samples_per_10ms = self.config.sample_rate // 100
        return samples_per_10ms * num_channels * self._BYTES_PER_SAMPLE

    def _close(self) -> None:
        self._closed = True
        self._pending_input.clear()
        self._pending_output.clear()


def build_webrtc_noise_canceller(kwargs: Optional[dict] = None) -> WebRTCNoiseGainCanceller:
    config = WebRTCNoiseGainConfig(**(kwargs or {}))
    return WebRTCNoiseGainCanceller(config=config)
