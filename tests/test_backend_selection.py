"""Tests for GPU enumeration/selection and the streaming decode parameters.

These exercise the pure-Python decision logic only -- no Vulkan loader, no
whisper.cpp model -- so they run in environments with no GPU at all.
"""
import pytest

from narrator.asr.backend import (
    VulkanDevice,
    audio_ctx_for_seconds,
    select_gpu_index,
)

DISCRETE = 2
INTEGRATED = 1

RX = VulkanDevice(index=1, name="AMD Radeon RX 6700 XT", device_type=DISCRETE)
IGPU = VulkanDevice(index=0, name="Intel HD Graphics 610", device_type=INTEGRATED)


def test_auto_prefers_discrete_gpu_over_integrated():
    """The failure this guards against is silent: whisper.cpp defaults to
    gpu_device=0, so a machine whose iGPU enumerates first runs the model on
    the slow adapter while still reporting that it is 'using the GPU'.
    """
    assert select_gpu_index([IGPU, RX], "auto") == 1


def test_auto_falls_back_to_first_device_when_none_are_discrete():
    assert select_gpu_index([IGPU], "auto") == 0


def test_auto_falls_back_to_zero_when_enumeration_found_nothing():
    assert select_gpu_index([], "auto") == 0


def test_explicit_index_overrides_auto_preference():
    assert select_gpu_index([IGPU, RX], 0) == 0


def test_explicit_out_of_range_index_is_passed_through(caplog):
    """whisper.cpp's own init is authoritative, so an unrecognized index is
    warned about but still forwarded rather than silently rewritten."""
    assert select_gpu_index([IGPU], 7) == 7
    assert "not among" in caplog.text


@pytest.mark.parametrize(
    "seconds,expected_range",
    [
        (1.0, (256, 256)),      # clamped up to the floor
        (20.0, (1000, 1100)),   # ~1000 frames + margin
        (60.0, (1500, 1500)),   # clamped down to whisper's native context
    ],
)
def test_audio_ctx_scales_with_window_and_clamps(seconds, expected_range):
    value = audio_ctx_for_seconds(seconds)
    assert expected_range[0] <= value <= expected_range[1]


def test_audio_ctx_never_exceeds_whisper_native_context():
    assert audio_ctx_for_seconds(10_000.0) == 1500
