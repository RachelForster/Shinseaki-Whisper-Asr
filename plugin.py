from __future__ import annotations

from pathlib import Path

from sdk.plugin import PluginBase
from sdk.plugin_host_context import PluginHostContext
from sdk.register import PluginCapabilityRegistry

from plugins.whisper_asr.adapters import FasterWhisperAdapter, RealtimeSTTAdapter


class WhisperAsrPlugin(PluginBase):
    """Optional Whisper backends (faster-whisper, RealtimeSTT); heavy dependencies."""

    @property
    def plugin_id(self) -> str:
        return "com.shinsekai.whisper_asr"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    @property
    def priority(self) -> int:
        return 95

    def initialize(
        self,
        register: PluginCapabilityRegistry,
        plugin_root: Path,
        host: PluginHostContext,
    ) -> None:
        register.register_asr_adapter("faster_whisper", FasterWhisperAdapter)
        register.register_asr_adapter("realtime_stt", RealtimeSTTAdapter)
