from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_color=False))

from pocket_tts.models.tts_model import TTSModel  # noqa: E402

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

# Speaker fingerprinting (lazy import to avoid pulling heavy deps at package load):
# from pocket_tts.speaker_id import SpeakerID, SpeakerFingerprint

__all__ = ["TTSModel"]
