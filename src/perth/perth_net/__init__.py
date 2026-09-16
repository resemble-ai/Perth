from perth.config import Config

PREPACKAGED_MODELS_DIR = Config.get_default_models_dir()


def __getattr__(name: str):
    """Load the implicit watermarker only when requested.

    Importing ``perth.perth_net`` or ``PerthNet`` must not pull in librosa/soxr.
    """
    if name == "PerthImplicitWatermarker":
        from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

        return PerthImplicitWatermarker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
