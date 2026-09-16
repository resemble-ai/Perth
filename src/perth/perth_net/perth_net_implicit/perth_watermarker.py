import torch
import numpy as np

from .model.perth_net import PerthNet
from perth.watermarker import WatermarkerBase


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    return x.to(dtype=torch.float, device=device)


def _resample(signal, orig_sr, target_sr, *, res_type=None):
    """Resample with torchaudio when possible; librosa is an optional fallback."""
    try:
        import torch
        from torchaudio.functional import resample as torchaudio_resample

        tensor = signal if isinstance(signal, torch.Tensor) else torch.as_tensor(signal)
        was_numpy = not isinstance(signal, torch.Tensor)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        out = torchaudio_resample(tensor.float(), orig_sr, target_sr)
        if squeezed:
            out = out.squeeze(0)
        return out.numpy() if was_numpy else out
    except Exception:
        from librosa import resample

        kwargs = {} if res_type is None else {"res_type": res_type}
        return resample(signal, orig_sr=orig_sr, target_sr=target_sr, **kwargs)


class PerthImplicitWatermarker(WatermarkerBase):
    def __init__(
        self,
        run_name: str = "implicit",
        models_dir=None,
        device="cpu",
        perth_net=None,
    ):
        assert (run_name is None) or (perth_net is None)
        if perth_net is None:
            if not models_dir:
                from perth.config import Config

                models_dir = Config.get_default_models_dir()
            self.perth_net = PerthNet.load(run_name, models_dir).to(device)
        else:
            self.perth_net = perth_net.to(device)

    def apply_watermark(self, signal, sample_rate, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        signal = (
            _resample(
                signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate
            )
            if change_rate
            else signal
        )

        # split signal into magnitude and phase
        signal = _to_tensor(signal, self.perth_net.device)
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        return (
            _resample(
                wm_signal, orig_sr=self.perth_net.hp.sample_rate, target_sr=sample_rate
            )
            if change_rate
            else wm_signal
        )

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = _resample(
                wm_signal,
                orig_sr=sample_rate,
                target_sr=self.perth_net.hp.sample_rate,
                res_type="polyphase",
            )
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clip(0.0, 1.0)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        return wmark_pred.detach().cpu().numpy()
