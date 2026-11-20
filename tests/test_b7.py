"""Milestone B7: which dates are kept, the network's inputs, padding of a batch, and U-TAE ignoring padded dates
(no data, no network files)."""

from datetime import date

import numpy as np
import pytest


def test_dates_kept_per_window():
    from core.data.impactmesh_ts import select
    dates = [f"2021-07-{d:02d}" for d in range(1, 21)]
    clear = np.linspace(0.0, 0.95, 20)
    idx = select(dates, clear, (date(2021, 7, 1), date(2021, 7, 20)))
    assert len(idx) == 12 and idx == sorted(idx) and idx[0] == 8          # the 12 clearest, in time order
    assert 0 not in select(dates, clear, (date(2021, 7, 1), date(2021, 7, 5)))   # 0 % clear: dropped


def test_inputs_have_a_clear_channel_and_no_nan():
    pytest.importorskip("torch")
    from experiments.b7_train import inputs
    x = np.full((2, 7, 4, 4), 0.1, np.float32)
    x[0, :, 0, 0] = np.nan
    a = inputs(x)
    assert a.shape == (2, 8, 4, 4) and np.isfinite(a).all()
    assert a[0, 7, 0, 0] == 0 and a[0, 7, 1, 1] == 1 and a[0, 0, 0, 0] == 0


def test_batch_pads_to_the_longest_series():
    pytest.importorskip("torch")
    from experiments.b7_train import batch
    y = np.zeros((4, 4), np.int64)
    items = [(np.ones((3, 8, 4, 4), np.float32), np.array([-5, 2, 9]), np.array([0, 1, 1]), y),
             (np.ones((1, 8, 4, 4), np.float32), np.array([3]), np.array([1]), y)]
    X, D, P, pad, Y = batch(items)
    assert X.shape == (2, 3, 8, 4, 4) and pad.tolist() == [[False] * 3, [False, True, True]]
    assert X[1, 1:].abs().sum() == 0


def test_utae_ignores_padded_dates():
    torch = pytest.importorskip("torch")
    from core.models.utae import UTAE
    torch.manual_seed(0)
    m = UTAE().eval()
    x = torch.randn(1, 4, 8, 32, 32)
    days, per = torch.tensor([[-30, -10, 5, 0]]), torch.tensor([[0, 0, 1, 1]])
    pad = torch.tensor([[False, False, False, True]])
    y1 = m(x, days, per, pad)
    x[0, 3] = 50.0
    assert torch.allclose(m(x, days, per, pad), y1, atol=1e-5)
