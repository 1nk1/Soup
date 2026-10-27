"""The train pre-flight refuses only inside the range its coefficients were fitted on.

The VRAM estimate (``utils/vram_estimate.py``) is fitted on dense models of 3B and
below on one 6 GB card (``benchmarks/gate-2049-resident-vram.md``). Outside that range
(a larger model, a mixture-of-experts model, ``medium`` / ``selective`` checkpointing,
a fused-loss flag, ``backend: unsloth``) an estimate that does not fit is printed as a
warning and the run continues; inside it the refusal stays and names
``--allow-oom-attempt``.

No GPU here: the gate is called with a card size, and the model's ``config.json`` is
replaced by the dict a Hub lookup would return.
"""

from __future__ import annotations

import io

import pytest
import typer
import yaml
from rich.console import Console

from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TEMPLATES
from soup_cli.utils import hardware_fit, model_shape
from tests.conftest import strip_ansi

# Llama-3.1-8B: 8.03B parameters, a 128,256-token vocabulary.
_LLAMA_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}
_LLAMA_8B_PARAMS = 8_030_261_248

# Qwen2.5-0.5B: 0.49B parameters, inside the calibrated range.
_SMALL = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}
_SMALL_PARAMS = 494_032_768

#: A 6 GB card: too small for either model once the logits are priced.
SIX_GB_CARD = {"memory_total_bytes": int(6e9)}
BIG_CARD = {"memory_total_bytes": int(10**15)}


def _plain(text: str) -> str:
    """ANSI stripped and whitespace collapsed, as the CLI-output assertions need."""
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def shape_of(monkeypatch):
    """Pick the model the Hub lookup 'finds': ``shape_of(config, params)``."""

    def pick(config: dict, params: int) -> None:
        monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(config), None))
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: params)

    return pick


@pytest.fixture
def screen(monkeypatch):
    """The pre-flight's console, rendered wide enough that no line wraps."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return lambda: _plain(buffer.getvalue())


def _yaml(*, backend: str | None = None, **training) -> str:
    document = {
        "base": "someone/model",
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 2048},
        "training": {"quantization": "4bit", "batch_size": 4, **training},
    }
    if backend:
        document["backend"] = backend
    return yaml.safe_dump(document)


def _run(cfg_text: str, *, card=SIX_GB_CARD, allow_oom_attempt=False) -> int | None:
    """Run the pre-flight; the exit code if it refused, else ``None``."""
    try:
        _hardware_fit_preflight(
            load_config_from_string(cfg_text), card, allow_oom_attempt=allow_oom_attempt
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


OUTSIDE = "outside the calibrated range ("
OVERSTATE = "the estimate may overstate the peak"


class TestInsideTheRange:
    def test_a_dense_model_of_3b_and_below_is_still_refused(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml()) == 1
        text = screen()
        assert "Hardware-fit gate" in text
        assert "Predicted peak VRAM" in text

    def test_the_refusal_names_the_override_and_the_range(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        _run(_yaml())
        text = screen()
        assert "inside the calibrated range (dense models of 3B and below" in text
        assert "one 6 GB card" in text
        assert "--allow-oom-attempt launches it anyway" in text
        assert OUTSIDE not in text

    def test_full_gradient_checkpointing_is_inside(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(gradient_checkpointing=True)) == 1

    def test_allow_oom_attempt_warns_and_launches_without_the_range_line(
        self, shape_of, screen
    ):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(), allow_oom_attempt=True) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "--allow-oom-attempt set: launching anyway." in text
        assert OUTSIDE not in text

    def test_a_card_that_fits_is_let_through_without_a_word(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(), card=BIG_CARD) is None
        assert screen() == ""


class TestOutsideTheRange:
    def _assert_warned(self, screen, reason: str) -> None:
        text = screen()
        assert "Hardware-fit warning" in text
        assert "Hardware-fit gate" not in text
        assert "Predicted peak VRAM" in text
        assert OUTSIDE in text
        assert reason in text
        assert OVERSTATE in text
        assert "launching anyway" in text

    def test_a_model_above_3b_is_warned_about_and_launched(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        assert _run(_yaml(gradient_checkpointing=True)) is None
        self._assert_warned(screen, "8.0B parameters")

    def test_the_review_case_batch_4_on_a_24_gib_card_is_not_refused(self, shape_of, screen):
        # Llama-3.1-8B, 4-bit, 2,048 tokens, batch 4, checkpointing: priced at 29 GB.
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        card = {"memory_total_bytes": int(24 * 1024**3)}
        assert _run(_yaml(gradient_checkpointing=True), card=card) is None
        self._assert_warned(screen, "8.0B parameters")

    def test_a_mixture_of_experts_model_is_warned_about(self, shape_of, screen):
        shape_of({**_SMALL, "num_local_experts": 8}, _SMALL_PARAMS)
        assert _run(_yaml()) is None
        self._assert_warned(screen, "mixture-of-experts")

    @pytest.mark.parametrize("tier", ["medium", "selective"])
    def test_medium_and_selective_checkpointing_are_warned_about(
        self, shape_of, screen, tier
    ):
        shape_of(_SMALL, _SMALL_PARAMS)
        card = {"memory_total_bytes": int(2e9)}
        assert _run(_yaml(gradient_checkpointing=tier), card=card) is None
        self._assert_warned(screen, f"{tier} gradient checkpointing")

    @pytest.mark.parametrize("flag", ["use_liger", "use_cut_ce"])
    def test_a_fused_loss_flag_is_warned_about(self, shape_of, screen, flag):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(**{flag: True})) is None
        self._assert_warned(screen, flag)

    def test_backend_unsloth_is_warned_about(self, shape_of, screen):
        shape_of(_SMALL, _SMALL_PARAMS)
        assert _run(_yaml(backend="unsloth")) is None
        self._assert_warned(screen, "backend unsloth")

    def test_every_reason_is_listed_when_several_apply(self, shape_of, screen):
        shape_of({**_LLAMA_8B, "num_local_experts": 8}, _LLAMA_8B_PARAMS)
        assert _run(_yaml(use_liger=True, gradient_checkpointing="medium")) is None
        text = screen()
        for reason in (
            "8.0B parameters",
            "mixture-of-experts",
            "medium gradient checkpointing",
            "use_liger",
        ):
            assert reason in text

    def test_the_warning_is_the_same_with_allow_oom_attempt(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        assert _run(_yaml(), allow_oom_attempt=True) is None
        self._assert_warned(screen, "8.0B parameters")


class TestTheRangeCheck:
    """``outside_calibrated_range`` is the one place the envelope is decided."""

    def _inp(self, config=_SMALL, params=_SMALL_PARAMS, **fields):
        shape = model_shape.shape_from_config(
            config, params=params, params_source="safetensors"
        )
        base = dict(
            params_b=params / 1e9,
            seq_len=2048,
            batch_size=1,
            optimizer="adamw_torch",
            quant="4bit",
            peft="qlora",
            gradient_checkpointing=False,
            shape=shape,
        )
        return hardware_fit.HardwareFitInput(**{**base, **fields})

    def test_a_small_dense_run_is_inside(self):
        assert hardware_fit.outside_calibrated_range(self._inp()) == ()

    def test_the_bound_is_between_3b_and_the_next_size_up(self):
        three_b = self._inp(params=3_090_000_000, params_b=3.09)
        four_b = self._inp(params=4_020_000_000, params_b=4.02)
        assert hardware_fit.outside_calibrated_range(three_b) == ()
        assert len(hardware_fit.outside_calibrated_range(four_b)) == 1

    def test_full_checkpointing_and_none_are_inside(self):
        for value in (False, True, "full"):
            assert not hardware_fit.outside_calibrated_range(
                self._inp(gradient_checkpointing=value)
            )

    def test_a_shape_with_no_config_is_inside_when_small(self):
        inp = hardware_fit.HardwareFitInput(
            params_b=1.0, seq_len=2048, batch_size=1, optimizer="adamw_torch",
            quant="none", peft="lora", gradient_checkpointing=False,
        )
        assert hardware_fit.outside_calibrated_range(inp) == ()

    def test_the_backend_is_a_reason_only_when_it_is_unsloth(self):
        inp = self._inp()
        assert hardware_fit.outside_calibrated_range(inp, backend="transformers") == ()
        assert hardware_fit.outside_calibrated_range(inp, backend=None) == ()
        assert hardware_fit.outside_calibrated_range(inp, backend="mlx") == ()
        assert hardware_fit.outside_calibrated_range(inp, backend="unsloth") == (
            "backend unsloth",
        )

    def test_the_reasons_read_as_a_phrase(self):
        reasons = hardware_fit.outside_calibrated_range(
            self._inp(config={**_SMALL, "num_experts": 64}, fused_loss=("use_liger",)),
            backend="unsloth",
        )
        assert reasons == (
            "a mixture-of-experts model",
            "fused-loss flag use_liger",
            "backend unsloth",
        )


class TestTheLongcontextTemplate:
    """The shipped template is an 8B model at 131,072 tokens: priced far above any card."""

    def _template(self, *, liger: bool) -> str:
        text = TEMPLATES["longcontext"]
        assert "# use_liger: true" in text
        return text.replace("# use_liger: true", "use_liger: true") if liger else text

    @pytest.mark.parametrize("liger", [True, False])
    def test_it_is_never_refused_on_a_24_gib_card(self, shape_of, screen, liger):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        card = {"memory_total_bytes": int(24 * 1024**3)}
        assert _run(self._template(liger=liger), card=card) is None
        text = screen()
        assert "Hardware-fit warning" in text
        assert "Hardware-fit gate" not in text

    def test_with_the_fused_loss_on_the_warning_names_it(self, shape_of, screen):
        shape_of(_LLAMA_8B, _LLAMA_8B_PARAMS)
        _run(self._template(liger=True), card={"memory_total_bytes": int(80e9)})
        text = screen()
        assert "use_liger" in text
        assert "overstates the peak" in text

    def test_the_header_says_what_the_pre_flight_does_and_how_to_turn_on_the_fused_loss(self):
        header = [
            line for line in TEMPLATES["longcontext"].splitlines() if line.startswith("#")
        ]
        text = _plain(" ".join(header))
        assert "pre-flight" in text
        assert "warning" in text
        assert "use_liger" in text
        assert "soup-cli[liger]" in text
