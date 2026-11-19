from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest
import scipy.stats

import tea_tasting.config
import tea_tasting.data
import tea_tasting.datasets
import tea_tasting.metrics.resampling


if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy.typing as npt


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, rng=42)

@pytest.fixture
def data_gran(data_arrow: pa.Table) -> dict[Hashable, pa.Table]:
    return tea_tasting.data.read_granular(
        data_arrow,
        ("sessions", "orders", "revenue"),
        variant="variant",
    )


def test_bootstrap_init_default() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.columns == "a"
    assert metric.statistic == np.mean
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.n_resamples == tea_tasting.config.get_config("n_resamples")
    assert metric.method == "bca"
    assert metric.batch is None
    assert metric.nan_policy == tea_tasting.config.get_config("nan_policy")
    assert metric.rng is None

def test_bootstrap_init_custom() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        np.mean,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=1000,
        method="basic",
        batch=100,
        nan_policy="omit",
        rng=42,
    )
    assert metric.columns == ("a", "b")
    assert metric.statistic == np.mean
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.n_resamples == 1000
    assert metric.method == "basic"
    assert metric.batch == 100
    assert metric.nan_policy == "omit"
    assert metric.rng == 42


def test_bootstrap_cols() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.cols == ("a",)

    metric = tea_tasting.metrics.resampling.Bootstrap(("a", "b"), np.mean)
    assert metric.cols == ("a", "b")


def test_bootstrap_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("sessions", np.mean)
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)


def test_bootstrap_analyze_default(data_gran: dict[Hashable, pa.Table]) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "revenue",
        np.mean,
        n_resamples=100,
        rng=42,
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(5.029811320754717)
    assert result.treatment == pytest.approx(5.43)
    assert result.effect_size == pytest.approx(0.4001886792452831)
    assert result.effect_size_ci_lower == pytest.approx(-3.595551144566696)
    assert result.effect_size_ci_upper == pytest.approx(6.426573101094806)
    assert result.rel_effect_size == pytest.approx(0.07956335809137971)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.6261291499213544)
    assert result.rel_effect_size_ci_upper == pytest.approx(1.7431421649724208)

def test_bootstrap_analyze_multiple_columns(
    data_gran: dict[Hashable, pa.Table],
) -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("orders", "sessions"),
        ratio_of_means,
        n_resamples=100,
        rng=42,
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(0.2857142857142857)
    assert result.treatment == pytest.approx(0.20224719101123595)
    assert result.effect_size == pytest.approx(-0.08346709470304975)
    assert result.effect_size_ci_lower == pytest.approx(-0.21061531484819673)
    assert result.effect_size_ci_upper == pytest.approx(0.08491730472112008)
    assert result.rel_effect_size == pytest.approx(-0.2921348314606741)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.653616879375374)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.3635392812388218)

def test_bootstrap_analyze_division_by_zero(
    data_gran: dict[Hashable, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "orders",
        np.median,
        n_resamples=100,
        rng=42,
        method="basic",
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == 0
    assert result.treatment == 0
    assert result.effect_size == 0
    assert result.effect_size_ci_lower == 0
    assert result.effect_size_ci_upper == 0
    assert np.isnan(result.rel_effect_size)
    assert np.isnan(result.rel_effect_size_ci_lower)
    assert np.isnan(result.rel_effect_size_ci_upper)

def test_bootstrap_analyze_nan_policy_raise() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        nan_policy="raise",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    with pytest.raises(ValueError, match=r"Input contains nan\."):
        metric.analyze(data, 0, 1)

def test_bootstrap_analyze_nan_policy_omit() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, float("nan")]}),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


def test_bootstrap_analyze_nan_policy_omit_multiple_columns() -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        ratio_of_means,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({
            "a": [2.0, 4.0, float("nan"), 6.0],
            "b": [1.0, 2.0, 4.0, float("nan")],
        }),
        1: pa.table({
            "a": [6.0, 9.0, 8.0, float("nan")],
            "b": [2.0, 3.0, float("nan"), 3.0],
        }),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


def test_bootstrap_analyze_nan_policy_omit_empty() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        nan_policy="omit",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [float("nan"), float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert np.isnan(result.control)
    assert np.isnan(result.treatment)
    assert np.isnan(result.effect_size)
    assert np.isnan(result.effect_size_ci_lower)
    assert np.isnan(result.effect_size_ci_upper)
    assert np.isnan(result.rel_effect_size)
    assert np.isnan(result.rel_effect_size_ci_lower)
    assert np.isnan(result.rel_effect_size_ci_upper)


def test_quantile(data_gran: dict[Hashable, pa.Table]) -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "revenue",
        q=0.8,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=100,
        rng=42,
    )
    assert metric.column == "revenue"
    assert metric.q == 0.8
    assert metric.nan_policy == tea_tasting.config.get_config("nan_policy")
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(11.972000000000001)
    assert result.treatment == pytest.approx(6.2820000000000045)
    assert result.effect_size == pytest.approx(-5.689999999999997)
    assert result.effect_size_ci_lower == pytest.approx(-11.43539999999998)
    assert result.effect_size_ci_upper == float("inf")
    assert result.rel_effect_size == pytest.approx(-0.47527564316739024)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.9643443256453782)
    assert result.rel_effect_size_ci_upper == float("inf")


def test_quantile_analyze_nan_policy_propagate() -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "value",
        q=0.5,
        n_resamples=10,
        method="basic",
        rng=42,
        nan_policy="propagate",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, 6.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert np.isnan(result.control)
    assert result.treatment == 4
    assert np.isnan(result.effect_size)
    assert np.isnan(result.rel_effect_size)


def test_quantile_analyze_nan_policy_omit() -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "value",
        q=0.5,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, float("nan")]}),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


@pytest.fixture
def data_paired() -> tuple[pa.Table, pa.Table]:
    control = pa.table({"pair_id": [0, 1, 2, 3, 4], "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    treatment = pa.table(
        {"pair_id": [0, 1, 2, 3, 4], "value": [2.0, 4.0, 6.0, 8.0, 10.0]},
    )
    return control, treatment


def test_bootstrap_paired(data_paired: tuple[pa.Table, pa.Table]) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=100,
        rng=42,
        method="basic",
    )
    control, treatment = data_paired
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(3)
    assert result.treatment == pytest.approx(6)
    assert result.effect_size == pytest.approx(3)
    assert result.rel_effect_size == pytest.approx(1)

    # Reference: bootstrapping the paired differences with shared indices
    # must produce the same distribution as the paired bootstrap.
    diffs = treatment["value"].to_numpy() - control["value"].to_numpy()
    reference = scipy.stats.bootstrap(
        (diffs,),
        lambda sample, axis: np.stack((
            np.mean(sample, axis=axis),
            np.divide(np.mean(sample, axis=axis), np.mean(diffs)),
        )),
        n_resamples=100,
        axis=0,
        method="basic",
        rng=42,
    )
    assert result.effect_size_ci_lower == pytest.approx(
        reference.confidence_interval.low[0],
    )
    assert result.effect_size_ci_upper == pytest.approx(
        reference.confidence_interval.high[0],
    )


def test_bootstrap_paired_rng_override(
    data_paired: tuple[pa.Table, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=100,
        rng=1,
        method="basic",
    )
    control, treatment = data_paired
    result1 = metric.analyze_paired(control, treatment, "pair_id", rng=42)
    result2 = metric.analyze_paired(control, treatment, "pair_id", rng=42)
    result3 = metric.analyze_paired(control, treatment, "pair_id")
    assert result1 == result2
    assert result1 != result3


def test_bootstrap_paired_multiple_columns() -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("numer", "denom"),
        ratio_of_means,
        n_resamples=100,
        rng=42,
        method="basic",
    )
    control = pa.table({
        "pair_id": [0, 1, 2, 3],
        "numer": [1.0, 2.0, 3.0, 4.0],
        "denom": [2.0, 2.0, 2.0, 2.0],
    })
    treatment = pa.table({
        "pair_id": [0, 1, 2, 3],
        "numer": [3.0, 4.0, 5.0, 6.0],
        "denom": [4.0, 4.0, 4.0, 4.0],
    })
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert result.control == pytest.approx(1.25)
    assert result.treatment == pytest.approx(1.125)
    assert result.effect_size == pytest.approx(-0.125)


def test_bootstrap_paired_nan_policy_omit() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    control = pa.table({"pair_id": [0, 1, 2], "value": [1.0, float("nan"), 5.0]})
    treatment = pa.table({"pair_id": [0, 1, 2], "value": [2.0, 2.0, 2.0]})
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert result.control == pytest.approx(3)
    assert result.treatment == pytest.approx(2)
    assert result.effect_size == pytest.approx(-1)


def test_bootstrap_paired_nan_policy_omit_multiple_columns() -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        ratio_of_means,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    control = pa.table({
        "pair_id": [0, 1, 2, 3],
        "a": [2.0, float("nan"), 6.0, 9.0],
        "b": [1.0, 2.0, 2.0, 3.0],
    })
    treatment = pa.table({
        "pair_id": [0, 1, 2, 3],
        "a": [6.0, 9.0, float("nan"), 12.0],
        "b": [2.0, 3.0, 3.0, 4.0],
    })
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert result.control == pytest.approx(2.75)
    assert result.treatment == pytest.approx(3)


def test_bootstrap_paired_nan_policy_raise() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        nan_policy="raise",
    )
    control = pa.table({"pair_id": [0, 1], "value": [1.0, float("nan")]})
    treatment = pa.table({"pair_id": [0, 1], "value": [2.0, 3.0]})
    with pytest.raises(ValueError, match=r"Input contains nan\."):
        metric.analyze_paired(control, treatment, "pair_id")


def test_bootstrap_paired_nan_policy_propagate() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=10,
        method="basic",
        nan_policy="propagate",
        rng=42,
    )
    control = pa.table({"pair_id": [0, 1], "value": [1.0, float("nan")]})
    treatment = pa.table({"pair_id": [0, 1], "value": [2.0, 3.0]})
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert np.isnan(result.control)
    assert np.isnan(result.effect_size)


def test_bootstrap_paired_empty() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("value", np.mean)
    empty = pa.table({"pair_id": pa.array([], type=pa.int64()), "value": pa.array([])})
    result = metric.analyze_paired(empty, empty, "pair_id")
    assert np.isnan(result.control)
    assert np.isnan(result.treatment)
    assert np.isnan(result.effect_size)
    assert np.isnan(result.rel_effect_size)


def test_bootstrap_paired_raises_for_missing_pair_id(
    data_paired: tuple[pa.Table, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("value", np.mean)
    control, treatment = data_paired
    with pytest.raises(ValueError, match="must include the 'pair_id' column"):
        metric.analyze_paired(
            control.drop_columns("pair_id"),
            treatment,
            "pair_id",
        )


def test_bootstrap_paired_raises_for_length_mismatch(
    data_paired: tuple[pa.Table, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("value", np.mean)
    control, treatment = data_paired
    with pytest.raises(ValueError, match="same number of rows"):
        metric.analyze_paired(control, treatment.slice(0, 2), "pair_id")


def test_bootstrap_paired_raises_for_misaligned(
    data_paired: tuple[pa.Table, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("value", np.mean)
    control, treatment = data_paired
    misaligned = treatment.set_column(
        0,
        "pair_id",
        pa.array([1, 0, 2, 3, 4], type=pa.int64()),
    )
    with pytest.raises(ValueError, match="not aligned"):
        metric.analyze_paired(control, misaligned, "pair_id")


def test_bootstrap_not_paired_by_default(
    data_gran: dict[Hashable, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("revenue", np.mean)
    assert metric.paired is True
    result = metric.analyze_granular(data_gran[0], data_gran[1])
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)


def test_quantile_paired() -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "value",
        q=0.5,
        n_resamples=100,
        method="basic",
        rng=42,
    )
    control = pa.table({"pair_id": [0, 1, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]})
    treatment = pa.table({"pair_id": [0, 1, 2, 3], "value": [2.0, 3.0, 4.0, 5.0]})
    result = metric.analyze_paired(control, treatment, "pair_id")
    assert result.control == pytest.approx(2.5)
    assert result.treatment == pytest.approx(3.5)
    assert result.effect_size == pytest.approx(1)

