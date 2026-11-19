from __future__ import annotations

from typing import TYPE_CHECKING
import unittest.mock

import pytest

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.data
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.mean


if TYPE_CHECKING:
    from collections.abc import Hashable

    import pyarrow as pa


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, covariates=True, rng=42)

@pytest.fixture
def data_aggr(data_arrow: pa.Table) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    cols = (
        "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    )
    return tea_tasting.data.read_aggregates(
        data_arrow,
        aggr_cols=tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=cols,
            var_cols=cols,
            cov_cols=tuple(
                (col0, col1)
                for col0 in cols
                for col1 in cols
                if col0 < col1
            ),
        ),
        variant="variant",
    )

@pytest.fixture
def power_data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(
        n_users=100, covariates=True, rng=42,
        sessions_uplift=0, orders_uplift=0, revenue_uplift=0,
    )

@pytest.fixture
def power_data_aggr(power_data_arrow: pa.Table) -> tea_tasting.aggr.Aggregates:
    cols = (
        "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    )
    return tea_tasting.data.read_aggregates(
        power_data_arrow,
        aggr_cols=tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=cols,
            var_cols=cols,
            cov_cols=tuple(
                (col0, col1)
                for col0 in cols
                for col1 in cols
                if col0 < col1
            ),
        ),
    )


def test_ratio_of_means_init_default() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans("a")
    assert metric.numer == "a"
    assert metric.denom is None
    assert metric.numer_covariate is None
    assert metric.denom_covariate is None
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.equal_var == tea_tasting.config.get_config("equal_var")
    assert metric.use_t == tea_tasting.config.get_config("use_t")
    assert metric.alpha == tea_tasting.config.get_config("alpha")
    assert metric.ratio == tea_tasting.config.get_config("ratio")
    assert metric.power == tea_tasting.config.get_config("power")
    assert metric.effect_size is None
    assert metric.rel_effect_size is None
    assert metric.n_obs is None

def test_ratio_of_means_init_custom() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="a",
        denom="b",
        numer_covariate="c",
        denom_covariate="d",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        alpha=0.1,
        ratio=0.5,
        power=0.75,
        rel_effect_size=0.08,
        n_obs=(5_000, 10_000),
    )
    assert metric.numer == "a"
    assert metric.denom == "b"
    assert metric.numer_covariate == "c"
    assert metric.denom_covariate == "d"
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False
    assert metric.alpha == 0.1
    assert metric.ratio == 0.5
    assert metric.power == 0.75
    assert metric.effect_size is None
    assert metric.rel_effect_size == 0.08
    assert metric.n_obs == (5_000, 10_000)
    metric = tea_tasting.metrics.mean.RatioOfMeans("a", effect_size=(1, 0.2))
    assert metric.effect_size == (1, 0.2)

def test_ratio_of_means_init_config() -> None:
    with tea_tasting.config.config_context(
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        alpha=0.1,
        ratio=0.5,
        power=0.75,
        n_obs=(5_000, 10_000),
    ):
        metric = tea_tasting.metrics.mean.RatioOfMeans("a")
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False
    assert metric.alpha == 0.1
    assert metric.ratio == 0.5
    assert metric.power == 0.75
    assert metric.n_obs == (5_000, 10_000)


def test_ratio_of_means_aggr_cols() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="a",
        denom="b",
        numer_covariate="c",
    )
    aggr_cols = metric.aggr_cols
    assert isinstance(aggr_cols, tea_tasting.aggr.AggrCols)
    assert aggr_cols.has_count is True
    assert set(aggr_cols.mean_cols) == {"a", "b", "c"}
    assert len(aggr_cols.mean_cols) == 3
    assert set(aggr_cols.var_cols) == {"a", "b", "c"}
    assert len(aggr_cols.var_cols) == 3
    assert len(aggr_cols.cov_cols) == 3
    assert set(aggr_cols.cov_cols) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_ratio_of_means_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
    )
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)

def test_ratio_of_means_analyze_basic(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.5660377358490566)
    assert result.treatment == pytest.approx(0.3829787234042553)
    assert result.effect_size == pytest.approx(-0.18305901244480127)
    assert result.effect_size_ci_lower == pytest.approx(-0.5004359360326984)
    assert result.effect_size_ci_upper == pytest.approx(0.13431791114309583)
    assert result.rel_effect_size == pytest.approx(-0.32340425531914896)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.6593351111187646)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.34378920945924096)
    assert result.pvalue == pytest.approx(0.2551230546709908)
    assert result.statistic == pytest.approx(-1.1447678040118034)

def test_ratio_of_means_analyze_ratio_greater_equal_var(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        alternative="greater",
        equal_var=True,
    )
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.2857142857142857)
    assert result.treatment == pytest.approx(0.20224719101123595)
    assert result.effect_size == pytest.approx(-0.08346709470304975)
    assert result.effect_size_ci_lower == pytest.approx(-0.21187246704082818)
    assert result.effect_size_ci_upper == float("inf")
    assert result.rel_effect_size == pytest.approx(-0.2921348314606741)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5852384654309937)
    assert result.rel_effect_size_ci_upper == float("inf")
    assert result.pvalue == pytest.approx(0.8584716347525132)
    assert result.statistic == pytest.approx(-1.0794048813446926)

def test_ratio_of_means_analyze_ratio_less_use_norm(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        alternative="less",
        use_t=False,
    )
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.25572348175909004)
    assert result.treatment == pytest.approx(0.23549786496156158)
    assert result.effect_size == pytest.approx(-0.020225616797528462)
    assert result.effect_size_ci_lower == float("-inf")
    assert result.effect_size_ci_upper == pytest.approx(0.07287939184944564)
    assert result.rel_effect_size == pytest.approx(-0.07909174651619377)
    assert result.rel_effect_size_ci_lower == float("-inf")
    assert result.rel_effect_size_ci_upper == pytest.approx(0.34578466493619153)
    assert result.pvalue == pytest.approx(0.3604265417728255)
    assert result.statistic == pytest.approx(-0.3573188986307722)


def test_ratio_of_means_solve_power_frame(power_data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        rel_effect_size=0.1,
    )
    results = metric.solve_power(power_data_arrow, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power > 0
    assert result.power < 1
    assert result.rel_effect_size == 0.1

def test_ratio_of_means_solve_power_power(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        alternative="greater",
        rel_effect_size=0.1,
        n_obs=10_000,
    )
    results = metric.solve_power(power_data_aggr, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == pytest.approx(0.895653144822902)
    assert result.effect_size == pytest.approx(0.1 * power_data_aggr.mean("orders"))
    assert result.rel_effect_size == 0.1
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        alternative="less",
        n_obs=10_000,
    )
    results = metric.solve_power(power_data_aggr, "effect_size")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == pytest.approx(-0.04027000268908317)
    assert result.rel_effect_size == pytest.approx(-0.08568085678528335)
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_rel_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", n_obs=10_000)
    results = metric.solve_power(power_data_aggr, "rel_effect_size")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == pytest.approx(0.04537464375458489)
    assert result.rel_effect_size == pytest.approx(0.09654179522252104)
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_n_obs(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", effect_size=0.05)
    results = metric.solve_power(power_data_aggr, "n_obs")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == 0.05
    assert result.rel_effect_size == pytest.approx(
        0.05 / power_data_aggr.mean("orders"))
    assert result.n_obs == 8236

def test_ratio_of_means_solve_power_multi(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        rel_effect_size=(0.05, 0.1),
        n_obs=(5_000, 10_000),
    )
    results = metric.solve_power(power_data_aggr, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 4

def test_ratio_of_means_solve_power_raises_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    with pytest.raises(ValueError, match="One of them should be defined"):
        metric.solve_power(power_data_aggr, "power")
    with pytest.raises(ValueError, match="Only one of them should be defined"):
        tea_tasting.metrics.mean.RatioOfMeans(
            numer="orders",
            effect_size=0.05,
            rel_effect_size=0.1,
        )

def test_ratio_of_means_solve_power_raises_max_iter(
    power_data_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", rel_effect_size=0.01)
    with (
        unittest.mock.patch("tea_tasting.metrics.mean.MAX_ITER", 1),
        pytest.raises(RuntimeError, match="Maximum number of iterations"),
    ):
        metric.solve_power(power_data_aggr, "n_obs")


def test_mean_analyze(data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates]) -> None:
    metric = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
    )
    ratio_metric = tea_tasting.metrics.mean.RatioOfMeans(
        "orders",
        numer_covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
    )
    assert metric.analyze(data_aggr, 0, 1) == ratio_metric.analyze(data_aggr, 0, 1)


def test_mean_solve_power(power_data_aggr: tea_tasting.aggr.Aggregates) -> None:
    metric = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        rel_effect_size=0.1,
    )
    ratio_metric = tea_tasting.metrics.mean.RatioOfMeans(
        "orders",
        numer_covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        rel_effect_size=0.1,
    )
    assert metric.solve_power(power_data_aggr, "power") ==  ratio_metric.solve_power(
        power_data_aggr, "power")


@pytest.fixture
def covariate_metric() -> tea_tasting.metrics.mean.RatioOfMeans:
    return tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
    )

@pytest.fixture
def reference_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(
        n_users=100, covariates=True, rng=43)

@pytest.fixture
def covariate_fit(
    reference_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> tea_tasting.metrics.mean.CovariateFit:
    return covariate_metric.fit_covariate(reference_arrow)


def test_covariate_fit_fit(
    reference_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> None:
    fit = covariate_metric.fit_covariate(reference_arrow)
    aggr = tea_tasting.data.read_aggregates(
        reference_arrow, aggr_cols=covariate_metric.aggr_cols).with_zero_div()
    coef = (
        aggr.ratio_cov(
            "orders", "sessions", "orders_covariate", "sessions_covariate")
        / aggr.ratio_var("orders_covariate", "sessions_covariate")
    )
    assert fit.version == tea_tasting.metrics.mean.COVARIATE_FIT_VERSION
    assert fit.numer == "orders"
    assert fit.denom == "sessions"
    assert fit.numer_covariate == "orders_covariate"
    assert fit.denom_covariate == "sessions_covariate"
    assert fit.coef == pytest.approx(coef)
    assert fit.coef != 0
    assert fit.covariate_mean == pytest.approx(
        aggr.mean("orders_covariate") / aggr.mean("sessions_covariate"))
    assert fit.n_obs == 100

def test_covariate_fit_accepts_aggregates(
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
    reference_arrow: pa.Table,
) -> None:
    aggr = tea_tasting.data.read_aggregates(
        reference_arrow, aggr_cols=covariate_metric.aggr_cols)
    assert covariate_metric.fit_covariate(aggr) == covariate_metric.fit_covariate(
        reference_arrow)

def test_covariate_fit_to_dict_from_dict(
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    state = covariate_fit.to_dict()
    assert state["version"] == tea_tasting.metrics.mean.COVARIATE_FIT_VERSION
    assert state["numer"] == "orders"
    assert state["denom"] == "sessions"
    assert state["numer_covariate"] == "orders_covariate"
    assert state["denom_covariate"] == "sessions_covariate"
    assert state["coef"] == covariate_fit.coef
    assert state["covariate_mean"] == covariate_fit.covariate_mean
    assert state["n_obs"] == covariate_fit.n_obs
    restored = tea_tasting.metrics.mean.CovariateFit.from_dict(state)
    assert restored.to_dict() == state

def test_covariate_fit_from_dict_missing_field() -> None:
    state = {
        "version": tea_tasting.metrics.mean.COVARIATE_FIT_VERSION,
        "numer": "orders",
        "denom": None,
        "numer_covariate": "orders_covariate",
        "denom_covariate": None,
        "coef": 1.0,
    }
    with pytest.raises(ValueError, match=r"missing required fields.*covariate_mean"):
        tea_tasting.metrics.mean.CovariateFit.from_dict(state)

def test_covariate_fit_from_dict_version_mismatch(
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    state = covariate_fit.to_dict()
    state["version"] = tea_tasting.metrics.mean.COVARIATE_FIT_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported covariate fit version"):
        tea_tasting.metrics.mean.CovariateFit.from_dict(state)

def test_covariate_fit_from_dict_not_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        tea_tasting.metrics.mean.CovariateFit.from_dict([])  # ty:ignore[invalid-argument-type]

def test_covariate_fit_degenerate_reference(
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> None:
    aggr = tea_tasting.aggr.Aggregates(
        count_=10,
        mean_={
            "orders": 1.0,
            "sessions": 1.0,
            "orders_covariate": 1.0,
            "sessions_covariate": 1.0,
        },
        var_={
            "orders": 1.0,
            "sessions": 1.0,
            "orders_covariate": 0.0,
            "sessions_covariate": 0.0,
        },
        cov_={
            ("orders", "sessions"): 0.0,
            ("orders", "orders_covariate"): 0.0,
            ("orders", "sessions_covariate"): 0.0,
            ("sessions", "orders_covariate"): 0.0,
            ("sessions", "sessions_covariate"): 0.0,
            ("orders_covariate", "sessions_covariate"): 0.0,
        },
    )
    with pytest.raises(ValueError, match="degenerate"):
        covariate_metric.fit_covariate(aggr)

def test_covariate_fit_no_covariate() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    with pytest.raises(ValueError, match="no `numer_covariate`"):
        metric.fit_covariate(tea_tasting.datasets.make_users_data(n_users=10))

def test_covariate_fit_grouped_reference(
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> None:
    data_aggr = tea_tasting.data.read_aggregates(
        tea_tasting.datasets.make_users_data(
            n_users=100, covariates=True, rng=42),
        aggr_cols=covariate_metric.aggr_cols,
        variant="variant",
    )
    with pytest.raises(TypeError, match="grouped by variant"):
        covariate_metric.fit_covariate(data_aggr)

def test_covariate_fit_missing_reference_columns(
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> None:
    aggr = tea_tasting.aggr.Aggregates(
        count_=10,
        mean_={"orders": 1.0, "sessions": 1.0},
        var_={"orders": 1.0, "sessions": 1.0},
        cov_={("orders", "sessions"): 0.0},
    )
    with pytest.raises(ValueError, match="missing aggregated columns"):
        covariate_metric.fit_covariate(aggr)


def test_ratio_of_means_frozen_fit_analyze(
    data_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    frozen = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        covariate_fit=covariate_fit,
    )
    assert frozen.covariate_fit is covariate_fit
    aggr = tea_tasting.data.read_aggregates(
        data_arrow, aggr_cols=covariate_metric.aggr_cols, variant="variant")
    result = frozen.analyze(aggr, 0, 1)
    total = aggr[0].with_zero_div() + aggr[1].with_zero_div()
    coef, center = covariate_fit.coef, covariate_fit.covariate_mean
    for variant, value in (("control", 0), ("treatment", 1)):
        aggr_i = aggr[value].with_zero_div()
        expected_mean = (
            aggr_i.mean("orders") / aggr_i.mean("sessions")
            - coef * (
                aggr_i.mean("orders_covariate") / aggr_i.mean("sessions_covariate")
                - center
            )
        )
        assert getattr(result, variant) == pytest.approx(expected_mean)
    assert total.count() == aggr[0].count() + aggr[1].count()

def test_ratio_of_means_frozen_fit_does_not_reestimate(
    data_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    aggr = tea_tasting.data.read_aggregates(
        data_arrow, aggr_cols=covariate_metric.aggr_cols, variant="variant")
    frozen = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        covariate_fit=covariate_fit,
    )
    patched = unittest.mock.patch.object(
        tea_tasting.metrics.mean.RatioOfMeans,
        "_covariate_coef",
        side_effect=AssertionError("Coefficient was re-estimated."),
    )
    with patched:
        result = frozen.analyze(aggr, 0, 1)
    assert result.effect_size != 0

def test_ratio_of_means_frozen_fit_power(
    power_data_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    frozen = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        rel_effect_size=0.1,
        n_obs=10_000,
        covariate_fit=covariate_fit,
    )
    aggr = tea_tasting.data.read_aggregates(
        power_data_arrow, aggr_cols=covariate_metric.aggr_cols).with_zero_div()
    coef, center = covariate_fit.coef, covariate_fit.covariate_mean
    metric_mean = (
        aggr.mean("orders") / aggr.mean("sessions")
        - coef * (
            aggr.mean("orders_covariate") / aggr.mean("sessions_covariate")
            - center
        )
    )
    metric_var = (
        aggr.ratio_var("orders", "sessions")
        + coef * coef * aggr.ratio_var(
            "orders_covariate", "sessions_covariate")
        - 2 * coef * aggr.ratio_cov(
            "orders", "sessions", "orders_covariate", "sessions_covariate")
    )
    expected = frozen._power_from_stats(
        sample_var=metric_var,
        sample_count=10_000,
        effect_size=0.1 * metric_mean,
    )
    result = frozen.solve_power(power_data_arrow, "power")[0]
    assert result.power == pytest.approx(expected)
    assert result.effect_size == pytest.approx(0.1 * metric_mean)

def test_ratio_of_means_frozen_fit_accepts_state_dict(
    data_arrow: pa.Table,
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    frozen = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        covariate_fit=covariate_fit.to_dict(),
    )
    assert isinstance(frozen.covariate_fit, tea_tasting.metrics.mean.CovariateFit)
    assert frozen.covariate_fit.to_dict() == covariate_fit.to_dict()
    aggr = tea_tasting.data.read_aggregates(
        data_arrow, aggr_cols=covariate_metric.aggr_cols, variant="variant")
    frozen_other = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        covariate_fit=covariate_fit,
    )
    assert frozen.analyze(aggr, 0, 1) == frozen_other.analyze(aggr, 0, 1)

def test_ratio_of_means_frozen_fit_definition_mismatch(
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        tea_tasting.metrics.mean.RatioOfMeans(
            numer="revenue",
            denom="sessions",
            numer_covariate="orders_covariate",
            denom_covariate="sessions_covariate",
            covariate_fit=covariate_fit,
        )

def test_ratio_of_means_frozen_fit_without_covariate(
    reference_arrow: pa.Table,
) -> None:
    fit = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        numer_covariate="orders_covariate",
    ).fit_covariate(reference_arrow)
    with pytest.raises(ValueError, match="no `numer_covariate`"):
        tea_tasting.metrics.mean.RatioOfMeans(
            numer="orders",
            covariate_fit=fit,
        )

def test_ratio_of_means_frozen_fit_invalid_type() -> None:
    class FakeFit:
        pass

    with pytest.raises(TypeError, match="CovariateFit or a mapping"):
        tea_tasting.metrics.mean.RatioOfMeans(
            numer="orders",
            denom="sessions",
            numer_covariate="orders_covariate",
            denom_covariate="sessions_covariate",
            covariate_fit=FakeFit(),  # ty:ignore[invalid-argument-type]
        )

def test_ratio_of_means_analyze_missing_columns() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
    )
    aggr = tea_tasting.aggr.Aggregates(
        count_=10,
        mean_={"orders": 1.0, "sessions": 1.0},
        var_={"orders": 1.0, "sessions": 1.0},
        cov_={("orders", "sessions"): 0.0},
    )
    with pytest.raises(ValueError, match="missing aggregated columns"):
        metric.analyze_aggregates(aggr, aggr)

def test_mean_frozen_fit(
    data_arrow: pa.Table,
    reference_arrow: pa.Table,
) -> None:
    metric = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
    )
    fit = metric.fit_covariate(reference_arrow)
    assert fit.denom is None
    assert fit.denom_covariate is None
    frozen = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
        covariate_fit=fit,
    )
    dynamic = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
    )
    assert frozen.analyze(data_arrow, 0, 1, variant="variant") != (
        dynamic.analyze(data_arrow, 0, 1, variant="variant"))


def _covariate_aggregates(
    *,
    orders: float,
    orders_var: float,
    orders_covariate: float,
    orders_covariate_var: float,
    cov: float,
    count: int = 10,
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=count,
        mean_={"orders": orders, "orders_covariate": orders_covariate},
        var_={"orders": orders_var, "orders_covariate": orders_covariate_var},
        cov_={("orders", "orders_covariate"): cov},
    )

def test_covariate_fit_coef_not_finite(
    covariate_metric: tea_tasting.metrics.mean.RatioOfMeans,
) -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        numer_covariate="orders_covariate",
    )
    aggr = _covariate_aggregates(
        orders=1.0,
        orders_var=1.0,
        orders_covariate=1.0,
        orders_covariate_var=1.0,
        cov=float("inf"),
    )
    with pytest.raises(ValueError, match="coefficient is not finite"):
        metric.fit_covariate(aggr)
    assert covariate_metric.numer_covariate == "orders_covariate"

def test_covariate_fit_mean_not_finite() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        numer_covariate="orders_covariate",
    )
    aggr = tea_tasting.aggr.Aggregates(
        count_=10,
        mean_={"orders": 1.0, "orders_covariate": float("nan")},
        var_={"orders": 1.0, "orders_covariate": 1.0},
        cov_={("orders", "orders_covariate"): 0.0},
    )
    with pytest.raises(ValueError, match="center is not finite"):
        metric.fit_covariate(aggr)

def test_covariate_fit_value_not_finite() -> None:
    with pytest.raises(ValueError, match="coef must be finite"):
        tea_tasting.metrics.mean.CovariateFit(
            numer="orders",
            denom=None,
            numer_covariate="orders_covariate",
            denom_covariate=None,
            coef=float("nan"),
            covariate_mean=1.0,
        )
    with pytest.raises(ValueError, match="covariate_mean must be finite"):
        tea_tasting.metrics.mean.CovariateFit(
            numer="orders",
            denom=None,
            numer_covariate="orders_covariate",
            denom_covariate=None,
            coef=1.0,
            covariate_mean=float("inf"),
        )

def test_covariate_fit_eq_and_hash(
    covariate_fit: tea_tasting.metrics.mean.CovariateFit,
) -> None:
    same = tea_tasting.metrics.mean.CovariateFit.from_dict(
        covariate_fit.to_dict())
    assert covariate_fit == same
    assert hash(covariate_fit) == hash(same)
    assert covariate_fit != "not a fit"

def test_ratio_of_means_analyze_degenerate_covariate() -> None:
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        numer_covariate="orders_covariate",
    )
    aggr = _covariate_aggregates(
        orders=1.0,
        orders_var=1.0,
        orders_covariate=1.0,
        orders_covariate_var=0.0,
        cov=0.0,
    )
    result = metric.analyze_aggregates(aggr, aggr)
    assert result.control == pytest.approx(1.0)
    assert result.effect_size == pytest.approx(0.0)
