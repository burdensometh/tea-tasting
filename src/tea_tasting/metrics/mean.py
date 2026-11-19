"""Metrics for the analysis of means."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, NamedTuple, overload

import scipy.optimize
import scipy.stats

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.data
from tea_tasting.metrics.base import (
    MetricBaseAggregated,
    MetricPowerResults,
    PowerBaseAggregated,
)
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


MAX_ITER = 100

COVARIATE_FIT_VERSION = 1
"""Version of the frozen covariate fit state format."""

_COVARIATE_FIT_FIELDS = (
    "version",
    "numer",
    "denom",
    "numer_covariate",
    "denom_covariate",
    "coef",
    "covariate_mean",
)


class CovariateFit(tea_tasting.utils.ReprMixin):  # noqa: D101
    def __init__(
        self,
        numer: str,
        denom: str | None,
        numer_covariate: str,
        denom_covariate: str | None,
        coef: float,
        covariate_mean: float,
        n_obs: int | None = None,
        version: int = COVARIATE_FIT_VERSION,
    ) -> None:
        """Frozen covariate adjustment estimated on historical reference data.

        A frozen fit stores the covariate coefficient and the covariate center
        (mean) together with the ratio definition they were estimated for.
        After freezing, an experiment analysis applies the stored coefficient
        and center to the experiment data; it never re-estimates them from the
        experiment data.

        Attributes:
            numer: Numerator column name of the metric.
            denom: Denominator column name of the metric.
            numer_covariate: Covariate numerator column name.
            denom_covariate: Covariate denominator column name.
            coef: Covariate coefficient.
            covariate_mean: Covariate center (mean of the covariate ratio
                in the reference data).
            n_obs: Number of observations in the reference data.
            version: Version of the frozen fit state format.

        References:
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).
        """  # noqa: E501
        self.version = tea_tasting.utils.check_scalar(
            version,
            "version",
            typ=int,
            in_={COVARIATE_FIT_VERSION},
        )
        self.numer = tea_tasting.utils.check_scalar(numer, "numer", typ=str)
        self.denom = tea_tasting.utils.check_scalar(denom, "denom", typ=str | None)
        self.numer_covariate = tea_tasting.utils.check_scalar(
            numer_covariate,
            "numer_covariate",
            typ=str,
        )
        self.denom_covariate = tea_tasting.utils.check_scalar(
            denom_covariate,
            "denom_covariate",
            typ=str | None,
        )
        self.coef = _finite_float(coef, "coef")
        self.covariate_mean = _finite_float(covariate_mean, "covariate_mean")
        self.n_obs = tea_tasting.utils.check_scalar(n_obs, "n_obs", typ=int | None)


    @classmethod
    def fit(
        cls,
        data: tea_tasting.data.Table | tea_tasting.aggr.Aggregates,
        metric: RatioOfMeans,
    ) -> CovariateFit:
        """Estimate a frozen covariate fit on independent reference data.

        Args:
            data: Historical reference data. It must not be grouped by variant.
            metric: Metric that defines the ratio and the covariate columns.

        Returns:
            Frozen covariate fit estimated on the reference data.

        Raises:
            TypeError: If `data` is grouped by variant.
            ValueError: If the metric has no covariate, or if the covariate
                is degenerate in the reference data (zero or non-finite
                variance, or non-finite coefficient or center).
        """
        if metric.numer_covariate is None:
            raise ValueError(
                "Cannot fit a covariate: the metric has no `numer_covariate`.",
            )
        aggr = _reference_aggregates(data, metric.aggr_cols)
        _check_aggregate_columns(aggr, metric, "Reference data")

        covariate_mean = (
            aggr.mean(metric.numer_covariate) /
            aggr.mean(metric.denom_covariate)
        )
        if not _is_finite(covariate_mean):
            raise ValueError(
                "Cannot fit a covariate: the estimated covariate center "
                f"is not finite ({covariate_mean!r}).",
            )
        covariate_var = aggr.ratio_var(
            metric.numer_covariate, metric.denom_covariate)
        if not _is_positive_finite(covariate_var):
            raise ValueError(
                "Cannot fit a covariate: the covariate ratio is degenerate, "
                f"its variance in the reference data is {covariate_var!r}. "
                "A non-zero finite variance is required to estimate the "
                "covariate coefficient.",
            )
        coef = metric._covariate_cov(aggr) / covariate_var
        if not _is_finite(coef):
            raise ValueError(
                "Cannot fit a covariate: the estimated covariate coefficient "
                f"is not finite ({coef!r}).",
            )

        return cls(
            numer=metric.numer,
            denom=metric.denom,
            numer_covariate=metric.numer_covariate,
            denom_covariate=metric.denom_covariate,
            coef=coef,
            covariate_mean=covariate_mean,
            n_obs=aggr.count_,
        )


    def __eq__(self, other: object) -> bool:
        """Check whether the fits store the same state."""
        if not isinstance(other, CovariateFit):
            return NotImplemented
        return self.to_dict() == other.to_dict()


    def __hash__(self) -> int:
        """Hash the stored state."""
        return hash(tuple(self.to_dict().items()))


    def to_dict(self) -> dict[str, object]:
        """Convert the frozen fit to a dictionary.

        Returns:
            Dictionary with the state fields and the ratio definition.
        """
        return {
            "version": self.version,
            "numer": self.numer,
            "denom": self.denom,
            "numer_covariate": self.numer_covariate,
            "denom_covariate": self.denom_covariate,
            "coef": self.coef,
            "covariate_mean": self.covariate_mean,
            "n_obs": self.n_obs,
        }


    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CovariateFit:
        """Create a frozen fit from a dictionary.

        Args:
            data: Dictionary produced by `to_dict` or equivalent.

        Returns:
            Frozen covariate fit.

        Raises:
            ValueError: If required fields are missing, if the state version
                is not supported, or if the state values are invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError("Covariate fit state must be a mapping.")
        missing = [field for field in _COVARIATE_FIT_FIELDS if field not in data]
        if missing:
            raise ValueError(
                "Covariate fit state is missing required fields: "
                f"{missing}.",
            )

        version = tea_tasting.utils.check_scalar(
            data["version"], "version", typ=int)
        if version != COVARIATE_FIT_VERSION:
            raise ValueError(
                f"Unsupported covariate fit version: {version!r}. "
                f"Expected version {COVARIATE_FIT_VERSION}.",
            )

        return cls(
            version=version,  # ty:ignore[invalid-argument-type]
            numer=tea_tasting.utils.check_scalar(data["numer"], "numer", typ=str),  # ty:ignore[invalid-argument-type]
            denom=tea_tasting.utils.check_scalar(
                data["denom"], "denom", typ=str | None),  # ty:ignore[invalid-argument-type]
            numer_covariate=tea_tasting.utils.check_scalar(
                data["numer_covariate"],
                "numer_covariate",
                typ=str,
            ),  # ty:ignore[invalid-argument-type]
            denom_covariate=tea_tasting.utils.check_scalar(
                data["denom_covariate"],
                "denom_covariate",
                typ=str | None,
            ),  # ty:ignore[invalid-argument-type]
            coef=data["coef"],  # ty:ignore[invalid-argument-type]
            covariate_mean=data["covariate_mean"],  # ty:ignore[invalid-argument-type]
            n_obs=tea_tasting.utils.check_scalar(
                data.get("n_obs"), "n_obs", typ=int | None),  # ty:ignore[invalid-argument-type]
        )


def _finite_float(value: float, name: str) -> float:
    value = tea_tasting.utils.check_scalar(value, name, typ=float | int)
    if not _is_finite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return float(value)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _is_positive_finite(value: float) -> bool:
    return _is_finite(value) and value > 0


def _reference_aggregates(
    data: tea_tasting.data.Table | tea_tasting.aggr.Aggregates,
    aggr_cols: tea_tasting.aggr.AggrCols,
) -> tea_tasting.aggr.Aggregates:
    if tea_tasting.data._is_aggregates_mapping(data):
        raise TypeError(
            "Reference data for a covariate fit must be a single sample, "
            "not aggregated statistics grouped by variant.",
        )
    if isinstance(data, tea_tasting.aggr.Aggregates):
        aggr = data
    else:
        aggr = tea_tasting.data.read_aggregates(data, aggr_cols=aggr_cols)
    return aggr.with_zero_div()


def _check_aggregate_columns(
    aggr: tea_tasting.aggr.Aggregates,
    metric: RatioOfMeans,
    context: str,
) -> None:
    mean_var_cols = tuple(
        col for col in (
            metric.numer,
            metric.denom,
            metric.numer_covariate,
            metric.denom_covariate,
        )
        if col is not None
    )
    cov_cols = [(metric.numer, metric.denom)]
    if metric.numer_covariate is not None:
        cov_cols.extend(
            (metric_col, covariate_col)
            for metric_col in (metric.numer, metric.denom)
            for covariate_col in (
                metric.numer_covariate, metric.denom_covariate)
            if metric_col is not None and covariate_col is not None
        )

    missing: list[object] = [
        col for col in mean_var_cols
        if col not in aggr.mean_ or col not in aggr.var_
    ]
    missing.extend(
        tuple(sorted((left, right)))
        for left, right in cov_cols
        if left is not None and right is not None
        and tuple(sorted((left, right))) not in aggr.cov_
    )
    if missing:
        raise ValueError(
            f"{context} is missing aggregated columns required by the metric: "
            f"{missing}. "
            f"Available mean columns: {sorted(aggr.mean_)}.",
        )


class MeanResult(NamedTuple):
    """Result of the analysis of means.

    Attributes:
        control: Control mean.
        treatment: Treatment mean.
        effect_size: Absolute effect size. Difference between the two means.
        effect_size_ci_lower: Lower bound of the absolute effect size
            confidence interval.
        effect_size_ci_upper: Upper bound of the absolute effect size
            confidence interval.
        rel_effect_size: Relative effect size. Difference between the two means,
            divided by the control mean.
        rel_effect_size_ci_lower: Lower bound of the relative effect size
            confidence interval.
        rel_effect_size_ci_upper: Upper bound of the relative effect size
            confidence interval.
        pvalue: P-value.
        statistic: Statistic (standardized effect size).
    """
    control: float
    treatment: float
    effect_size: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    rel_effect_size: float
    rel_effect_size_ci_lower: float
    rel_effect_size_ci_upper: float
    pvalue: float
    statistic: float


class MeanPowerResult(NamedTuple):
    """Power analysis results.

    Attributes:
        power: Statistical power.
        effect_size: Absolute effect size. Difference between the two means.
        rel_effect_size: Relative effect size. Difference between the two means,
            divided by the control mean.
        n_obs: Number of observations in the control and in the treatment together.
    """
    power: float
    effect_size: float
    rel_effect_size: float
    n_obs: float

MeanPowerResults = MetricPowerResults[MeanPowerResult]


class RatioOfMeans(  # noqa: D101
    MetricBaseAggregated[MeanResult],
    PowerBaseAggregated[MeanPowerResults],
):
    def __init__(  # noqa: PLR0913
        self,
        numer: str,
        denom: str | None = None,
        numer_covariate: str | None = None,
        denom_covariate: str | None = None,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
        alpha: float | None = None,
        ratio: float | int | None = None,
        power: float | None = None,
        effect_size: float | int | Sequence[float | int] | None = None,
        rel_effect_size: float | Sequence[float] | None = None,
        n_obs: int | Sequence[int] | None = None,
        covariate_fit: CovariateFit | Mapping[str, object] | None = None,
    ) -> None:
        """Metric for the analysis of ratios of means.

        Args:
            numer: Numerator column name.
            denom: Denominator column name.
            numer_covariate: Covariate numerator column name.
            denom_covariate: Covariate denominator column name.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than the mean
                    in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).
            alpha: Significance level. Only for the analysis of power.
            ratio: Ratio of the number of observations in the treatment
                relative to the control. Only for the analysis of power.
            power: Statistical power. Only for the analysis of power.
            effect_size: Absolute effect size. Difference between the two means.
                Only for the analysis of power.
            rel_effect_size: Relative effect size. Difference between the two means,
                divided by the control mean. Only for the analysis of power.
            n_obs: Number of observations in the control and in the treatment together.
                Only for the analysis of power.
            covariate_fit: Frozen covariate fit estimated on independent
                reference data. If `None`, the covariate coefficient and center
                are estimated from the analyzed data (CUPED/CUPAC). If defined,
                the metric applies the frozen coefficient and center and never
                re-estimates them from the analyzed data. The fit must have
                the same ratio definition as the metric.

        Parameter defaults:
            Defaults for parameters `alpha`, `alternative`, `confidence_level`,
            `equal_var`, `n_obs`, `power`, `ratio`, and `use_t` can be changed
            using the `config_context` and `set_config` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            ... )
            >>> data = tt.make_users_data(rng=42)
            >>> result = experiment.analyze(data)
            >>> result
            metric             control treatment rel_effect_size rel_effect_size_ci pvalue
            orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762

            ```

            With CUPED:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.RatioOfMeans(
            ...         "orders",
            ...         "sessions",
            ...         "orders_covariate",
            ...         "sessions_covariate",
            ...     ),
            ... )
            >>> data = tt.make_users_data(rng=42, covariates=True)
            >>> result = experiment.analyze(data)
            >>> result
            metric             control treatment rel_effect_size rel_effect_size_ci  pvalue
            orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229

            ```

            With a frozen covariate fit estimated on reference data:

            ```pycon
            >>> reference = tt.make_users_data(rng=43, covariates=True)
            >>> metric = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ... )
            >>> fit = metric.fit_covariate(reference)
            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.RatioOfMeans(
            ...         "orders",
            ...         "sessions",
            ...         "orders_covariate",
            ...         "sessions_covariate",
            ...         covariate_fit=fit,
            ...     ),
            ... )
            >>> data = tt.make_users_data(rng=42, covariates=True)
            >>> round(experiment.analyze(data).to_dicts()[0]["rel_effect_size"], 3)
            0.125

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     rng=42,
            ...     sessions_uplift=0,
            ...     orders_uplift=0,
            ...     revenue_uplift=0,
            ...     covariates=True,
            ... )
            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     n_obs=(10_000, 20_000),
            ... )
            >>> # Solve for effect size.
            >>> orders_per_session.solve_power(data)
            power effect_size rel_effect_size n_obs
              80%      0.0177            6.8% 10000
              80%      0.0125            4.8% 20000

            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     rel_effect_size=0.05,
            ... )
            >>> # Solve for the total number of observations.
            >>> orders_per_session.solve_power(data, "n_obs")
            power effect_size rel_effect_size n_obs
              80%      0.0130            5.0% 18515

            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     rel_effect_size=0.1,
            ... )
            >>> # Solve for power. Infer number of observations from the sample.
            >>> orders_per_session.solve_power(data, "power")
            power effect_size rel_effect_size n_obs
              74%      0.0261             10%  4000

            ```
        """  # noqa: E501
        self.numer = tea_tasting.utils.check_scalar(numer, "numer", typ=str)
        self.denom = tea_tasting.utils.check_scalar(denom, "denom", typ=str | None)
        self.numer_covariate = tea_tasting.utils.check_scalar(
            numer_covariate, "numer_covariate", typ=str | None)
        self.denom_covariate = tea_tasting.utils.check_scalar(
            denom_covariate, "denom_covariate", typ=str | None)
        self.alternative = (
            tea_tasting.utils.auto_check(alternative, "alternative")
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )
        self.confidence_level = (
            tea_tasting.utils.auto_check(confidence_level, "confidence_level")
            if confidence_level is not None
            else tea_tasting.config.get_config("confidence_level")
        )
        self.equal_var = (
            tea_tasting.utils.auto_check(equal_var, "equal_var")
            if equal_var is not None
            else tea_tasting.config.get_config("equal_var")
        )
        self.use_t = (
            tea_tasting.utils.auto_check(use_t, "use_t")
            if use_t is not None
            else tea_tasting.config.get_config("use_t")
        )
        self.alpha = (
            tea_tasting.utils.auto_check(alpha, "alpha")
            if alpha is not None
            else tea_tasting.config.get_config("alpha")
        )
        self.ratio = (
            tea_tasting.utils.auto_check(ratio, "ratio")
            if ratio is not None
            else tea_tasting.config.get_config("ratio")
        )
        self.power = (
            tea_tasting.utils.auto_check(power, "power")
            if power is not None
            else tea_tasting.config.get_config("power")
        )
        if effect_size is not None and rel_effect_size is not None:
            raise ValueError(
                "Both `effect_size` and `rel_effect_size` are not `None`. "
                "Only one of them should be defined.",
            )
        if isinstance(effect_size, Sequence):
            for x in effect_size:
                tea_tasting.utils.check_scalar(
                    x, "effect_size", typ=float | int,
                    gt=float("-inf"), lt=float("inf"), ne=0,
                )
        elif effect_size is not None:
            tea_tasting.utils.check_scalar(
                effect_size, "effect_size", typ=float | int,
                gt=float("-inf"), lt=float("inf"), ne=0,
            )
        self.effect_size = effect_size
        if isinstance(rel_effect_size, Sequence):
            for x in rel_effect_size:
                tea_tasting.utils.check_scalar(
                    x, "rel_effect_size", typ=float | int,
                    gt=float("-inf"), lt=float("inf"), ne=0,
                )
        elif rel_effect_size is not None:
            tea_tasting.utils.check_scalar(
                rel_effect_size, "rel_effect_size", typ=float | int,
                gt=float("-inf"), lt=float("inf"), ne=0,
            )
        self.rel_effect_size = rel_effect_size
        self.n_obs = (
            tea_tasting.utils.auto_check(n_obs, "n_obs")
            if n_obs is not None
            else tea_tasting.config.get_config("n_obs")
        )
        self.covariate_fit = self._validate_covariate_fit(covariate_fit)


    def fit_covariate(
        self,
        data: tea_tasting.data.Table | tea_tasting.aggr.Aggregates,
    ) -> CovariateFit:
        """Fit a frozen covariate adjustment on independent reference data.

        The returned fit is estimated from `data` only. Pass it to
        a new metric via the `covariate_fit` parameter to apply the frozen
        coefficient and center to the experiment analysis without re-estimating
        them from the experiment data.

        Args:
            data: Historical reference data. It must not be grouped by variant.

        Returns:
            Frozen covariate fit estimated on the reference data.

        Raises:
            TypeError: If `data` is grouped by variant.
            ValueError: If the metric has no covariate, if the reference data
                misses the aggregated columns required by the metric, or if the
                covariate is degenerate in the reference data.

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> reference = tt.make_users_data(rng=42, covariates=True)
            >>> fit = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ... ).fit_covariate(reference)
            >>> round(fit.coef, 3)
            0.557

            ```
        """
        return CovariateFit.fit(data, self)


    def _validate_covariate_fit(
        self,
        covariate_fit: CovariateFit | Mapping[str, object] | None,
    ) -> CovariateFit | None:
        if covariate_fit is None:
            return None
        if isinstance(covariate_fit, Mapping):
            covariate_fit = CovariateFit.from_dict(covariate_fit)
        if not isinstance(covariate_fit, CovariateFit):
            raise TypeError(
                "covariate_fit must be an instance of CovariateFit or a mapping, "
                f"got {type(covariate_fit).__name__}.",
            )
        if self.numer_covariate is None:
            raise ValueError(
                "covariate_fit is defined, but the metric has no "
                "`numer_covariate` to apply it to.",
            )
        expected = (
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        )
        actual = (
            covariate_fit.numer,
            covariate_fit.denom,
            covariate_fit.numer_covariate,
            covariate_fit.denom_covariate,
        )
        if actual != expected:
            raise ValueError(
                "covariate_fit ratio definition does not match the metric: "
                f"fit (numer, denom, numer_covariate, denom_covariate)={actual!r}, "
                f"metric={expected!r}.",
            )
        return covariate_fit


    @property
    def aggr_cols(self) -> tea_tasting.aggr.AggrCols:
        """Columns to be aggregated for a metric analysis."""
        cols = tuple(
            col for col in (
                self.numer,
                self.denom,
                self.numer_covariate,
                self.denom_covariate,
            )
            if col is not None
        )
        return tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=cols,
            var_cols=cols,
            cov_cols=tuple(
                (col0, col1)
                for col0 in cols
                for col1 in cols
                if col0 < col1
            ),
        )


    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> MeanResult:
        """Analyze a metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        control = control.with_zero_div()
        treatment = treatment.with_zero_div()
        _check_aggregate_columns(control, self, "Control data")
        _check_aggregate_columns(treatment, self, "Treatment data")
        total = control + treatment
        covariate_coef, covariate_mean = self._covariate_parameters(total)
        return self._analyze_stats(
            contr_mean=self._metric_mean(control, covariate_coef, covariate_mean),
            contr_var=self._metric_var(control, covariate_coef),
            contr_count=control.count(),
            treat_mean=self._metric_mean(treatment, covariate_coef, covariate_mean),
            treat_var=self._metric_var(treatment, covariate_coef),
            treat_count=treatment.count(),
        )


    def solve_power_from_aggregates(
        self,
        data: tea_tasting.aggr.Aggregates,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> MeanPowerResults:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """
        data = data.with_zero_div()
        _check_aggregate_columns(data, self, "Sample data")
        covariate_coef, covariate_mean = self._covariate_parameters(data)
        metric_mean = self._metric_mean(data, covariate_coef, covariate_mean)
        metric_var = self._metric_var(data, covariate_coef)

        power, effect_size, rel_effect_size, n_obs = self._validate_power_parameters(
            metric_mean=metric_mean,
            sample_count=data.count(),
            parameter=parameter,
        )

        result = MeanPowerResults()
        for effect_size_i, rel_effect_size_i in zip(
            effect_size,
            rel_effect_size,
            strict=True,
        ):
            for n_obs_i in n_obs:
                parameter_value = self._solve_power_from_stats(
                    sample_var=metric_var,
                    sample_count=n_obs_i,
                    effect_size=effect_size_i,
                    power=power,
                )
                result.append(MeanPowerResult(
                    power=parameter_value if parameter == "power" else power,  # ty:ignore[invalid-argument-type]
                    effect_size=(
                        parameter_value
                        if parameter in {"effect_size", "rel_effect_size"}
                        else effect_size_i
                    ),  # ty:ignore[invalid-argument-type]
                    rel_effect_size=(
                        parameter_value / metric_mean
                        if parameter in {"effect_size", "rel_effect_size"}
                        else rel_effect_size_i
                    ),  # ty:ignore[invalid-argument-type]
                    n_obs=(
                        math.ceil(parameter_value)
                        if parameter == "n_obs"
                        else n_obs_i
                    ),  # ty:ignore[invalid-argument-type]
                ))

        return result


    def _validate_power_parameters(
        self,
        metric_mean: float,
        sample_count: int,
        parameter: Literal["power", "effect_size", "rel_effect_size", "n_obs"],
    ) -> tuple[
        float | None,  # power
        Sequence[float | int | None],  # effect_size
        Sequence[float | None],  # rel_effect_size
        Sequence[int | None],  # n_obs
    ]:
        n_obs = None
        effect_size = None
        rel_effect_size = None
        power = None

        if parameter in {"power", "n_obs"}:
            if self.effect_size is None and self.rel_effect_size is None:
                raise ValueError(
                    "Both `effect_size` and `rel_effect_size` are `None`. "
                    "One of them should be defined.",
                )
            effect_size = (
                self.effect_size if self.rel_effect_size is None
                else tuple(
                    rel_effect_size * metric_mean
                    for rel_effect_size in _to_seq(self.rel_effect_size)
                )
            )
            rel_effect_size = (
                self.rel_effect_size if self.effect_size is None
                else tuple(
                    effect_size / metric_mean
                    for effect_size in _to_seq(self.effect_size)
                )
            )

        if parameter in {"power", "effect_size", "rel_effect_size"}:
            n_obs = (sample_count,) if self.n_obs is None else self.n_obs

        if parameter in {"effect_size", "rel_effect_size", "n_obs"}:
            power = self.power

        return power, _to_seq(effect_size), _to_seq(rel_effect_size), _to_seq(n_obs)


    def _covariate_parameters(
        self,
        aggr: tea_tasting.aggr.Aggregates,
    ) -> tuple[float, float]:
        if self.covariate_fit is not None:
            return self.covariate_fit.coef, self.covariate_fit.covariate_mean
        if self.numer_covariate is None:
            return 0, 0
        return (
            self._covariate_coef(aggr),
            aggr.mean(self.numer_covariate) / aggr.mean(self.denom_covariate),
        )


    def _covariate_coef(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        covariate_var = aggr.ratio_var(self.numer_covariate, self.denom_covariate)
        if covariate_var == 0:
            return 0
        return self._covariate_cov(aggr) / covariate_var


    def _covariate_cov(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        return aggr.ratio_cov(
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        )


    def _metric_mean(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
        covariate_mean: float,
    ) -> float:
        value = aggr.mean(self.numer) / aggr.mean(self.denom)
        if self.numer_covariate is None:
            return value
        covariate = aggr.mean(self.numer_covariate) / aggr.mean(self.denom_covariate)
        return value - covariate_coef*(covariate - covariate_mean)


    def _metric_var(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
    ) -> float:
        var = aggr.ratio_var(self.numer, self.denom)
        if self.numer_covariate is None:
            return var
        covariate_var = aggr.ratio_var(self.numer_covariate, self.denom_covariate)
        covariate_cov = self._covariate_cov(aggr)
        return (
            var
            + covariate_coef * covariate_coef * covariate_var
            - 2 * covariate_coef * covariate_cov
        )


    def _analyze_stats(
        self,
        contr_mean: float,
        contr_var: float,
        contr_count: int,
        treat_mean: float,
        treat_var: float,
        treat_count: int,
    ) -> MeanResult:
        distr, _, scale = self._distr_and_scale(
            contr_var=contr_var,
            contr_count=contr_count,
            treat_var=treat_var,
            treat_count=treat_count,
        )
        log_distr, _, _ = self._distr_and_scale(
            contr_var=contr_var / contr_mean / contr_mean,
            contr_count=contr_count,
            treat_var=treat_var / treat_mean / treat_mean,
            treat_count=treat_count,
        )

        means_ratio = treat_mean / contr_mean
        effect_size = treat_mean - contr_mean

        if self.alternative == "greater":
            q = self.confidence_level
            effect_size_ci_lower = effect_size + distr.isf(q)
            means_ratio_ci_lower = means_ratio * math.exp(log_distr.isf(q))
            effect_size_ci_upper = means_ratio_ci_upper = float("+inf")
            pvalue = distr.sf(effect_size)
        elif self.alternative == "less":
            q = self.confidence_level
            effect_size_ci_lower = means_ratio_ci_lower = float("-inf")
            effect_size_ci_upper = effect_size + distr.ppf(q)
            means_ratio_ci_upper = means_ratio * math.exp(log_distr.ppf(q))
            pvalue = distr.cdf(effect_size)
        else:  # two-sided
            q = (1 + self.confidence_level) / 2
            half_ci = distr.ppf(q)
            effect_size_ci_lower = effect_size - half_ci
            effect_size_ci_upper = effect_size + half_ci

            rel_half_ci = math.exp(log_distr.ppf(q))
            means_ratio_ci_lower = means_ratio / rel_half_ci
            means_ratio_ci_upper = means_ratio * rel_half_ci

            pvalue = 2 * distr.sf(abs(effect_size))

        return MeanResult(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=effect_size,
            effect_size_ci_lower=effect_size_ci_lower,
            effect_size_ci_upper=effect_size_ci_upper,
            rel_effect_size=means_ratio - 1,
            rel_effect_size_ci_lower=means_ratio_ci_lower - 1,
            rel_effect_size_ci_upper=means_ratio_ci_upper - 1,
            pvalue=float(pvalue),
            statistic=effect_size / scale,
        )


    def _solve_power_from_stats(
        self,
        sample_var: float,
        sample_count: int | None = None,
        effect_size: float | None = None,
        power: float | None = None,
    ) -> float | int:
        if power is None and effect_size is not None and sample_count is not None:
            return self._power_from_stats(
                sample_var=sample_var,
                sample_count=sample_count,
                effect_size=effect_size,
            )

        if power is not None and effect_size is None and sample_count is not None:
            def fn(x: float | int) -> float:
                return power - self._power_from_stats(
                    sample_var=sample_var,
                    sample_count=sample_count,
                    effect_size=x,
                )
            sign = -1 if self.alternative == "less" else 1
            other_bound = _find_boundary(
                fn,
                sign * 10 * math.sqrt(sample_var / sample_count),
            )
            lower_bound, upper_bound = sorted((0, other_bound))

        if power is not None and effect_size is not None and sample_count is None:
            def fn(x: float | int) -> float:
                return power - self._power_from_stats(
                    sample_var=sample_var,
                    sample_count=x,
                    effect_size=effect_size,
                )
            lower_bound = 3
            upper_bound = _find_boundary(fn, 10)

        return scipy.optimize.brentq(fn, lower_bound, upper_bound, maxiter=MAX_ITER)


    def _power_from_stats(
        self,
        sample_var: float,
        sample_count: int | float,
        effect_size: float,
    ) -> float:
        null_distr, alt_distr, _ = self._distr_and_scale(
            contr_var=sample_var,
            contr_count=sample_count / (1 + self.ratio),
            treat_var=sample_var,
            treat_count=sample_count * self.ratio / (1 + self.ratio),
            effect_size=effect_size,
        )
        if self.alternative == "greater":
            stat_critical = null_distr.isf(self.alpha)
            return float(alt_distr.sf(stat_critical))
        if self.alternative == "less":
            stat_critical = null_distr.ppf(self.alpha)
            return float(alt_distr.cdf(stat_critical))
        # two-sided
        stat_critical = null_distr.isf(self.alpha / 2)
        return float(alt_distr.cdf(-stat_critical) + alt_distr.sf(stat_critical))


    @overload
    def _distr_and_scale(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: None = None,
    ) -> tuple[scipy.stats.rv_frozen, None, float]:
        ...

    @overload
    def _distr_and_scale(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: float,
    ) -> tuple[scipy.stats.rv_frozen, scipy.stats.rv_frozen, float]:
        ...

    def _distr_and_scale(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: float | None = None,
    ) -> tuple[scipy.stats.rv_frozen, scipy.stats.rv_frozen | None, float]:
        if self.equal_var:
            pooled_var = (
                (contr_count - 1)*contr_var + (treat_count - 1)*treat_var
            ) / (contr_count + treat_count - 2)
            scale = math.sqrt(pooled_var/contr_count + pooled_var/treat_count)
        else:
            scale = math.sqrt(contr_var/contr_count + treat_var/treat_count)

        if self.use_t:
            if self.equal_var:
                df = contr_count + treat_count - 2
            else:
                contr_mean_var = contr_var / contr_count
                treat_mean_var = treat_var / treat_count
                df = (contr_mean_var + treat_mean_var)**2 / (
                    contr_mean_var**2 / (contr_count - 1)
                    + treat_mean_var**2 / (treat_count - 1)
                )
            null_distr = scipy.stats.t(df=df, scale=scale)
            alt_distr = None if effect_size is None else scipy.stats.nct(
                df=df, nc=effect_size / scale, scale=scale)
        else:
            null_distr = scipy.stats.norm(scale=scale)
            alt_distr = None if effect_size is None else scipy.stats.norm(
                loc=effect_size, scale=scale)

        return null_distr, alt_distr, scale


def _find_boundary(
    fn: Callable[[float | int], float],
    init: float | int,
    mult: float | int = 10,
) -> float:
    b = init
    i = 0
    while fn(b) > 0:
        b *= mult
        i += 1
        if i == MAX_ITER:
            raise RuntimeError(
                "Cannot find parameter boundaries. "
                "Maximum number of iterations is reached.",
            )
    return b


def _to_seq[N: float | int | None](x: N | Sequence[N]) -> Sequence[N]:
    if isinstance(x, Sequence):
        return x  # ty:ignore[invalid-return-type]
    return (x,)


class Mean(RatioOfMeans):  # noqa: D101
    def __init__(  # noqa: PLR0913
        self,
        column: str,
        covariate: str | None = None,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
        alpha: float | None = None,
        ratio: float | int | None = None,
        power: float | None = None,
        effect_size: float | int | Sequence[float | int] | None = None,
        rel_effect_size: float | Sequence[float] | None = None,
        n_obs: int | Sequence[int] | None = None,
        covariate_fit: CovariateFit | Mapping[str, object] | None = None,
    ) -> None:
        """Metric for the analysis of means.

        Args:
            column: Metric column name.
            covariate: Metric covariate column name.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than
                    the mean in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).
            alpha: Significance level. Only for the analysis of power.
            ratio: Ratio of the number of observations in the treatment
                relative to the control. Only for the analysis of power.
            power: Statistical power. Only for the analysis of power.
            effect_size: Absolute effect size. Difference between the two means.
                Only for the analysis of power.
            rel_effect_size: Relative effect size. Difference between the two means,
                divided by the control mean. Only for the analysis of power.
            n_obs: Number of observations in the control and in the treatment together.
                Only for the analysis of power.
            covariate_fit: Frozen covariate fit estimated on independent
                reference data. If `None`, the covariate coefficient and center
                are estimated from the analyzed data (CUPED/CUPAC). If defined,
                the metric applies the frozen coefficient and center and never
                re-estimates them from the analyzed data.

        Parameter defaults:
            Defaults for parameters `alpha`, `alternative`, `confidence_level`,
            `equal_var`, `n_obs`, `power`, `ratio`, and `use_t` can be changed
            using the `config_context` and `set_config` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Mean("orders"),
            ...     revenue_per_user=tt.Mean("revenue"),
            ... )
            >>> data = tt.make_users_data(rng=42)
            >>> result = experiment.analyze(data)
            >>> result
            metric           control treatment rel_effect_size rel_effect_size_ci pvalue
            orders_per_user    0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            With CUPED:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Mean("orders", "orders_covariate"),
            ...     revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
            ... )
            >>> data = tt.make_users_data(rng=42, covariates=True)
            >>> result = experiment.analyze(data)
            >>> result
            metric           control treatment rel_effect_size rel_effect_size_ci  pvalue
            orders_per_user    0.523     0.581             11%        [2.9%, 20%] 0.00733
            revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00674

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     rng=42,
            ...     sessions_uplift=0,
            ...     orders_uplift=0,
            ...     revenue_uplift=0,
            ...     covariates=True,
            ... )
            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     n_obs=(10_000, 20_000),
            ... )
            >>> # Solve for effect size.
            >>> orders_per_user.solve_power(data)
            power effect_size rel_effect_size n_obs
              80%      0.0374            7.2% 10000
              80%      0.0264            5.1% 20000

            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     rel_effect_size=0.05,
            ... )
            >>> # Solve for the total number of observations.
            >>> orders_per_user.solve_power(data, "n_obs")
            power effect_size rel_effect_size n_obs
              80%      0.0260            5.0% 20733

            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     rel_effect_size=0.1,
            ... )
            >>> # Solve for power. Infer number of observations from the sample.
            >>> orders_per_user.solve_power(data, "power")
            power effect_size rel_effect_size n_obs
              69%      0.0519             10%  4000

            ```
        """  # noqa: E501
        super().__init__(
            numer=column,
            denom=None,
            numer_covariate=covariate,
            denom_covariate=None,
            alternative=alternative,
            confidence_level=confidence_level,
            equal_var=equal_var,
            use_t=use_t,
            alpha=alpha,
            ratio=ratio,
            power=power,
            effect_size=effect_size,
            rel_effect_size=rel_effect_size,
            n_obs=n_obs,
            covariate_fit=covariate_fit,
        )
        self.column = column
        self.covariate = covariate
