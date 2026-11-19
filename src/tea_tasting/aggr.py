"""Module for working with aggregated statistics: count, mean, var, cov."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


type CommonObservationsPolicy = Literal["drop", "raise"]

_POPULATION_FINGERPRINT_VERSION = "1"


class AggrCols(tea_tasting.utils.ReprMixin):  # noqa: D101
    has_count: bool
    mean_cols: tuple[str, ...]
    var_cols: tuple[str, ...]
    cov_cols: tuple[tuple[str, str], ...]

    def __init__(
        self,
        has_count: bool = False,  # noqa: FBT001, FBT002
        mean_cols: Sequence[str] = (),
        var_cols: Sequence[str] = (),
        cov_cols: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Columns to be aggregated for a metric analysis.

        Args:
            has_count: If `True`, include the sample size.
            mean_cols: Column names for calculation of sample means.
            var_cols: Column names for calculation of sample variances.
            cov_cols: Pairs of column names for calculation of sample covariances.
        """
        self.has_count = tea_tasting.utils.check_scalar(
            has_count,
            "has_count",
            typ=bool,
        )
        self.mean_cols, self.var_cols, self.cov_cols = _validate_aggr_cols(
            mean_cols,
            var_cols,
            cov_cols,
        )

    def __or__(self, other: AggrCols) -> AggrCols:
        """Merge two aggregation column specifications.

        Args:
            other: Second object.

        Returns:
            Merged column specifications.
        """
        return AggrCols(
            has_count=self.has_count or other.has_count,
            mean_cols=(*self.mean_cols, *other.mean_cols),
            var_cols=(*self.var_cols, *other.var_cols),
            cov_cols=(*self.cov_cols, *other.cov_cols),
        )

    def __len__(self) -> int:
        """Total length of all object attributes.

        If has_count is True then its value is 1, or 0 otherwise.

        Returns:
            Total length of all object attributes.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )

    @property
    def dep_cols(self) -> tuple[str, ...]:
        """Columns that an observation must define to contribute to all statistics.

        A missing value in any of these columns makes some statistics
        (a mean, a variance, or a covariance) be calculated on a subset
        of observations. Columns are sorted to provide a deterministic order.

        Returns:
            Columns required for the aggregation statistics.
        """
        cols: set[str] = set(self.mean_cols) | set(self.var_cols)
        for left, right in self.cov_cols:
            cols.update((left, right))
        return tuple(sorted(cols))


class Population(tea_tasting.utils.ReprMixin):  # noqa: D101
    fingerprint: str
    count: int
    columns: tuple[str, ...]

    def __init__(
        self,
        fingerprint: str,
        count: int,
        columns: Sequence[str] = (),
    ) -> None:
        """Definition and size of the population of observations.

        A population is defined by the set of columns that every included
        observation must define (non-null). The fingerprint uniquely identifies
        this set of columns. The count is the actual number of observations
        in the population.

        Args:
            fingerprint: Population fingerprint, usually produced by
                `population_fingerprint`.
            count: Number of observations in the population.
            columns: Columns that observations must define to be included
                in the population.

        Examples:
            ```pycon
            >>> import tea_tasting.aggr

            >>> columns = tea_tasting.aggr.AggrCols(
            ...     mean_cols=("orders",),
            ...     var_cols=("orders",),
            ... ).dep_cols
            >>> population = tea_tasting.aggr.Population(
            ...     fingerprint=tea_tasting.aggr.population_fingerprint(columns),
            ...     count=123,
            ...     columns=columns,
            ... )
            >>> population
            Population(fingerprint='8e6cb8347953971e', count=123, columns=('orders',))

            ```
        """
        self.fingerprint = tea_tasting.utils.check_scalar(
            fingerprint,
            "fingerprint",
            typ=str,
        )
        self.count = tea_tasting.utils.check_scalar(count, "count", typ=int, ge=0)
        self.columns = _validate_population_columns(columns)


def population_fingerprint(columns: Sequence[str]) -> str:
    """Calculate the fingerprint of a population definition.

    The fingerprint depends only on the set of columns that observations
    must define. Two analyses that require the same set of columns share
    the same population fingerprint, regardless of the variant column
    or of how missing values were handled.

    Args:
        columns: Columns that observations must define to be included
            in the population.

    Returns:
        Population fingerprint.

    Examples:
        ```pycon
        >>> import tea_tasting.aggr

        >>> tea_tasting.aggr.population_fingerprint(("orders", "sessions"))
        '5289727d3f614617'
        >>> tea_tasting.aggr.population_fingerprint(("sessions", "orders"))
        '5289727d3f614617'

        ```
    """
    columns = _validate_population_columns(columns)
    payload = "\n".join((_POPULATION_FINGERPRINT_VERSION, *sorted(columns)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_population_columns(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        tea_tasting.utils.check_scalar(col, "column", typ=str)
        for col in columns
    ))


class Aggregates(tea_tasting.utils.ReprMixin):  # noqa: D101
    count_: int | None
    mean_: dict[str, float | int]
    var_: dict[str, float | int]
    cov_: dict[tuple[str, str], float | int]
    population_: Population | None

    def __init__(
        self,
        count_: int | None = None,
        mean_: dict[str, float | int] = {},  # noqa: B006
        var_: dict[str, float | int] = {},  # noqa: B006
        cov_: dict[tuple[str, str], float | int] = {},  # noqa: B006
        population_: Population | None = None,
    ) -> None:
        """Aggregated statistics.

        Args:
            count_: Sample size (number of observations).
            mean_: Dictionary of sample means with variable names as keys.
            var_: Dictionary of sample variances with variable names as keys.
            cov_: Dictionary of sample covariances with pairs of variable names as keys.
            population_: Population of observations used for the aggregation.
                If `None`, the aggregation is assumed to be calculated on all
                available observations, potentially a different subset
                for each statistic.
        """
        self.count_ = count_
        self.mean_ = mean_
        self.var_ = var_
        self.cov_ = {_sorted_tuple(*k): v for k, v in cov_.items()}
        self.population_ = population_

    @property
    def population(self) -> Population | None:
        """Population of observations used for the aggregation, if known.

        Returns:
            Population of observations or `None` if not known.
        """
        return self.population_

    def with_population(
        self,
        columns: Sequence[str],
        *,
        fingerprint: str | None = None,
    ) -> Aggregates:
        """Return aggregates annotated with the population contract.

        Use this method to attach a population contract to statistics
        calculated outside of tea-tasting, so that experiments analyzed with
        the `common_observations` policy can verify that all statistics
        are calculated on the same population of observations.

        Args:
            columns: Columns that observations must define (be non-null)
                to be included in the population.
            fingerprint: Population fingerprint. If `None`, calculate it
                from `columns` using `population_fingerprint`.

        Returns:
            A copy of the aggregates with the population of observations.

        Raises:
            ValueError: If the sample size (`count_`) is `None`, because
                the population size is a part of the contract.
        """
        columns = _validate_population_columns(columns)
        if self.count_ is None:
            raise ValueError(
                "The sample size is required to define a population contract. "
                "Compute aggregates with `AggrCols(has_count=True)`.",
            )
        return Aggregates(
            count_=self.count_,
            mean_=self.mean_,
            var_=self.var_,
            cov_=self.cov_,
            population_=Population(
                fingerprint=(
                    population_fingerprint(columns)
                    if fingerprint is None
                    else fingerprint
                ),
                count=self.count_,
                columns=columns,
            ),
        )

    def with_zero_div(self) -> Aggregates:
        """Return aggregates that do not raise an error on division by zero.

        Division by zero returns:

        - `inf` if numerator is greater than `0`,
        - `nan` if numerator is equal to or less than `0`.
        """
        return Aggregates(
            count_=None if self.count_ is None else tea_tasting.utils.Int(self.count_),
            mean_={k: tea_tasting.utils.numeric(v) for k, v in self.mean_.items()},
            var_={k: tea_tasting.utils.numeric(v) for k, v in self.var_.items()},
            cov_={k: tea_tasting.utils.numeric(v) for k, v in self.cov_.items()},
            population_=self.population_,
        )

    def count(self) -> int:
        """Sample size (number of observations).

        Returns:
            Sample size (number of observations).
        """
        if self.count_ is None:
            raise RuntimeError("Count is None.")
        return self.count_

    def mean(self, name: str | None) -> float | int:
        """Sample mean.

        Assume the variable is a constant `1` if the variable name is `None`.

        Args:
            name: Variable name.

        Returns:
            Sample mean.
        """
        if name is None:
            return 1
        return self.mean_[name]

    def var(self, name: str | None) -> float | int:
        """Sample variance.

        Assume the variable is a constant if the variable name is `None`.

        Args:
            name: Variable name.

        Returns:
            Sample variance.
        """
        if name is None:
            return 0
        return self.var_[name]

    def cov(self, left: str | None, right: str | None) -> float | int:
        """Sample covariance.

        Assume the variable is a constant if the variable name is `None`.

        Args:
            left: First variable name.
            right: Second variable name.

        Returns:
            Sample covariance.
        """
        if left is None or right is None:
            return 0
        return self.cov_[_sorted_tuple(left, right)]

    def ratio_var(
        self,
        numer: str | None,
        denom: str | None,
    ) -> float | int:
        """Sample variance of the ratio of two variables using the delta method.

        Args:
            numer: Numerator variable name.
            denom: Denominator variable name.

        Returns:
            Sample variance of the ratio of two variables.

        References:
            - [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            - [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).
        """
        numer_mean_sq = self.mean(numer) * self.mean(numer)
        denom_mean_sq = self.mean(denom) * self.mean(denom)
        return (
            self.var(numer)
            - 2 * self.cov(numer, denom) * self.mean(numer) / self.mean(denom)
            + self.var(denom) * numer_mean_sq / denom_mean_sq
        ) / denom_mean_sq

    def ratio_cov(
        self,
        left_numer: str | None,
        left_denom: str | None,
        right_numer: str | None,
        right_denom: str | None,
    ) -> float | int:
        """Sample covariance of the ratios of variables using the delta method.

        Args:
            left_numer: First numerator variable name.
            left_denom: First denominator variable name.
            right_numer: Second numerator variable name.
            right_denom: Second denominator variable name.

        Returns:
            Sample covariance of the ratios of variables.

        References:
            - [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            - [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).
        """
        left_ratio_of_means = self.mean(left_numer) / self.mean(left_denom)
        right_ratio_of_means = self.mean(right_numer) / self.mean(right_denom)
        return (
            self.cov(left_numer, right_numer)
            - self.cov(left_numer, right_denom) * right_ratio_of_means
            - self.cov(left_denom, right_numer) * left_ratio_of_means
            + self.cov(left_denom, right_denom)
                * left_ratio_of_means * right_ratio_of_means
        ) / self.mean(left_denom) / self.mean(right_denom)

    def __add__(self, other: Aggregates) -> Aggregates:
        """Calculate the aggregated statistics of the concatenation of two samples.

        Samples are assumed to be independent.

        Args:
            other: Aggregated statistics of the second sample.

        Returns:
            Aggregated statistics of the concatenation of two samples.
        """
        return Aggregates(
            count_=self.count() + other.count() if self.count_ is not None else None,
            mean_={col: _add_mean(self, other, col) for col in self.mean_},
            var_={col: _add_var(self, other, col) for col in self.var_},
            cov_={cols: _add_cov(self, other, cols) for cols in self.cov_},
            population_=_add_population(self.population_, other.population_),
        )


def _add_population(
    left: Population | None,
    right: Population | None,
) -> Population | None:
    if (
        left is None or
        right is None or
        left.fingerprint != right.fingerprint
    ):
        return None
    return Population(
        fingerprint=left.fingerprint,
        count=left.count + right.count,
        columns=left.columns,
    )


def _add_mean(left: Aggregates, right: Aggregates, col: str) -> float | int:
    sum_ = left.count()*left.mean(col) + right.count()*right.mean(col)
    count = left.count() + right.count()
    return sum_ / count

def _add_var(left: Aggregates, right: Aggregates, col: str) -> float | int:
    left_n = left.count()
    right_n = right.count()
    total_n = left_n + right_n
    diff_of_means = left.mean(col) - right.mean(col)
    return (
        left.var(col) * (left_n - 1)
        + right.var(col) * (right_n - 1)
        + diff_of_means * diff_of_means * left_n * right_n / total_n
    ) / (total_n - 1)

def _add_cov(left: Aggregates, right: Aggregates, cols: tuple[str, str]) -> float | int:
    left_n = left.count()
    right_n = right.count()
    total_n = left_n + right_n
    diff_of_means0 = left.mean(cols[0]) - right.mean(cols[0])
    diff_of_means1 = left.mean(cols[1]) - right.mean(cols[1])
    return (
        left.cov(*cols) * (left_n - 1)
        + right.cov(*cols) * (right_n - 1)
        + diff_of_means0 * diff_of_means1 * left_n * right_n / total_n
    ) / (total_n - 1)


def _validate_aggr_cols(
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    mean_cols = tuple({*mean_cols})
    var_cols = tuple({*var_cols})
    cov_cols = tuple({_sorted_tuple(left, right) for left, right in cov_cols})
    return mean_cols, var_cols, cov_cols


def _sorted_tuple(left: str, right: str) -> tuple[str, str]:
    if right < left:
        return right, left
    return left, right
