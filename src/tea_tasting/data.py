"""Data access helpers and backend dispatch."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Literal, TypeGuard, overload

import narwhals.typing
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.backends
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types


type Table = (
    tea_tasting.backends.BaseTable |
    narwhals.typing.IntoFrame |
    narwhals.typing.Frame
)
type AggregatesByVariant = Mapping[Hashable, tea_tasting.aggr.Aggregates]
type TablesByVariant = Mapping[Hashable, pa.Table]


@overload
def read_aggregates(
    data: AggregatesByVariant,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: str | None = None,
) -> AggregatesByVariant:
    ...

@overload
def read_aggregates(
    data: Table | tea_tasting.aggr.Aggregates,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def read_aggregates(
    data: Table,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def read_aggregates(
    data: Table
        | tea_tasting.aggr.Aggregates
        | AggregatesByVariant,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: str | None = None,
) -> tea_tasting.aggr.Aggregates | AggregatesByVariant:
    """Read aggregated statistics.

    Args:
        data: Experimental data or aggregated statistics.
        aggr_cols: Columns to be aggregated.
        variant: Variant column name.

    Returns:
        Aggregated statistics.
    """
    if _is_aggregates_mapping(data):
        return data
    if isinstance(data, tea_tasting.aggr.Aggregates):
        if variant is not None:
            raise ValueError("The variant parameter is not supported for Aggregates.")
        return data

    table = _table(data)  # ty: ignore[invalid-argument-type]
    if variant is not None:
        table = table.group_by(variant)
    return table.aggregate(aggr_cols)


@overload
def read_granular(
    data: Table,
    cols: Sequence[str] = (),
    variant: None = None,
) -> pa.Table:
    ...

@overload
def read_granular(
    data: TablesByVariant,
    cols: Sequence[str] = (),
    variant: str | None = None,
) -> TablesByVariant:
    ...

@overload
def read_granular(
    data: Table,
    cols: Sequence[str],
    variant: str,
) -> dict[Hashable, pa.Table]:
    ...

def read_granular(
    data: Table | TablesByVariant,
    cols: Sequence[str] = (),
    variant: str | None = None,
) -> pa.Table | TablesByVariant:
    """Read granular experimental data.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant: Variant column name.

    Returns:
        Experimental data as a PyArrow Table or as PyArrow Tables by variant.
    """
    if _is_tables_mapping(data):
        return data

    variant_cols = () if variant is None else (variant,)
    table = _table(data).select(*cols, *variant_cols)  # ty: ignore[invalid-argument-type]
    if variant is None:
        return table

    variant_array = table[variant]
    table = table.select(cols) if len(cols) > 0 else table.select([])
    return {
        var: table.filter(pc.equal(variant_array, pa.scalar(var)))  # ty:ignore[unresolved-attribute]
        for var in variant_array.unique().to_pylist()
    }


def read_variants(data: Table, variant: str) -> list[Hashable]:
    """Read unique variant values.

    Args:
        data: Experimental data.
        variant: Variant column name.

    Returns:
        Unique variant values.
    """
    return _table(data).select_col_unique(variant)


def read_paired(
    data: Table,
    pair_id: str,
    variant: str,
    cols: Sequence[str] = (),
    *,
    missing_side: Literal["drop", "raise"] = "drop",
) -> dict[Hashable, pa.Table]:
    """Read pair-aligned granular experimental data.

    Each matching pair contributes one control observation and one treatment
    observation. The returned tables are aligned: the row at position `i`
    belongs to the same `pair_id` in every variant. Rows are ordered by
    `pair_id`, so the result does not depend on the order of observations
    in the input data.

    The function validates the pair structure before returning data:

    - `pair_id` must not be null.
    - Each variant must occur at most once in each pair. Duplicates raise
      a `ValueError`, because a pair cannot contribute more than one
      observation to the same side.
    - A pair is *complete* if it contains exactly one observation of every
      variant present in the data. Pairs with a missing side are isolated
      according to the `missing_side` rule: they are either dropped
      (`"drop"`) or reported with a `ValueError` (`"raise"`).

    Args:
        data: Experimental data.
        pair_id: Pair identifier column name.
        variant: Variant column name.
        cols: Columns to read.
        missing_side: Rule for pairs missing a side:

            - `"drop"`: exclude incomplete pairs from the result.
            - `"raise"`: raise an exception if any pair is incomplete.

    Returns:
        Experimental data as PyArrow Tables by variant, aligned by `pair_id`.
        Each table contains the `pair_id` column in addition to `cols`.

    Raises:
        ValueError: If `pair_id` or `variant` contains nulls, if a pair
            contains more than one observation of the same variant, or if
            `missing_side="raise"` and some pairs are incomplete.

    Examples:
        ```pycon
        >>> import pyarrow as pa
        >>> import tea_tasting.data

        >>> data = pa.table({
        ...     "pair_id": [0, 0, 1, 1, 2],
        ...     "variant": [0, 1, 0, 1, 0],
        ...     "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        ... })
        >>> paired = tea_tasting.data.read_paired(
        ...     data, pair_id="pair_id", variant="variant", cols=("value",))
        >>> paired[0].to_pydict()
        {'pair_id': [0, 1], 'value': [1.0, 3.0]}
        >>> paired[1].to_pydict()
        {'pair_id': [0, 1], 'value': [2.0, 4.0]}

        ```
    """
    tea_tasting.utils.check_scalar(pair_id, "pair_id", typ=str)
    tea_tasting.utils.check_scalar(variant, "variant", typ=str)
    tea_tasting.utils.check_scalar(
        missing_side, "missing_side", typ=str, in_={"drop", "raise"})
    if pair_id == variant:
        raise ValueError("pair_id and variant must be different columns.")

    names = _unique_names(cols, pair_id, variant)
    table = _table(data).select(*names)

    _check_no_nulls(table.column(pair_id), pair_id)
    _check_no_nulls(table.column(variant), variant)
    _check_unique_sides(table, pair_id, variant)

    variants = sorted(table.column(variant).unique().to_pylist())
    incomplete = _incomplete_pair_ids(table, pair_id, len(variants))
    if len(incomplete) > 0:
        if missing_side == "raise":
            raise ValueError(
                "Some pairs are missing a variant side: "
                f"{_format_values(incomplete)}. "
                "Use missing_side='drop' to exclude such pairs.",
            )
        table = table.filter(_not_in(table.column(pair_id), incomplete))

    table = table.sort_by(pair_id)
    return {
        var: table.filter(
            pc.equal(table.column(variant), pa.scalar(var)),  # ty:ignore[missing-attribute]
        ).drop_columns(variant)
        for var in variants
    }


def _unique_names(
    cols: Sequence[str],
    pair_id: str,
    variant: str,
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((pair_id, variant, *cols)))


def _check_no_nulls(values: pa.ChunkedArray, name: str) -> None:
    if values.null_count > 0:
        raise ValueError(f"{name} must not contain null values.")


def _check_unique_sides(
    table: pa.Table,
    pair_id: str,
    variant: str,
) -> None:
    counts = (
        table.group_by([pair_id, variant])
        .aggregate([(pair_id, "count")])
    )
    count_col = counts.column_names[-1]
    duplicates = counts.filter(
        pc.greater(counts.column(count_col), pa.scalar(1)),  # ty:ignore[missing-attribute]
    )
    if duplicates.num_rows > 0:
        pairs = duplicates.column(pair_id).to_pylist()
        raise ValueError(
            "Each pair must contain at most one observation per variant. "
            f"Duplicated pair sides: {_format_values(pairs)}.",
        )


def _incomplete_pair_ids(
    table: pa.Table,
    pair_id: str,
    n_variants: int,
) -> list[Hashable]:
    counts = table.group_by(pair_id).aggregate([(pair_id, "count")])
    count_col = counts.column_names[1]
    incomplete = counts.filter(
        pc.not_equal(counts.column(count_col), pa.scalar(n_variants)),  # ty:ignore[missing-attribute]
    )
    return incomplete.column(pair_id).to_pylist()


def _not_in(values: pa.ChunkedArray, excluded: Sequence[Hashable]) -> pa.Array:
    return pc.invert(pc.is_in(values, value_set=pa.array(excluded)))  # ty:ignore[missing-attribute]


def _format_values(values: Sequence[Hashable], max_values: int = 10) -> str:
    shown = list(values[:max_values])
    suffix = ", ..." if len(values) > max_values else ""
    return f"{shown}{suffix}"


def _table(data: Table) -> tea_tasting.backends.BaseTable:
    if isinstance(data, tea_tasting.backends.BaseTable):
        return data
    if _is_ibis_table(data):
        return tea_tasting.backends.IbisTable(data)
    return tea_tasting.backends.NarwhalsFrame(data)


def _is_aggregates_mapping(data: object) -> TypeGuard[AggregatesByVariant]:
    return (
        isinstance(data, Mapping)
        and all(
            isinstance(value, tea_tasting.aggr.Aggregates)
            for value in data.values()
        )
    )


def _is_tables_mapping(data: object) -> TypeGuard[TablesByVariant]:
    return (
        isinstance(data, Mapping)
        and all(isinstance(value, pa.Table) for value in data.values())
    )


def _is_ibis_table(data: object) -> TypeGuard[ibis.expr.types.Table]:
    try:
        import ibis.expr.types  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name == "ibis":
            return False
        raise

    return isinstance(data, ibis.expr.types.Table)
