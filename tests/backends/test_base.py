from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tea_tasting.aggr
import tea_tasting.backends.base


if TYPE_CHECKING:
    from tea_tasting.backends.base import BaseTableGroupBy


def test_get_aggregates_handles_folded_aliases() -> None:
    aggr = tea_tasting.backends.base._get_aggregates(
        [{
            "__COUNT__": 2,
            "__MEAN__X__": 1.5,
            "__VAR__X__": 0.5,
            "__COV__X__Y__": 0.25,
        }],
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("x",),
            var_cols=("x",),
            cov_cols=(("x", "y"),),
        ),
    )

    assert aggr.count_ == 2
    assert aggr.mean_ == {"x": 1.5}
    assert aggr.var_ == {"x": 0.5}
    assert aggr.cov_ == {("x", "y"): 0.25}


def test_get_aggregates_handles_folded_group_column() -> None:
    aggrs = tea_tasting.backends.base._get_aggregates(
        [
            {"VARIANT": 0, "__COUNT__": 2},
            {"VARIANT": 1, "__COUNT__": 3},
        ],
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=(),
            var_cols=(),
            cov_cols=(),
        ),
        "variant",
    )

    assert set(aggrs) == {0, 1}
    assert aggrs[0].count_ == 2
    assert aggrs[1].count_ == 3


def test_get_aggregates_fails_on_ambiguous_folded_alias() -> None:
    with pytest.raises(KeyError, match="Ambiguous result column name"):
        tea_tasting.backends.base._get_aggregates(
            [{
                "__MEAN__X__": 1,
                "__Mean__x__": 2,
            }],
            tea_tasting.aggr.AggrCols(
                has_count=False,
                mean_cols=("x",),
                var_cols=(),
                cov_cols=(),
            ),
        )


def test_get_aggregates_converts_null_to_nan() -> None:
    aggr = tea_tasting.backends.base._get_aggregates(
        [{"__MEAN__X__": None}],
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=("x",),
            var_cols=(),
            cov_cols=(),
        ),
    )

    assert aggr.mean_["x"] != aggr.mean_["x"]


def test_base_table_filter_complete_default() -> None:
    import pyarrow as pa  # noqa: PLC0415

    import tea_tasting.backends  # noqa: PLC0415

    class _Table(tea_tasting.backends.BaseTable):
        def __init__(self, data: pa.Table) -> None:
            self.data = data

        def select(self, *cols: str) -> pa.Table:
            return self.data.select(cols) if len(cols) > 0 else self.data

        def select_col_unique(self, col: str) -> list[object]:
            return self.data[col].unique().to_pylist()

        def group_by(self, by: str) -> BaseTableGroupBy:
            raise NotImplementedError

        def aggregate(
            self,
            aggr_cols: tea_tasting.aggr.AggrCols,
        ) -> tea_tasting.aggr.Aggregates:
            raise NotImplementedError

    data = pa.table({
        "a": pa.array([1.0, None, 3.0, 4.0]),
        "b": pa.array([1.0, 2.0, None, 4.0]),
    })
    table = _Table(data)

    filtered, total, complete = table.filter_complete(("a", "b"))

    assert total == 4
    assert complete == 2
    assert filtered.select().num_rows == 2


def test_base_table_filter_complete_default_empty_cols() -> None:
    import pyarrow as pa  # noqa: PLC0415

    import tea_tasting.backends  # noqa: PLC0415

    class _Table(tea_tasting.backends.BaseTable):
        def __init__(self, data: pa.Table) -> None:
            self.data = data

        def select(self, *cols: str) -> pa.Table:
            return self.data.select(cols) if len(cols) > 0 else self.data

        def select_col_unique(self, col: str) -> list[object]:  # noqa: ARG002
            return []

        def group_by(self, by: str) -> BaseTableGroupBy:
            raise NotImplementedError

        def aggregate(
            self,
            aggr_cols: tea_tasting.aggr.AggrCols,
        ) -> tea_tasting.aggr.Aggregates:
            raise NotImplementedError

    data = pa.table({"a": pa.array([1.0, None, 3.0, 4.0])})
    table = _Table(data)

    filtered, total, complete = table.filter_complete(())

    assert total == complete == 4
    assert filtered is table
