from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from loguru import logger

from wip_analysis.data_loader.postgres_loader import fetch_data
from wip_analysis.data_loader.sql_template import render_sql


@dataclass(frozen=True)
class LotIdLegalityResult:
    lot_id: str
    exists: bool
    is_legal: bool
    detail: str
    distinct_product_count: int
    unique_product_id: Optional[str]


@lru_cache(maxsize=512)
def check_lot_id_legality(lot_id: str) -> LotIdLegalityResult:
    normalized_lot_id = lot_id.strip()
    if not normalized_lot_id:
        return LotIdLegalityResult(
            lot_id=lot_id,
            exists=False,
            is_legal=False,
            detail="lot_id is empty",
            distinct_product_count=0,
            unique_product_id=None,
        )

    query = render_sql("check_lot_id_legality.sql.j2", {"lot_id": normalized_lot_id})
    result_df = fetch_data(query)

    if result_df.is_empty():
        logger.warning("Legality query returned no rows for lot_id={}", normalized_lot_id)
        return LotIdLegalityResult(
            lot_id=normalized_lot_id,
            exists=False,
            is_legal=False,
            detail="lot_id not found",
            distinct_product_count=0,
            unique_product_id=None,
        )

    row = result_df.row(0, named=True)
    row_count = int(row.get("row_count") or 0)
    parent_match_count = int(row.get("parent_match_count") or 0)
    distinct_product_count = int(row.get("distinct_product_count") or 0)
    unique_product_id = row.get("unique_product_id")

    if row_count == 0:
        return LotIdLegalityResult(
            lot_id=normalized_lot_id,
            exists=False,
            is_legal=False,
            detail="lot_id not found",
            distinct_product_count=0,
            unique_product_id=None,
        )

    if parent_match_count != row_count:
        invalid_rows = row_count - parent_match_count
        return LotIdLegalityResult(
            lot_id=normalized_lot_id,
            exists=True,
            is_legal=False,
            detail=(
                f"lot_id is illegal: parent_lot_id must equal lot_id (violations={invalid_rows})"
            ),
            distinct_product_count=distinct_product_count,
            unique_product_id=unique_product_id,
        )

    if distinct_product_count != 1:
        return LotIdLegalityResult(
            lot_id=normalized_lot_id,
            exists=True,
            is_legal=False,
            detail=(
                "lot_id is illegal: lot_id must map to exactly one distinct product_id "
                f"(found={distinct_product_count})"
            ),
            distinct_product_count=distinct_product_count,
            unique_product_id=unique_product_id,
        )

    return LotIdLegalityResult(
        lot_id=normalized_lot_id,
        exists=True,
        is_legal=True,
        detail="ok",
        distinct_product_count=distinct_product_count,
        unique_product_id=unique_product_id,
    )


def clear_lot_id_legality_cache() -> None:
    check_lot_id_legality.cache_clear()
