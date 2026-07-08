"""馆藏匹配模块：负责文本规范化、候选检索和匹配置信度计算。"""

from book_inventory.matching.catalog_matcher import (
    CatalogEntry,
    CatalogMatcher,
    MatchCandidate,
    make_grams,
    normalize_for_match,
    score_title_similarity,
)

__all__ = [
    "CatalogEntry",
    "CatalogMatcher",
    "MatchCandidate",
    "make_grams",
    "normalize_for_match",
    "score_title_similarity",
]
