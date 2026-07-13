"""馆藏目录书名匹配模块。

OCR 输出通常不是干净书名，常会混入作者、出版社、索书号、年份等文本。
本模块的目标不是“证明 OCR 完美”，而是把 OCR 的候选文本拉回馆藏目录中的规范书名。

实现策略：
1. 读取 `catalog.csv` 中的规范书目；
2. 对馆藏书名建立字符 n-gram 倒排索引；
3. 查询时先用倒排索引找少量候选，再计算相似度；
4. 相似度低于阈值时返回“待确认”，避免强制错误匹配。
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass(frozen=True)
class CatalogEntry:
    """馆藏目录中的一条规范书目。"""

    metadata_id: str
    title: str
    title_normalized: str
    author: str
    call_number: str
    publisher: str
    total_copy_count: int
    match_text: str
    search_text: str


@dataclass(frozen=True)
class MatchCandidate:
    """一次 OCR 查询对应的馆藏匹配候选。"""

    entry: CatalogEntry
    score: float
    status: str


class CatalogMatcher:
    """馆藏目录相似度匹配器。"""

    def __init__(
        self,
        entries: list[CatalogEntry],
        *,
        gram_size: int = 2,
        max_candidates: int = 800,
    ) -> None:
        """初始化匹配器并建立倒排索引。

        Args:
            entries: 馆藏目录条目。
            gram_size: 字符片段长度。中文书名使用 2-gram 通常比较稳。
            max_candidates: 每次查询最多进入精排的候选数量。
        """

        self.entries = entries
        self.gram_size = gram_size
        self.max_candidates = max_candidates
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        self.call_item_index: dict[str, set[int]] = defaultdict(set)
        self._build_index()

    @classmethod
    def from_csv(cls, catalog_path: str | Path) -> "CatalogMatcher":
        """从 `catalog.csv` 构建匹配器。"""

        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到馆藏目录 CSV：{path}")

        entries: list[CatalogEntry] = []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                title = clean_catalog_cell(row.get("title"))
                title_normalized = clean_catalog_cell(row.get("title_normalized")) or title
                match_text = normalize_for_match(title_normalized or title)
                if not match_text:
                    continue

                entries.append(
                    CatalogEntry(
                        metadata_id=clean_catalog_cell(row.get("metadata_id")),
                        title=title,
                        title_normalized=title_normalized,
                        author=clean_catalog_cell(row.get("author")),
                        call_number=clean_catalog_cell(row.get("call_number")),
                        publisher=clean_catalog_cell(row.get("publisher")),
                        total_copy_count=_safe_int(row.get("total_copy_count")),
                        match_text=match_text,
                        # 倒排索引用书名 + 作者 + 索书号 + 出版社。
                        # 这样当 OCR 把书名读崩、但读到了作者或索书号时，仍能召回正确候选。
                        search_text=" ".join(
                            [
                                normalize_for_match(
                                    " ".join(
                                        [
                                            title_normalized,
                                            title,
                                            clean_catalog_cell(row.get("author")),
                                            clean_catalog_cell(row.get("publisher")),
                                        ]
                                    )
                                ),
                                # 索书号不能走书名清洗逻辑，否则 F272/202 会被当噪声删掉。
                                normalize_identifier(clean_catalog_cell(row.get("call_number"))),
                            ]
                        ),
                    )
                )

        return cls(entries)

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        accept_threshold: float = 0.72,
    ) -> list[MatchCandidate]:
        """检索 OCR 文本对应的馆藏候选。

        Args:
            query_text: OCR 拼接文本或人工输入文本。
            top_k: 返回候选数量。
            accept_threshold: 高于该分数视为自动匹配，否则标记为待确认。
        """

        raw_query = query_text
        query = normalize_for_match(query_text)
        if not query:
            return []

        candidate_ids = self._retrieve_candidate_ids(raw_query)
        scored: list[MatchCandidate] = []
        for entry_id in candidate_ids:
            entry = self.entries[entry_id]
            # 评分必须使用 OCR 原文，不能只用清洗后的 query。
            # 因为 normalize_for_match() 会删除 F272/202 这类索书号噪声；
            # 但在馆藏匹配阶段，索书号反而是非常有价值的辅助证据。
            score = score_catalog_candidate(raw_query, entry)
            status = "matched" if score >= accept_threshold else "pending"
            scored.append(MatchCandidate(entry=entry, score=score, status=status))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def best_match(
        self,
        query_text: str,
        *,
        accept_threshold: float = 0.72,
    ) -> MatchCandidate | None:
        """返回最高分候选；无候选时返回 None。"""

        candidates = self.search(query_text, top_k=1, accept_threshold=accept_threshold)
        return candidates[0] if candidates else None

    def _build_index(self) -> None:
        """为所有书名建立 n-gram 倒排索引。"""

        for entry_id, entry in enumerate(self.entries):
            for gram in make_grams(entry.search_text or entry.match_text, self.gram_size):
                self.inverted_index[gram].add(entry_id)
            _, item_digits = split_call_number_digits(entry.call_number)
            if len(item_digits) >= 3:
                self.call_item_index[item_digits].add(entry_id)

    def _retrieve_candidate_ids(self, query: str) -> list[int]:
        """根据查询文本召回候选书目编号。"""

        query_normalized = normalize_for_match(query)
        query_identifier = normalize_identifier(query)
        grams = make_grams(query_normalized, self.gram_size) | make_grams(
            query_identifier, self.gram_size
        )
        if not grams:
            return []

        exact_title_hits: list[int] = []
        exact_call_hits: list[int] = []
        for entry_id, entry in enumerate(self.entries):
            # 对《复盘》《战略》这类短主标题，n-gram 票数太少，容易在召回阶段被长标题淹没。
            # 只要规范书名完整出现在 OCR 文本中，就强制加入候选。
            if len(entry.match_text) >= 2 and entry.match_text in query_normalized:
                exact_title_hits.append(entry_id)

        # 对索书号种次号命中的候选强制召回，但不再全表逐条计算相似度。
        # 例如 OCR `P212179` 中的 `179` 可以召回馆藏 `F272/179`。
        for token in make_digit_tokens(query):
            exact_call_hits.extend(self.call_item_index.get(token, set()))

        counter: Counter[int] = Counter()
        for gram in grams:
            for entry_id in self.inverted_index.get(gram, set()):
                counter[entry_id] += 1

        # 如果 OCR 文本特别短，2-gram 可能召回太少，此时降级为全表中较短的子集不划算；
        # 直接返回已有候选即可，让结果进入人工确认。
        ranked_ids = [entry_id for entry_id, _ in counter.most_common(self.max_candidates)]
        return list(dict.fromkeys(exact_title_hits + exact_call_hits + ranked_ids))


def normalize_for_match(text: str) -> str:
    """面向书名匹配的文本规范化。

    这里会去除标点、空白和常见索书号片段，但保留中文、英文字母和数字。
    数字不能全部删除，因为很多书名本身含有版本号、年份或软件版本号。
    """

    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)

    # 粗略删除 OCR 中常见的索书号，如 F272、TP391.41/12、I247.5 等。
    # 这一步只针对“字母 + 多位数字 + 可选分类符号”的连续片段，避免误删 SharePoint2010。
    text = re.sub(r"(?<![a-z])[a-z]{1,4}\d{2,4}(?:[./:：-]?\d+)*(?![a-z])", "", text)

    # 删除明显不是书名主体的符号。
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text


def clean_catalog_cell(value: str | None) -> str:
    """清理馆藏 CSV 单元格，避免换行符污染导出的盘点表。"""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def make_grams(text: str, gram_size: int = 2) -> set[str]:
    """生成字符 n-gram 集合。"""

    text = normalize_for_match(text)
    if not text:
        return set()
    if len(text) <= gram_size:
        return {text}
    return {text[index : index + gram_size] for index in range(len(text) - gram_size + 1)}


def score_title_similarity(query: str, title: str) -> float:
    """计算 OCR 文本与馆藏书名的相似度。

    OCR 文本常常是“书名 + 作者 + 索书号”的混合串，因此只用完整串相似度会低估。
    这里综合三个信号：
    - 完整串相似度：适合 OCR 很干净的情况；
    - 最长公共片段覆盖率：适合书名被夹在噪声中的情况；
    - 字符集合 Jaccard：适合少量错字、漏字的情况。
    """

    query = normalize_for_match(query)
    title = normalize_for_match(title)
    if not query or not title:
        return 0.0

    full_ratio = SequenceMatcher(None, query, title).ratio()
    longest = SequenceMatcher(None, query, title).find_longest_match(
        0, len(query), 0, len(title)
    )
    coverage = longest.size / max(len(title), 1)

    query_chars = set(query)
    title_chars = set(title)
    jaccard = len(query_chars & title_chars) / max(len(query_chars | title_chars), 1)

    # 书名完整出现在 OCR 噪声串中时，coverage 会接近 1；
    # 若 OCR 比较干净，则 full_ratio 会更可靠。
    score = max(full_ratio, 0.75 * coverage + 0.25 * jaccard)

    # 短书名很容易被出版社、学校名、作者名中的普通词误命中。
    # 例如 OCR 噪声串里有“清华大学出版社”，不能因此把书名自动匹配成“大学”。
    #
    # 但书脊上也常见“短主标题 + 长副标题”的情况，例如：
    #   馆藏规范书名：复盘
    #   OCR 文本：复盘反思创新与商业模式孙北P272186
    # 这种情况下短书名位于 OCR 文本开头，符合书脊阅读顺序，应允许自动匹配。
    if len(title) < 4 and len(query) > len(title) * 2:
        if query.startswith(title):
            score = max(score, 0.78)
        else:
            score = min(score, 0.65)

    return round(score, 4)


def score_catalog_candidate(query: str, entry: CatalogEntry) -> float:
    """综合书名、作者和索书号证据计算候选匹配分数。

    只看书名会很脆弱：书脊 OCR 常把竖排标题读错，但作者、索书号、出版社片段可能是对的。
    因此这里采用“书名为主，作者/索书号兜底”的策略：
    - 书名读得好时，仍由书名相似度决定；
    - 书名读崩但索书号高度吻合时，可以提升到自动匹配；
    - 作者和索书号同时命中时，认为匹配更可靠；
    - 没有作者/索书号证据时，不会盲目提高短词命中分数。
    """

    title_score = score_title_similarity(query, entry.match_text)
    author_score = score_author_similarity(query, entry.author)
    call_score = score_call_number_similarity(query, entry.call_number)

    score = title_score

    # 索书号几乎是馆藏定位强证据。OCR 常把 F 读成 P，所以这里只比数字主体。
    if call_score >= 0.95 and (title_score >= 0.15 or author_score >= 0.55):
        score = max(score, 0.82)

    # 作者 + 索书号同时命中，基本可以锁定同一本书。
    if author_score >= 0.75 and call_score >= 0.80:
        score = max(score, 0.92)

    # 标题有一定相似度，再叠加作者或索书号，可以认为比较可靠。
    if title_score >= 0.45 and (author_score >= 0.75 or call_score >= 0.80):
        score = max(score, 0.82)

    # 索书号高度相似，且标题存在少量字符重合时，可以作为“书名 OCR 较差”的兜底。
    # 例：OCR `企算题美P212179` 可对应 `云计算揭秘` + `F272/179`。
    # 但如果标题完全不像，仍不自动接受，避免只凭索书号造成重复/误合并。
    if call_score >= 0.85 and title_score >= 0.20 and len(entry.match_text) >= 5:
        score = max(score, 0.78)

    # 作者明确命中，索书号尾号也命中时，虽然 OCR 可能漏掉分类号中间数字，也应认为较可靠。
    # 例：张少平 + P2I172，可以对应馆藏索书号 F272/172。
    if author_score >= 0.75 and call_score >= 0.70:
        score = max(score, 0.86)

    # 标题本身已经较接近时，作者轻微命中也能增强可信度。
    if title_score >= 0.60 and author_score >= 0.55:
        score = max(score, min(0.95, title_score + 0.08))

    # 四字及以上标题相似度超过 0.60，通常已经包含足够的连续书名证据；
    # 用于处理“战略咨询”被 OCR 为“成略咨询”这类一字错误。
    if len(entry.match_text) >= 4 and title_score >= 0.60:
        score = max(score, 0.74)

    return round(score, 4)


def score_author_similarity(query: str, author: str) -> float:
    """计算 OCR 文本与馆藏作者字段的重合程度。"""

    query = normalize_for_match(query)
    author = normalize_author(author)
    if not query or len(author) < 2:
        return 0.0

    longest = SequenceMatcher(None, query, author).find_longest_match(
        0, len(query), 0, len(author)
    )

    # 短作者名通常是人名，3 个连续字符命中可信度较高，例如“邓正红”。
    if len(author) <= 4 and longest.size >= 3:
        return 0.90
    if len(author) <= 4 and longest.size == 2:
        return 0.72

    # 中长作者名需要更长片段，避免“中国人”这类泛词误命中“中国人民...”。
    if len(author) <= 8 and longest.size >= 4:
        return 0.82
    if len(author) > 8 and longest.size >= 6:
        return 0.78

    return 0.0


def score_call_number_similarity(query: str, call_number: str) -> float:
    """计算 OCR 文本与索书号的相似度。

    书脊底部索书号很常见，但 OCR 容易把 `F272/202` 读成 `P272202` 或 `P272 202`。
    这里不能简单拼接全部数字做相似度，否则 `F272202` 会和 `O22/207=2`
    这种无关索书号产生虚假的高分。
    """

    query_identifier = normalize_identifier(query)
    call_identifier = normalize_identifier(call_number)
    query_digits = "".join(re.findall(r"\d+", query_identifier))
    call_digits = "".join(re.findall(r"\d+", call_identifier))
    if not query_digits or not call_digits:
        return 0.0

    # 只有一两个数字没有定位意义，例如 OCR 文本末尾单独出现“2”。
    if len(query_digits) < 4 or len(call_digits) < 4:
        return 0.0

    if call_identifier and call_identifier in query_identifier:
        return 0.98

    class_digits, item_digits = split_call_number_digits(call_number)
    if not item_digits:
        return 0.0

    item_exact = item_digits in query_digits
    item_ratio = best_substring_ratio(query_digits, item_digits)
    class_exact = len(class_digits) >= 3 and class_digits in query_digits
    class_ratio = best_substring_ratio(query_digits, class_digits) if len(class_digits) >= 3 else 0.0

    # 完整索书号非常接近，例如 F272/202 -> P272200。
    # 这里要求分类号主体也接近，避免 O22/207=2 误命中 F272202。
    full_ratio = SequenceMatcher(None, query_digits, call_digits).ratio()
    if len(call_digits) >= 5 and full_ratio >= 0.82 and (class_exact or class_ratio >= 0.66):
        return 0.86

    # 种次号精确命中，同时分类号主体也可靠，索书号证据较强。
    if item_exact and (class_exact or class_ratio >= 0.66):
        return 0.86

    # 分类号可靠，种次号只有轻微 OCR 错误，也作为较强证据。
    if (class_exact or class_ratio >= 0.75) and item_ratio >= 0.66:
        return 0.80

    # 只有种次号命中时，证据较弱，必须配合标题/作者使用。
    if item_exact:
        return 0.72

    return 0.0


def normalize_author(author: str) -> str:
    """清理作者字段，只保留对匹配有帮助的人名/机构文本。"""

    text = normalize_for_match(author)
    for noise in ["主编", "编著", "著", "编", "等", "美", "日", "英", "法", "德", "加"]:
        text = text.replace(noise, "")
    return text


def normalize_identifier(value: str) -> str:
    """保留索书号/编号中的字母和数字，用于召回和辅助匹配。"""

    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def split_call_number_digits(call_number: str) -> tuple[str, str]:
    """拆分馆藏索书号中的分类号数字和种次号数字。

    示例：
        `F272/202` -> (`272`, `202`)
        `O22/207=2` -> (`22`, `207`)
    """

    text = str(call_number)
    before_slash, _, after_slash = text.partition("/")
    class_digits = "".join(re.findall(r"\d+", before_slash))
    if after_slash:
        # `=2` 这类复本/版本后缀不参与主匹配，否则容易放大误匹配。
        item_part = re.split(r"[=:：\\-]", after_slash, maxsplit=1)[0]
    else:
        item_part = ""
    item_digits = "".join(re.findall(r"\d+", item_part))
    return class_digits, item_digits


def best_substring_ratio(text: str, pattern: str) -> float:
    """计算 pattern 与 text 任意等长片段的最高相似度。"""

    if not text or not pattern:
        return 0.0
    if pattern in text:
        return 1.0
    if len(text) <= len(pattern):
        return SequenceMatcher(None, text, pattern).ratio()

    best = 0.0
    window = len(pattern)
    for index in range(0, len(text) - window + 1):
        best = max(best, SequenceMatcher(None, text[index : index + window], pattern).ratio())
    return best


def make_digit_tokens(text: str) -> set[str]:
    """从 OCR 文本中提取可用于索书号召回的 3~5 位数字片段。"""

    tokens: set[str] = set()
    for group in re.findall(r"\d{3,}", str(text)):
        max_window = min(5, len(group))
        for window in range(3, max_window + 1):
            for index in range(0, len(group) - window + 1):
                tokens.add(group[index : index + window])
    return tokens


def _safe_int(value: str | None) -> int:
    """安全转换整数，馆藏册数字段为空时返回 0。"""

    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0
