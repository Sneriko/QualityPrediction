from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class CerCalculator:
    @staticmethod
    def levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j

        for i in range(1, la + 1):
            ca = a[i - 1]
            for j in range(1, lb + 1):
                cb = b[j - 1]
                cost = 0 if ca == cb else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[la][lb]

    @staticmethod
    def page_cer_permutation_invariant_htr_only(
        gt_lines: List[str],
        pred_lines: List[str],
        lambda_ins: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        m, n = len(gt_lines), len(pred_lines)
        K = max(m, n)

        if m == 0 and n == 0:
            return 0.0, 0.0, 0.0, float("nan")

        gt_lens = [len(s) for s in gt_lines]
        pred_lens = [len(s) for s in pred_lines]
        total_gt_chars = sum(gt_lens) if gt_lens else 1

        BIG = 10_000_000
        cost = np.full((K, K), BIG, dtype=float)

        for i in range(m):
            for j in range(n):
                cost[i, j] = CerCalculator.levenshtein(gt_lines[i], pred_lines[j])

        for i in range(m):
            for j in range(n, K):
                cost[i, j] = gt_lens[i]

        for j in range(n):
            for i in range(m, K):
                cost[i, j] = lambda_ins * pred_lens[j]

        for i in range(m, K):
            for j in range(n, K):
                cost[i, j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)
        total_edit_cost = float(cost[row_ind, col_ind].sum())

        matched_gt = [False] * m
        matched_pred = [False] * n
        for r, c in zip(row_ind, col_ind):
            if r < m and c < n:
                matched_gt[r] = True
                matched_pred[c] = True

        missing_gt = matched_gt.count(False)
        halluc_pred = matched_pred.count(False)

        missing_gt_ratio = missing_gt / max(1, m)
        halluc_ratio = halluc_pred / max(1, n)
        perm_cer = total_edit_cost / total_gt_chars

        pairs = [(r, c) for r, c in zip(row_ind, col_ind) if r < m and c < n]
        if not pairs:
            htr_only = float("nan")
        else:
            htr_cost = 0.0
            htr_gt_chars = 0
            for r, c in pairs:
                htr_cost += float(cost[r, c])
                htr_gt_chars += gt_lens[r]
            htr_only = float("nan") if htr_gt_chars == 0 else htr_cost / float(htr_gt_chars)

        return perm_cer, missing_gt_ratio, halluc_ratio, htr_only

    @staticmethod
    def line_cer(gt: str, pred: str) -> float:
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return CerCalculator.levenshtein(gt, pred) / max(1, len(gt))

    @staticmethod
    def page_cer_linewise_average(gt_lines: List[str], pred_lines: List[str]) -> Tuple[float, float, float]:
        m, n = len(gt_lines), len(pred_lines)
        if m == 0 and n == 0:
            return 0.0, 0.0, 0.0

        line_cers: List[float] = []
        for i in range(m):
            if i < n:
                line_cers.append(CerCalculator.line_cer(gt_lines[i], pred_lines[i]))
            else:
                line_cers.append(1.0)

        avg = sum(line_cers) / max(1, len(line_cers))
        missing = max(0, m - n) / max(1, m)
        halluc = max(0, n - m) / max(1, n)
        return avg, missing, halluc

    @staticmethod
    def page_cer_linewise_average_geom(gt_page, pred_page) -> Tuple[float, float, float]:
        return CerCalculator.page_cer_linewise_average(
            gt_page.sorted_by_reading_order().texts(),
            pred_page.sorted_by_reading_order().texts(),
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def page_bow_metrics(gt_lines: List[str], pred_lines: List[str]) -> Tuple[float, float, float]:
        gt_tokens = CerCalculator._tokenize(" ".join(gt_lines))
        pred_tokens = CerCalculator._tokenize(" ".join(pred_lines))

        if not gt_tokens and not pred_tokens:
            return 1.0, 1.0, 1.0

        gt_counts = Counter(gt_tokens)
        pred_counts = Counter(pred_tokens)
        total_gt = sum(gt_counts.values())
        total_pred = sum(pred_counts.values())

        if total_gt == 0 and total_pred > 0:
            return 0.0, 1.0, 0.0
        if total_pred == 0 and total_gt > 0:
            return 0.0, 0.0, 0.0

        matched = 0
        for tok in set(gt_counts.keys()) | set(pred_counts.keys()):
            matched += min(gt_counts.get(tok, 0), pred_counts.get(tok, 0))

        recall = matched / max(1, total_gt)
        prec = matched / max(1, total_pred)
        f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
        return prec, recall, f1
