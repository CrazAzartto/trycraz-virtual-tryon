"""Metric implementations checked against hand-computed values."""

import math

import pytest

from evaluation.rec_eval import kendall_tau, ndcg_at_k, precision_at_k


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c", "d"], {"a": 2, "c": 1}, 4) == 0.5
    assert precision_at_k(["a", "b"], {}, 2) == 0.0


def test_ndcg_perfect_and_hand_computed():
    rel = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)
    # order b, a, c: DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3) = 1 + 3/1.585 ; IDCG = 3 + 1/1.585
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a", "c"], rel, 3) == pytest.approx(expected)


def test_kendall_tau():
    assert kendall_tau(list("abcde"), list("abcde")) == 1.0
    assert kendall_tau(list("abcde"), list("edcba")) == -1.0
    assert kendall_tau(list("abc"), list("acb"), min_common=3) == pytest.approx(1 / 3)
    assert kendall_tau(list("abcde"), list("vwxyz")) is None      # nothing in common: undefined, not 1.0
