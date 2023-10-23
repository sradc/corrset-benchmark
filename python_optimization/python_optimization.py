#!/usr/bin/env python
# coding: utf-8

# Notes:
# - line_profiler is used for relative speeds of lines, but slows down the overall execution, that's why iteration speed is measured separately

# In[ ]:


import math
import io
from math import sqrt
import numba
import itertools
import pandas as pd
from pandas import IndexSlice as islice
import time
import numpy as np
import line_profiler

MAX_ITER = 1000


# In[ ]:


data = pd.read_json("data-large.json")


def profile_func(outer_func, func_to_profile) -> str:
    lp = line_profiler.LineProfiler()
    outer_func_with_profiler = lp(outer_func)
    lp.add_function(func_to_profile)
    outer_func_with_profiler()
    stream = io.StringIO()
    lp.print_stats(stream=stream)
    report = stream.getvalue()
    # Format report string
    s = ""
    # Get first table of report
    add = False
    for line in report.split("\n"):
        if f"Function: {func_to_profile.__name__}" in line:
            add = True
        if add:
            s += line + "\n"
        if add and "Total time" in line:
            break
    # Get only the % Time column onwards
    start_idx = s.find("% Time")
    for line in s.split("\n"):
        if "Line #" in line:
            start_idx = line.index("% Time")
            break
    s = "\n".join(x[start_idx:] for x in s.split("\n"))
    return s


# # Original

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    q_to_score = data.set_index(["question", "user"])
    grand_totals = data.groupby("user").score.sum().rename("grand_total")
    start = time.time()
    corrs = compute_corrs(qs_combinations, q_to_score, grand_totals, K)
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, q_to_score, grand_totals, K):
    result = []
    for qs in qs_combinations:
        qs_data = q_to_score.loc[islice[qs, :], :].swaplevel()
        answered_all = qs_data.groupby(level=[0]).size() == K
        answered_all = answered_all[answered_all].index
        qs_total = (
            qs_data.loc[islice[answered_all, :]]
            .groupby(level=[0])
            .sum()
            .rename(columns={"score": "qs"})
        )
        r = qs_total.join(grand_totals).corr().qs.grand_total
        result.append({"qs": qs, "r": r})
    return result


print(f"Computing times for baseline...")
corrs_baseline, avg_iter_time_secs_baseline = k_corrset(data, K=5, max_iter=MAX_ITER)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs_baseline} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs_baseline:0.1f}x"
)


# ## Optimisation 1

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = {q: set() for q in data.question.unique()}
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q].add(u)
    q_to_score = data.set_index(["question", "user"])
    grand_totals = data.groupby("user").score.sum().rename("grand_total")
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, q_to_score, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, q_to_score, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = [users_who_answered_q[q] for q in qs]
        answered_all = set.intersection(*user_sets_for_qs)
        qs_data = q_to_score.loc[islice[qs, :], :].swaplevel()
        qs_total = (
            qs_data.loc[islice[list(answered_all), :]]
            .groupby(level=[0])
            .sum()
            .rename(columns={"score": "qs"})
        )
        r = qs_total.join(grand_totals).corr().qs.grand_total
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 1 - using sets instead of groupby")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 2

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = {q: set() for q in data.question.unique()}
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q].add(u)
    score_dict = {
        (q, u): s
        for q, u, s in data[["question", "user", "score"]].itertuples(index=False)
    }
    grand_totals = data.groupby("user").score.sum().rename("grand_total")
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_dict, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_dict, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = [users_who_answered_q[q] for q in qs]
        answered_all = set.intersection(*user_sets_for_qs)
        qs_total = {u: sum(score_dict[q, u] for q in qs) for u in answered_all}
        qs_total = pd.DataFrame.from_dict(qs_total, orient="index", columns=["qs"])
        qs_total.index.name = "user"
        r = qs_total.join(grand_totals).corr().qs.grand_total
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 2 - using dicts for scores")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 3

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = {q: set() for q in data.question.unique()}
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q].add(u)
    score_dict = {
        (q, u): s
        for q, u, s in data[["question", "user", "score"]].itertuples(index=False)
    }
    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_dict, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_dict, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = [users_who_answered_q[q] for q in qs]
        answered_all = set.intersection(*user_sets_for_qs)
        qs_total = [sum(score_dict[q, u] for q in qs) for u in answered_all]
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = np.corrcoef(qs_total, user_grand_total)[0, 1]
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 3 - using dicts for grand totals")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 4

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = {q: set() for q in data.question.unique()}
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q].add(u)
    score_dict = {
        (q, u): s
        for q, u, s in data[["question", "user", "score"]].itertuples(index=False)
    }
    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_dict, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_dict, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = [users_who_answered_q[q] for q in qs]
        answered_all = set.intersection(*user_sets_for_qs)
        qs_total = [sum(score_dict[q, u] for q in qs) for u in answered_all]
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = np.corrcoef(qs_total, user_grand_total)[0, 1]
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 4 - ints instead of strings for user/questions")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 5

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True
    score_dict = {
        (q, u): s
        for q, u, s in data[["question", "user", "score"]].itertuples(index=False)
    }
    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_dict, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_dict, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = users_who_answered_q[qs, :]  # numpy indexing
        answered_all = np.logical_and.reduce(user_sets_for_qs)
        answered_all = np.where(answered_all)[0]
        qs_total = [sum(score_dict[q, u] for q in qs) for u in answered_all]
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = np.corrcoef(qs_total, user_grand_total)[0, 1]
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 5 - numpy boolean array for users who answered")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 6

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True

    score_matrix = np.zeros(
        (len(data.user.unique()), len(data.question.unique())), dtype=np.float64
    )
    for q, u, s in data[["question", "user", "score"]].itertuples(index=False):
        score_matrix[u, q] = s

    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = users_who_answered_q[qs, :]
        answered_all = np.logical_and.reduce(user_sets_for_qs)
        answered_all = np.where(answered_all)[0]
        qs_total = score_matrix[answered_all, :][:, qs].sum(axis=1)
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = np.corrcoef(qs_total, user_grand_total)[0, 1]
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 6 - score matrix instead of score dict")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 7

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True

    score_matrix = np.zeros(
        (len(data.user.unique()), len(data.question.unique())), dtype=np.float64
    )
    for q, u, s in data[["question", "user", "score"]].itertuples(index=False):
        score_matrix[u, q] = s

    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def corrcoef(a: list[float], b: list[float]) -> float | None:
    """same as np.corrcoef(a, b)[0, 1]"""
    n = len(a)
    sum_a = sum(a)
    sum_b = sum(b)
    sum_ab = sum(a_i * b_i for a_i, b_i in zip(a, b))
    sum_a_sq = sum(a_i**2 for a_i in a)
    sum_b_sq = sum(b_i**2 for b_i in b)
    num = n * sum_ab - sum_a * sum_b
    den = sqrt(n * sum_a_sq - sum_a**2) * sqrt(n * sum_b_sq - sum_b**2)
    if den == 0:
        return None
    return num / den


def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    result = []
    for qs in qs_combinations:
        user_sets_for_qs = users_who_answered_q[qs, :]  # numpy indexing
        answered_all = np.logical_and.reduce(user_sets_for_qs)
        answered_all = np.where(answered_all)[0]
        qs_total = score_matrix[answered_all, :][:, qs].sum(axis=1)
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = corrcoef(qs_total, user_grand_total)
        result.append({"qs": qs, "r": r})
    return result


print("Running optimization 7 - custom corrcoef")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 8

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    qs_combinations = np.array(list(qs_combinations))
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True

    score_matrix = np.zeros(
        (len(data.user.unique()), len(data.question.unique())), dtype=np.float64
    )
    for q, u, s in data[["question", "user", "score"]].itertuples(index=False):
        score_matrix[u, q] = s

    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    corrs = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame(corrs)
    return corrs, avg_iter_time_secs


def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    result = []
    for i in range(len(qs_combinations)):
        qs = qs_combinations[i]
        user_sets_for_qs = users_who_answered_q[qs, :]  # numpy indexing
        answered_all = np.logical_and.reduce(user_sets_for_qs)
        answered_all = np.where(answered_all)[0]
        qs_total = score_matrix[answered_all, :][:, qs].sum(axis=1)
        user_grand_total = [grand_totals[u] for u in answered_all]
        r = corrcoef(qs_total, user_grand_total)
        result.append({"qs": qs, "r": r})
    return result


print("Running change 8 - qs as array")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 9

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    qs_combinations = np.array(list(qs_combinations))
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True

    score_matrix = np.zeros(
        (len(data.user.unique()), len(data.question.unique())), dtype=np.float64
    )
    for q, u, s in data[["question", "user", "score"]].itertuples(index=False):
        score_matrix[u, q] = s

    grand_totals = data.groupby("user").score.sum().rename("grand_total").to_dict()
    start = time.time()
    r_vals = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame({"qs": [tuple(qs) for qs in qs_combinations], "r": r_vals})
    return corrs, avg_iter_time_secs


def corrcoef(a: list[float], b: list[float]) -> float | None:
    """same as np.corrcoef(a, b)[0, 1]"""
    n = len(a)
    sum_a = sum(a)
    sum_b = sum(b)
    sum_ab = sum(a_i * b_i for a_i, b_i in zip(a, b))
    sum_a_sq = sum(a_i**2 for a_i in a)
    sum_b_sq = sum(b_i**2 for b_i in b)
    num = n * sum_ab - sum_a * sum_b
    den = sqrt(n * sum_a_sq - sum_a**2) * sqrt(n * sum_b_sq - sum_b**2)
    if den == 0:
        return None
    return num / den


def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    result = np.empty(len(qs_combinations), dtype=np.float64)
    for i in range(len(qs_combinations)):
        qs = qs_combinations[i]
        user_sets_for_qs = users_who_answered_q[qs, :]  # numpy indexing
        answered_all = np.logical_and.reduce(user_sets_for_qs)
        answered_all = np.where(answered_all)[0]
        qs_total = score_matrix[answered_all, :][:, qs].sum(axis=1)
        user_grand_total = [grand_totals[u] for u in answered_all]
        result[i] = corrcoef(qs_total, user_grand_total)
    return result


print("Running change 9 - r as array")
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
outer_func = lambda: k_corrset(data, K=5, max_iter=MAX_ITER)
s = profile_func(outer_func, compute_corrs)
print(s.strip())
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimisation 10

# In[ ]:


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data.user = data.user.map({u: i for i, u in enumerate(data.user.unique())})
    data.question = data.question.map(
        {q: i for i, q in enumerate(data.question.unique())}
    )
    qs_combinations = itertools.islice(
        itertools.combinations(data.question.unique(), K), max_iter
    )
    qs_combinations = np.array(list(qs_combinations))
    users_who_answered_q = np.zeros(
        (len(data.question.unique()), len(data.user.unique())), dtype=np.bool_
    )
    for q, u in data[["question", "user"]].itertuples(index=False):
        users_who_answered_q[q, u] = True

    score_matrix = np.zeros(
        (len(data.user.unique()), len(data.question.unique())), dtype=np.float64
    )
    for q, u, s in data[["question", "user", "score"]].itertuples(index=False):
        score_matrix[u, q] = s

    grand_totals = score_matrix.sum(axis=1)

    start = time.time()
    r_vals = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame({"qs": [tuple(qs) for qs in qs_combinations], "r": r_vals})
    return corrs, avg_iter_time_secs


@numba.njit
def corrcoef_numba(a, b):
    """same as np.corrcoef(a, b)[0, 1]"""
    n = len(a)
    sum_a = sum(a)
    sum_b = sum(b)
    sum_ab = sum(a * b)
    sum_a_sq = sum(a * a)
    sum_b_sq = sum(b * b)
    num = n * sum_ab - sum_a * sum_b
    den = math.sqrt(n * sum_a_sq - sum_a**2) * math.sqrt(n * sum_b_sq - sum_b**2)
    return np.nan if den == 0 else num / den


@numba.njit(parallel=True)
def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    result = np.empty(len(qs_combinations), dtype=np.float64)
    for i in numba.prange(len(qs_combinations)):
        qs = qs_combinations[i]
        user_sets_for_qs = users_who_answered_q[qs, :]
        answered_all = user_sets_for_qs[
            0
        ]  # numba doesn't support np.logical_and.reduce
        for j in range(1, len(user_sets_for_qs)):
            answered_all = np.logical_and(answered_all, user_sets_for_qs[j])
        answered_all = np.where(answered_all)[0]
        qs_total = score_matrix[answered_all, :][:, qs].sum(axis=1)
        user_grand_total = grand_totals[answered_all]
        result[i] = corrcoef_numba(qs_total, user_grand_total)
    return result


print("Running optimization 10 - numba, parrallel=True")
k_corrset(data, K=1, max_iter=2)  # let jit happen
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimization 11

# In[ ]:


@numba.njit(boundscheck=False, fastmath=True, nogil=True)
def bitset_create(size):
    size_in_int64 = int(np.ceil(size / 64))
    return np.zeros(size_in_int64, dtype=np.int64)


@numba.njit(boundscheck=False, fastmath=True, nogil=True)
def bitset_add(arr, pos):
    int64_idx = pos // 64
    pos_in_int64 = pos % 64
    arr[int64_idx] |= np.int64(1) << np.int64(pos_in_int64)


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data["user"] = data["user"].map({u: i for i, u in enumerate(data.user.unique())})
    data["question"] = data["question"].map(
        {q: i for i, q in enumerate(data.question.unique())}
    )

    all_qs = data.question.unique()
    grand_totals = data.groupby("user").score.sum().values

    # create bitsets
    users_who_answered_q = np.array(
        [bitset_create(data.user.nunique()) for _ in range(data.question.nunique())]
    )
    for q, u in data[["question", "user"]].values:
        bitset_add(users_who_answered_q[q], u)

    score_matrix = np.zeros(
        (data.user.nunique(), data.question.nunique()), dtype=np.int64
    )
    for row in data.itertuples():
        score_matrix[row.user, row.question] = row.score

    # todo, would be nice to have a super fast iterator / generator in numba
    # rather than creating the whole array
    qs_combinations = []
    for i, qs in enumerate(itertools.combinations(all_qs, K)):
        if i == max_iter:
            break
        qs_combinations.append(qs)
    qs_combinations = np.array(qs_combinations)

    start = time.time()
    r_vals = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame({"qs": [tuple(qs) for qs in qs_combinations], "r": r_vals})
    return corrs, avg_iter_time_secs


@numba.njit(boundscheck=False, fastmath=True, parallel=False, nogil=True)
def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    num_qs = qs_combinations.shape[0]
    bitset_size = users_who_answered_q[0].shape[0]
    corrs = np.empty(qs_combinations.shape[0], dtype=np.float64)
    for i in numba.prange(num_qs):
        # bitset will contain users who answered all questions in qs_array[i]
        bitset = users_who_answered_q[qs_combinations[i, 0]].copy()
        for q in qs_combinations[i, 1:]:
            bitset &= users_who_answered_q[q]
        # retrieve stats for the users to compute correlation
        n = 0.0
        sum_a = 0.0
        sum_b = 0.0
        sum_ab = 0.0
        sum_a_sq = 0.0
        sum_b_sq = 0.0
        for idx in range(bitset_size):
            if bitset[idx] != 0:
                for pos in range(64):
                    if (bitset[idx] & (np.int64(1) << np.int64(pos))) != 0:
                        user_idx = idx * 64 + pos
                        score_for_qs = 0.0
                        for q in qs_combinations[i]:
                            score_for_qs += score_matrix[user_idx, q]
                        score_for_user = grand_totals[user_idx]
                        n += 1.0
                        sum_a += score_for_qs
                        sum_b += score_for_user
                        sum_ab += score_for_qs * score_for_user
                        sum_a_sq += score_for_qs * score_for_qs
                        sum_b_sq += score_for_user * score_for_user
        num = n * sum_ab - sum_a * sum_b
        den = np.sqrt(n * sum_a_sq - sum_a**2) * np.sqrt(n * sum_b_sq - sum_b**2)
        corrs[i] = np.nan if den == 0 else num / den
    return corrs


print("Running optimization 11 - numba inline no parallel")
k_corrset(data, K=1, max_iter=2)  # let jit happen
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)

# Run for longer because so fast
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=1_000_000)
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)


# ## Optimization 12

# In[ ]:


@numba.njit(boundscheck=False, fastmath=True, nogil=True)
def bitset_create(size):
    size_in_int64 = int(np.ceil(size / 64))
    return np.zeros(size_in_int64, dtype=np.int64)


@numba.njit(boundscheck=False, fastmath=True, nogil=True)
def bitset_add(arr, pos):
    int64_idx = pos // 64
    pos_in_int64 = pos % 64
    arr[int64_idx] |= np.int64(1) << np.int64(pos_in_int64)


def k_corrset(data, K, max_iter=1000):
    data = data.copy()
    data["user"] = data["user"].map({u: i for i, u in enumerate(data.user.unique())})
    data["question"] = data["question"].map(
        {q: i for i, q in enumerate(data.question.unique())}
    )

    all_qs = data.question.unique()
    grand_totals = data.groupby("user").score.sum().values

    # create bitsets
    users_who_answered_q = np.array(
        [bitset_create(data.user.nunique()) for _ in range(data.question.nunique())]
    )
    for q, u in data[["question", "user"]].values:
        bitset_add(users_who_answered_q[q], u)

    score_matrix = np.zeros(
        (data.user.nunique(), data.question.nunique()), dtype=np.int64
    )
    for row in data.itertuples():
        score_matrix[row.user, row.question] = row.score

    # todo, would be nice to have a super fast iterator / generator in numba
    # rather than creating the whole array
    qs_combinations = []
    for i, qs in enumerate(itertools.combinations(all_qs, K)):
        if i == max_iter:
            break
        qs_combinations.append(qs)
    qs_combinations = np.array(qs_combinations)

    start = time.time()
    r_vals = compute_corrs(
        qs_combinations, users_who_answered_q, score_matrix, grand_totals
    )
    avg_iter_time_secs = (time.time() - start) / max_iter
    corrs = pd.DataFrame({"qs": [tuple(qs) for qs in qs_combinations], "r": r_vals})
    return corrs, avg_iter_time_secs


@numba.njit(boundscheck=False, fastmath=True, parallel=True, nogil=True)
def compute_corrs(qs_combinations, users_who_answered_q, score_matrix, grand_totals):
    num_qs = qs_combinations.shape[0]
    bitset_size = users_who_answered_q[0].shape[0]
    corrs = np.empty(qs_combinations.shape[0], dtype=np.float64)
    for i in numba.prange(num_qs):
        # bitset will contain users who answered all questions in qs_array[i]
        bitset = users_who_answered_q[qs_combinations[i, 0]].copy()
        for q in qs_combinations[i, 1:]:
            bitset &= users_who_answered_q[q]
        # retrieve stats for the users and compute corrcoef
        n = 0.0
        sum_a = 0.0
        sum_b = 0.0
        sum_ab = 0.0
        sum_a_sq = 0.0
        sum_b_sq = 0.0
        for idx in range(bitset_size):
            if bitset[idx] != 0:
                for pos in range(64):
                    if (bitset[idx] & (np.int64(1) << np.int64(pos))) != 0:
                        score_for_qs = 0.0
                        for q in qs_combinations[i]:
                            score_for_qs += score_matrix[idx * 64 + pos, q]
                        score_for_user = grand_totals[idx * 64 + pos]
                        n += 1.0
                        sum_a += score_for_qs
                        sum_b += score_for_user
                        sum_ab += score_for_qs * score_for_user
                        sum_a_sq += score_for_qs * score_for_qs
                        sum_b_sq += score_for_user * score_for_user
        num = n * sum_ab - sum_a * sum_b
        den = np.sqrt(n * sum_a_sq - sum_a**2) * np.sqrt(n * sum_b_sq - sum_b**2)
        corrs[i] = np.nan if den == 0 else num / den
    return corrs


print("Running optimization 12 - numba inline parallel")
k_corrset(data, K=1, max_iter=2)  # let jit happen
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=MAX_ITER)
assert np.allclose(corrs_baseline.r, corrs.r, equal_nan=True)

# Run for longer because so fast
corrs, avg_iter_time_secs = k_corrset(data, K=5, max_iter=5_000_000)
print(f"\nAverage time per iteration:   {avg_iter_time_secs} seconds")
print(
    f"Speedup over baseline:        {avg_iter_time_secs_baseline/avg_iter_time_secs:0.1f}x"
)

