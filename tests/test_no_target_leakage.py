"""Guards against target leakage into either consumer.

The negative-review task (code: t3, paper: T2) is labelled from review_score,
so review_score must never reach the tree, the agent prompt, or the profile.
"""
from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS, T3_FEATURE_COLS
from consumers.prompts.templates import build_prompt

LABEL_SOURCES = {"review_score", "y_t1", "y_t2", "y_t3"}
# Post-delivery columns would leak the late-delivery label (code: t2, paper: T1)
POST_DELIVERY = set(T1_EXTRA_FEATURE_COLS)


def test_review_task_features_exclude_label_sources():
    assert not LABEL_SOURCES & set(T3_FEATURE_COLS)


def test_late_delivery_features_are_purchase_time_only():
    assert not (LABEL_SOURCES | POST_DELIVERY) & set(T2_FEATURE_COLS)


def test_prompt_never_mentions_review_score():
    row = {c: None for c in T3_FEATURE_COLS}
    row["review_score"] = 1  # even if a caller passes it, it must not be rendered
    msgs = build_prompt("t3", row, list(T3_FEATURE_COLS), "profile")
    assert "review_score" not in msgs[1]["content"]
