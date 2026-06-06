"""Schema description for prompts.

Plain-English column metadata used by the agent prompt.
Sourced from Olist's public documentation; manually curated for clarity.
"""
from __future__ import annotations

SCHEMA: dict[str, str] = {
    # time
    "purchase_hour": "Hour-of-day (0-23) the customer placed the order.",
    "purchase_dow": "Day-of-week (0=Mon..6=Sun) of the purchase.",
    "purchase_month": "Calendar month (1-12) of the purchase.",
    "days_to_estimated": "Days from purchase timestamp to the carrier-estimated delivery date.",
    # items / products
    "n_items": "Number of items in the order.",
    "total_price": "Sum of item prices in BRL.",
    "total_freight": "Sum of freight (shipping) charges in BRL.",
    "freight_ratio": "total_freight / total_price (cost-of-shipping ratio).",
    "max_price": "Maximum item price in BRL.",
    "mean_price": "Average item price in BRL.",
    "n_unique_products": "Distinct products in the order.",
    "n_unique_sellers": "Distinct sellers fulfilling the order.",
    "mean_product_weight_g": "Average product weight in grams.",
    "mean_product_volume_cm3": "Average product bounding-box volume in cubic cm.",
    "dominant_category": "Most common product category in the order (Portuguese).",
    # payments
    "n_payments": "Number of payment records (multi-payment orders are common).",
    "total_payment_value": "Sum of payment amounts in BRL.",
    "max_installments": "Largest installment count among payments (1 = pay-in-full).",
    "payment_type_modal": "Most-used payment method: credit_card, boleto, voucher, debit_card, etc.",
    # geography
    "customer_state": "Brazilian state of the customer (2-letter, e.g. SP, RJ, MG).",
    "n_seller_states": "Distinct seller states fulfilling the order.",
    # T1-only (post-delivery, leak-safe ONLY for T1)
    "days_to_approved": "Days from purchase to approval (T1 only; leaks for T2).",
    "days_carrier_to_customer": "Days from carrier handoff to customer delivery (T1 only; leaks for T2).",
    "days_purchase_to_delivery": "Days from purchase to final delivery (T1 only; leaks for T2).",
    "delivery_vs_estimate_days": "Delivery minus estimate, in days; positive = late (T1 only; leaks for T2).",
    "order_status": "Order lifecycle: delivered, shipped, canceled, etc. (T1 only).",
    # T3 text fields (Portuguese)
    "review_comment_title": "Free-form review title written by the customer (Portuguese; often empty).",
    "review_comment_message": "Free-form review body written by the customer (Portuguese; often empty).",
}

TASKS: dict[str, dict] = {
    "t1": {
        "name": "predict_review_high",
        "description": (
            "Predict whether the customer left a HIGH review (score 4 or 5) "
            "vs a LOW review (score 1 or 2) for this order."
        ),
        "label_meaning": "1 = high review (4-5 stars), 0 = low review (1-2 stars).",
    },
    "t2": {
        "name": "predict_late_delivery",
        "description": (
            "Predict whether this order will be delivered LATE — i.e., after "
            "the carrier-estimated delivery date. Only purchase-time features "
            "are available; do not use post-delivery columns."
        ),
        "label_meaning": "1 = late delivery, 0 = on-time or early.",
    },
    "t3": {
        "name": "predict_review_low",
        "description": (
            "Read the customer's Portuguese review text (title and message) "
            "together with order metadata, and predict whether this is a LOW "
            "review (1 or 2 stars) versus a HIGH review (4 or 5 stars). "
            "Reviews with a score of 3 are excluded from this task. The text "
            "fields may be empty; in that case rely on the tabular features."
        ),
        "label_meaning": "1 = low review (1-2 stars), 0 = high review (4-5 stars).",
    },
}
