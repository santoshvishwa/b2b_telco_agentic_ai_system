"""Shared constants for the synthetic data generator. Everything here is invented."""

from __future__ import annotations

from datetime import date

# --- Timeline ----------------------------------------------------------------
# The predictive model is frozen at MODEL_SNAPSHOT. Events after FRESH_START are
# invisible to it, which is how scores go stale and "paradox" cases arise.
AS_OF = date(2026, 10, 1)            # "today" for every agent: the scoring date
MODEL_SNAPSHOT = date(2026, 8, 31)   # last day of data the model saw
FRESH_START = date(2026, 9, 1)       # first day of "fresh" (model-invisible) events
FRESH_DAYS = 30                      # AS_OF - FRESH_START
PRE90_START = date(2026, 6, 3)       # 90-day window ending at MODEL_SNAPSHOT
HISTORY_START = date(2025, 10, 1)    # 12 months of ticket / call / offer history
SCORED_MONTH = "2026-10"
BILL_MONTHS = ("2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09")
MODEL_VERSION = "simulated-v0"       # replaced by a real model in Phase 2

# --- Scoring -----------------------------------------------------------------
FLAG_RATE = 0.0625   # top 6.25% per signal => 2 x 1,250 = 2,500 flagged at 20K accounts
CORE_RATE = 0.04     # planted eval accounts come from the top 4% so a Phase 2 model still flags them
LATE_DAYS = 5        # a bill paid (or overdue) by more than this many days is "late"

# --- Population --------------------------------------------------------------
SEGMENTS = ("Micro", "Small", "Medium")
SEGMENT_WEIGHTS = (0.45, 0.40, 0.15)
SEGMENT_EMPLOYEES = {"Micro": (1, 9), "Small": (10, 49), "Medium": (50, 249)}

# (name, speed_mbps, list price AED per month) - invented prices
PLANS = (
    ("Fiber Business 100", 100, 450),
    ("Fiber Business 300", 300, 700),
    ("Fiber Business 500", 500, 1000),
    ("Fiber Business 1G", 1000, 1500),
    ("Symmetric Premium 1G", 1000, 2400),
    ("Symmetric Premium 2G", 2000, 3800),
)
PLAN_WEIGHTS_BY_SEGMENT = {
    "Micro": (0.50, 0.30, 0.15, 0.04, 0.01, 0.00),
    "Small": (0.15, 0.30, 0.30, 0.15, 0.08, 0.02),
    "Medium": (0.02, 0.10, 0.25, 0.30, 0.23, 0.10),
}

# (name, weight, churn-risk offset)
INDUSTRIES = (
    ("Retail", 0.18, 0.15),
    ("Construction", 0.12, 0.20),
    ("Logistics", 0.10, 0.00),
    ("Healthcare Clinics", 0.08, -0.15),
    ("Hospitality", 0.12, 0.25),
    ("Professional Services", 0.14, -0.10),
    ("Education", 0.06, -0.20),
    ("Manufacturing", 0.07, 0.00),
    ("Real Estate", 0.07, 0.10),
    ("IT Services", 0.06, -0.05),
)

EMIRATES = (
    ("Dubai", 0.38),
    ("Abu Dhabi", 0.24),
    ("Sharjah", 0.18),
    ("Ajman", 0.06),
    ("Ras Al Khaimah", 0.06),
    ("Fujairah", 0.04),
    ("Umm Al Quwain", 0.04),
)

NAME_ADJ = (
    "Golden", "Blue", "Falcon", "Desert", "Pearl", "Crescent", "Horizon", "Oasis",
    "Summit", "Meridian", "Nova", "Prime", "Alpha", "Zenith", "Coral", "Amber",
)
NAME_NOUN = (
    "Trading", "Logistics", "Clinic", "Consulting", "Interiors", "Traders", "Contracting",
    "Technologies", "Foods", "Motors", "Solutions", "Studio", "Supplies", "Services",
)
NAME_SUFFIX = ("LLC", "FZE", "FZCO", "Co.")

PAYMENT_METHODS = ("Auto-debit", "Invoice", "Cheque")
PAYMENT_WEIGHTS = (0.45, 0.45, 0.10)

# Add-on products: name -> (monthly price AED, weight in random draws)
ADDONS = {
    "Static IP": (90, 0.20),
    "Managed Wi-Fi": (200, 0.18),
    "Cloud Backup": (150, 0.14),
    "Cybersecurity Bundle": (220, 0.14),
    "VoIP Lines": (280, 0.12),
    "SD-WAN Lite": (900, 0.04),
    "Extra Bandwidth": (350, 0.10),
    "4G Backup": (240, 0.08),
}

# --- Tickets -----------------------------------------------------------------
TICKET_CATEGORIES = (
    "Outage", "Slow speed", "Billing dispute", "Installation delay", "SLA breach",
    "Service request", "Upgrade request",
)
TICKET_CATEGORY_P = (0.20, 0.20, 0.12, 0.08, 0.05, 0.25, 0.10)
COMPLAINT_CATEGORIES = TICKET_CATEGORIES[:5]
REQUEST_CATEGORIES = TICKET_CATEGORIES[5:]
SEVERITIES = ("Low", "Medium", "High")

# --- Calls and external signals (no PII anywhere) ------------------------------
CALL_SUMMARIES = {
    ("complaint", "neg"): [
        "Customer reported repeated outages and asked about compensation.",
        "Customer unhappy about slow speeds at peak hours and asked about contract exit terms.",
        "Customer disputed the last bill and said they are comparing other providers.",
        "Customer escalated an unresolved fault and mentioned cancelling the service.",
    ],
    ("complaint", "neu"): [
        "Customer called to follow up on an open ticket; no new issue raised.",
        "Customer asked for a status update on a delayed installation.",
    ],
    ("complaint", "pos"): [
        "Customer called to thank the team for resolving a fault quickly.",
    ],
    ("enquiry", "pos"): [
        "Customer thanked support and asked about adding more bandwidth.",
        "Customer happy with a smooth upgrade and asked about the cybersecurity bundle.",
        "Customer satisfied with service and asked about renewal options.",
    ],
    ("enquiry", "neu"): [
        "Customer asked for an invoice copy and payment options.",
        "Customer asked about static IP availability.",
        "Customer asked how to change the authorised contact on the account.",
    ],
    ("enquiry", "neg"): [
        "Customer asked about early termination charges.",
    ],
    ("acquisition", "pos"): [
        "Customer is opening a second office and asked about extra lines and bandwidth.",
        "Customer plans to add staff and asked about a higher speed plan.",
        "Customer asked for a quote on managed Wi-Fi and cloud backup for a new branch.",
    ],
    ("acquisition", "neu"): [
        "Customer asked in passing about higher speed plans; no timeline given.",
        "Customer asked the price of extra bandwidth and said they would think about it.",
    ],
    ("acquisition", "neg"): [
        "Customer asked about extra bandwidth but raised concerns about recent reliability.",
    ],
}

# kind -> list of (source, text); {emirate} is filled in per account
EXTERNAL_SIGNALS = {
    "growth": [
        ("LinkedIn", "Company announced a new branch in {emirate}"),
        ("LinkedIn", "Hiring surge: several new roles posted this month"),
        ("X/Twitter", "Company announced the opening of a second office"),
        ("Financial services", "Trade licence expanded with additional business activities"),
    ],
    "distress": [
        ("Financial services", "Credit bureau flag: delayed supplier payments"),
        ("Financial services", "Trade licence renewal overdue"),
        ("X/Twitter", "Company announced downsizing and a branch closure"),
        ("LinkedIn", "Multiple staff departures posted this quarter"),
    ],
    "competitor": [
        ("X/Twitter", "Competitor promotion targeting SMB fiber customers in {emirate}"),
        ("Utility service", "Utility-service complaints in this area up sharply this month"),
        ("LinkedIn", "Company posted about evaluating new connectivity providers"),
    ],
    "positive_brand": [
        ("X/Twitter", "Public post praising network reliability"),
        ("LinkedIn", "Company shared a case study featuring our connectivity"),
    ],
    "neutral": [
        ("Utility service", "Planned maintenance notice in the area"),
        ("Financial services", "Routine trade licence renewal completed"),
        ("LinkedIn", "Company posted a routine news update"),
    ],
}
EXTERNAL_KIND_P = (("neutral", 0.55), ("growth", 0.12), ("positive_brand", 0.10),
                   ("competitor", 0.13), ("distress", 0.10))
EXTERNAL_COVERAGE = 0.90
