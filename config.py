"""Simulation parameters for the Daraz Pakistan ad-spend simulator.

All money is in Pakistani Rupees (PKR). The numbers are *assumptions* chosen to be
plausible for Pakistani digital advertising in 2025-26; they are not Daraz's real
figures. Because the data is simulated, these "true" values are known, so we can
later check whether the recommendation engine recovers them.
"""
import datetime as dt

SEED = 42
START = dt.date(2025, 1, 1)
END = dt.date(2026, 10, 31)          # 22 months of history
MONTHLY_BUDGET = 1_000_000           # historical spend is ~Rs 1M / month on average

# ---------------------------------------------------------------------------
# Platforms
#   share : share of the historical budget (how Daraz *used* to split money)
#   cpm   : cost per 1,000 impressions (PKR)
#   ctr   : click-through rate at the reference spend level
#   cvr   : click -> purchase conversion rate at the reference spend level
#   aov   : average order value (PKR)
#   b     : spend elasticity of revenue (diminishing returns). Revenue ~ spend^b.
#           b = 1 means no saturation; lower b = the platform saturates faster.
# ---------------------------------------------------------------------------
PLATFORMS = {
    "Facebook":  dict(share=0.30, cpm=160, ctr=0.0110, cvr=0.0220, aov=2300, b=0.75),
    "Instagram": dict(share=0.20, cpm=230, ctr=0.0095, cvr=0.0300, aov=2900, b=0.75),
    "Google":    dict(share=0.20, cpm=420, ctr=0.0250, cvr=0.0320, aov=3300, b=0.50),
    "TikTok":    dict(share=0.15, cpm=85,  ctr=0.0090, cvr=0.0125, aov=1800, b=0.85),
    "YouTube":   dict(share=0.15, cpm=190, ctr=0.0060, cvr=0.0160, aov=2700, b=0.90),
}

# ---------------------------------------------------------------------------
# Campaigns -- taken from the real content types found on Daraz.pk's YouTube
# channel in Project 1.
#   mix      : default share of a platform's daily spend
#   ctr_m / cvr_m / aov_m : multipliers on the platform baseline
# Project 1 finding: the "Hansna Mana Hai" comedy series earned ~4.8x the comment
# engagement of sale videos -> modelled here as a strong CTR (attention) effect
# that converts less directly. Sale campaigns convert well but attract less
# attention, matching Project 1's "sale videos draw complaints, few likes".
# ---------------------------------------------------------------------------
CAMPAIGNS = {
    "Hansna Mana Hai (Comedy)": dict(mix=0.20, ctr_m=1.60, cvr_m=0.70, aov_m=1.00),
    "11.11 / 12.12 Mega Sale":  dict(mix=0.20, ctr_m=1.05, cvr_m=1.35, aov_m=0.85),
    "Ramadan & Eid":            dict(mix=0.10, ctr_m=1.10, cvr_m=1.00, aov_m=1.10),
    "Seller Stories / How-to":  dict(mix=0.15, ctr_m=0.80, cvr_m=0.85, aov_m=0.95),
    "Always-on Catalogue":      dict(mix=0.35, ctr_m=1.00, cvr_m=1.15, aov_m=1.00),
}
# Project 1 was measured on YouTube, so entertainment gets an extra boost there.
YOUTUBE_COMEDY_CTR_BOOST = 1.30

# ---------------------------------------------------------------------------
# Calendar: Pakistani e-commerce seasons
# ---------------------------------------------------------------------------
SALE_WINDOWS = [  # (start, end) inclusive
    (dt.date(2025, 3, 20), dt.date(2025, 3, 27)),    # Pakistan Day sale
    (dt.date(2025, 11, 1), dt.date(2025, 11, 12)),   # 11.11
    (dt.date(2025, 12, 1), dt.date(2025, 12, 13)),   # 12.12
    (dt.date(2026, 3, 20), dt.date(2026, 3, 27)),
]
RAMADAN_WINDOWS = [
    (dt.date(2025, 3, 1), dt.date(2025, 3, 31)),     # Ramadan + Eid-ul-Fitr 2025
    (dt.date(2026, 2, 18), dt.date(2026, 3, 21)),    # Ramadan + Eid-ul-Fitr 2026
]
SALE_SPEND_MULT, SALE_CVR_MULT = 1.8, 1.35
RAMADAN_SPEND_MULT, RAMADAN_CVR_MULT = 1.3, 1.15
DAILY_SPEND_NOISE = 0.35   # log-normal sigma on day-to-day budget changes

# ---------------------------------------------------------------------------
# Audience segments: (probability, ctr multiplier, cvr multiplier, aov multiplier)
# ---------------------------------------------------------------------------
AGES = {
    "18-24": (0.30, 1.15, 0.80, 0.85),
    "25-34": (0.35, 1.05, 1.15, 1.00),
    "35-44": (0.20, 0.95, 1.10, 1.15),
    "45-54": (0.10, 0.85, 0.95, 1.20),
    "55+":   (0.05, 0.80, 0.80, 1.10),
}
# Platform-specific age skews (re-weights AGES probabilities)
AGE_SKEW = {
    "TikTok":    [1.8, 1.0, 0.5, 0.3, 0.2],
    "Instagram": [1.3, 1.2, 0.7, 0.5, 0.3],
    "Facebook":  [0.8, 1.0, 1.2, 1.4, 1.5],
    "Google":    [0.8, 1.0, 1.2, 1.2, 1.2],
    "YouTube":   [1.2, 1.0, 0.9, 0.8, 0.8],
}
GENDERS = {
    "Male":   (0.62, 1.00, 0.95, 1.00),
    "Female": (0.38, 1.05, 1.10, 1.05),
}
GENDER_SKEW = {"Instagram": [0.8, 1.35]}   # Instagram audience is more female
CITIES = {
    "Karachi":    (0.22, 1.00, 1.10, 1.05),
    "Lahore":     (0.20, 1.00, 1.10, 1.05),
    "Islamabad":  (0.08, 1.05, 1.20, 1.25),
    "Rawalpindi": (0.08, 1.00, 1.05, 1.05),
    "Faisalabad": (0.08, 0.95, 0.95, 0.90),
    "Multan":     (0.06, 0.95, 0.90, 0.90),
    "Peshawar":   (0.07, 0.95, 0.90, 0.90),
    "Hyderabad":  (0.05, 0.95, 0.90, 0.85),
    "Quetta":     (0.04, 0.90, 0.80, 0.85),
    "Sialkot":    (0.06, 1.00, 1.00, 1.00),
    "Gujranwala": (0.06, 0.95, 0.95, 0.90),
}
DEVICES = {
    "Android": (0.78, 1.00, 0.95, 0.90),
    "iOS":     (0.12, 1.05, 1.15, 1.45),
    "Desktop": (0.10, 0.80, 1.30, 1.20),
}
DEVICE_SKEW = {"Google": [0.7, 1.2, 2.6]}  # search has much more desktop traffic
