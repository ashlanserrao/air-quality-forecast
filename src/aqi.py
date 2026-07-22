# src/aqi.py — CPCB PM2.5 sub-index / AQI category conversion
#
# The LSTM forecasts PM2.5 *concentration* in µg/m³. CPCB categories
# (Good/Satisfactory/...) are defined on the AQI *index*, which is a piecewise
# linear rescaling of concentration — the two are not interchangeable, and
# reading a µg/m³ value against index ranges silently understates severity.
# This module does that conversion in code so the agent never has to.

# (conc_low, conc_high, index_low, index_high, category)
# CPCB 24-hour PM2.5 breakpoints (National Air Quality Index, CPCB 2014).
PM25_BREAKPOINTS = [
    (0.0, 30.0, 0, 50, "Good"),
    (30.0, 60.0, 51, 100, "Satisfactory"),
    (60.0, 90.0, 101, 200, "Moderate"),
    (90.0, 120.0, 201, 300, "Poor"),
    (120.0, 250.0, 301, 400, "Very Poor"),
    # CPCB publishes the top band open-ended ("250+"). Its own calculator
    # extends it linearly to 380 µg/m³ = index 500; we follow that and clamp
    # above, so anything past 380 reports 500 / Severe rather than overflowing.
    (250.0, 380.0, 401, 500, "Severe"),
]

MAX_INDEX = 500
MAX_CATEGORY = "Severe"


def pm25_to_aqi(concentration: float) -> tuple[int, str]:
    """Convert a PM2.5 concentration (µg/m³) to a CPCB sub-index and category.

    Returns (index, category). Note this is the PM2.5 *sub-index*: the reported
    AQI for a location is the maximum sub-index across all pollutants, so the
    true AQI can be higher if another pollutant dominates — never lower.
    """
    if concentration < 0:
        raise ValueError(f"PM2.5 concentration cannot be negative: {concentration}")

    for conc_low, conc_high, index_low, index_high, category in PM25_BREAKPOINTS:
        if conc_low <= concentration <= conc_high:
            index = (index_high - index_low) / (conc_high - conc_low) * (
                concentration - conc_low
            ) + index_low
            return round(index), category

    return MAX_INDEX, MAX_CATEGORY
