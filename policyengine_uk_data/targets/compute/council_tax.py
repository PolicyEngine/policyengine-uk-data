"""Council tax compute functions."""

import numpy as np


def compute_council_tax_band(target, ctx) -> np.ndarray:
    """Compute council tax band count for a region."""
    parts = target.name.split("/")
    region = parts[2]
    band = parts[3]

    in_region = ctx.sim.calculate("region").values == region

    if band == "total":
        return in_region.astype(float)

    in_band = ctx.sim.calculate("council_tax_band") == band
    return (in_band * in_region).astype(float)


def compute_obr_council_tax(target, ctx) -> np.ndarray:
    """Compute OBR council tax receipts, optionally by country."""
    name = target.name
    ct = ctx.pe("council_tax")

    if name == "obr/council_tax":
        return ct
    if name == "obr/council_tax_england":
        return ct * (ctx.country == "ENGLAND")
    if name == "obr/council_tax_scotland":
        return ct * (ctx.country == "SCOTLAND")
    if name == "obr/council_tax_wales":
        return ct * (ctx.country == "WALES")
    return ct
