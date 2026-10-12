"""Benefit-unit membership helpers shared by the dataset build and targets."""


def claimant_or_partner_variable(variables) -> str:
    """Name of the policyengine-uk variable marking a benefit unit's claimant
    and any partner.

    Releases that define ``is_claimant_or_partner`` use it for Housing
    Benefit's pension-age route. Earlier releases, such as 2.102.5, use
    ``is_adult`` (age 18 or over) there, which also counts an 18 or 19 year
    old dependant.
    """
    if "is_claimant_or_partner" in variables:
        return "is_claimant_or_partner"
    return "is_adult"
