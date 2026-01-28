# DWP Stat-Xplore API findings

## Connection successful

API key works correctly (hex string format).

Rate limit: 9,999 remaining out of 10,000 requests.

## Database IDs discovered

### Personal Independence Payment (PIP)
**Folder:** `str:folder:fpip`

**Key database:** `str:database:PIP_Monthly_new`
- Label: "PIP Cases with Entitlement from 2019"
- Latest data: October 2025 (202510)

**Measures:**
- Count: `str:count:PIP_Monthly_new:V_F_PIP_MONTHLY`
- Financial award: `str:measure:PIP_Monthly_new:V_F_PIP_MONTHLY:PIP_AWARD_AMOUNT`

**Fields:**
- Date: `str:field:PIP_Monthly_new:F_PIP_DATE:DATE2`
- Daily Living Award Status: `str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE`
  - Enhanced: `str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:1`
  - Standard: `str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:2`
  - Nil: `str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:3`
- Mobility Award Status: `str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:MOB_AWARD_TYPE`
- Geography: `str:field:PIP_Monthly_new:F_PIP_GEOG:COA_CODE` (likely)

**Sample data (Oct 2025):**
- Daily Living Enhanced: 2,035,617 claimants
- Daily Living Standard: 1,685,421 claimants
- Daily Living Nil: 161,273
- Unknown: 259

### Universal Credit
**Folder:** `str:folder:fuc`

**Key databases:**
- People: `str:database:UC_Monthly`
- Households: `str:database:UC_Households`

**Need to explore for:**
- Two-child limit metrics
- Scotland geography filter
- Households with child under 1

### Benefit Cap
**Folder:** `str:folder:fbc`

**Key databases:**
- HB Point in Time Caseload: `str:database:Benefit_Cap_Monthly_2011`
- HB Cumulative Caseload: `str:database:Benefit_Cap_Cumulative_2011`

**Need to explore for:**
- Capped households count
- Total reduction amount

## Query patterns

### Basic count query
```json
{
  "database": "str:database:PIP_Monthly_new",
  "measures": ["str:count:PIP_Monthly_new:V_F_PIP_MONTHLY"],
  "dimensions": [
    ["str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE"]
  ]
}
```

### With date filtering (using recodes)
```json
{
  "database": "str:database:PIP_Monthly_new",
  "measures": ["str:count:PIP_Monthly_new:V_F_PIP_MONTHLY"],
  "dimensions": [
    ["str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE"],
    ["str:field:PIP_Monthly_new:F_PIP_DATE:DATE2"]
  ],
  "recodes": {
    "str:field:PIP_Monthly_new:F_PIP_DATE:DATE2": {
      "map": [["str:value:PIP_Monthly_new:F_PIP_DATE:DATE2:C_PIP_DATE:202510"]],
      "total": false
    }
  }
}
```

## Response structure

```json
{
  "cubes": {
    "str:count:PIP_Monthly_new:V_F_PIP_MONTHLY": {
      "values": [[2035617.0, 1685421.0, 161273.0, 259.0]],
      "precision": 0
    }
  },
  "fields": [
    {
      "uri": "str:field:PIP_Monthly_new:F_PIP_DATE:DATE2",
      "label": "Month",
      "items": [...]
    },
    {
      "uri": "str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE",
      "label": "Daily Living Award Status",
      "items": [
        {"labels": ["Daily Living - Enhanced"], ...},
        {"labels": ["Daily Living - Standard"], ...}
      ]
    }
  ]
}
```

The `values` array is multidimensional, matching the order of dimensions and items.

## Important notes

1. **Date format:** Dates are like `202510` (YYYYMM), stored as values like `str:value:PIP_Monthly_new:F_PIP_DATE:DATE2:C_PIP_DATE:202510`

2. **Recodes:** Default query returns latest month only. To get historical data, add date dimension.

3. **Geography:** Scotland data note - Adult Disability Payment (ADP) launched in Scotland from March 2022, completed transfer by June 2025. Scotland PIP data may be minimal/incomplete from mid-2025 onwards.

4. **Performance:** Broad queries (all dates, all dimensions) may timeout. Filter to specific months or use recodes.

5. **API key format:** Use the hex string directly (not decoded JWT).

## Next steps

1. Implement PIP queries for Daily Living Enhanced/Standard rates
2. Explore UC_Households database for two-child limit and Scotland metrics
3. Explore Benefit_Cap databases for capped households and reductions
4. Implement time series extraction (iterate through months)
5. Handle geography filtering for Scotland-specific metrics
6. Add proper error handling and retries for timeouts
