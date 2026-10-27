# Business Report (offline, data-grounded)

**Question:** Allocate 10k units across A/B/C to maximize profit with 95% service

**Data:**
warehouse  profit_per_unit  capacity  mu_demand  sigma_demand  holding_cost
        A                5      6000       4200           700           0.5
        B                4      4000       3100           500           0.5
        C                3      4000       2700           400           0.5

**Allocation:** {'A': 5360.0, 'B': 2480.0, 'C': 2160.0}
**Objective:** 160808.0
**Service:** 0.89% (target 95%)
**p95 cost:** 22391
**Findings:** ['feasible=True service=0.009 mean_shortage=1217']
**Recs:** ['Allocate proportional to demand*profit if service <0.95']

*Note: Offline fallback used business_data.csv — not dummy.*