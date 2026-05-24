# Sprint 5 — Per-Query Delta Analysis

Engines compared (best configurations from Sprint 5B/C):

| Tag | Engine | nDCG@5 | MRR |
|---|---|---:|---:|
| `oa`      | OpenAI / 009 stock (production)      | 0.3752 | 0.6721 |
| `snow`    | Snow-base @ w=0.30 cb=1.5 (cite-only) | 0.3824 | 0.6727 |
| `f2`      | f2 @ w=0.40 cb=1.5 (cite-only)        | 0.3893 | 0.6713 |
| `d3-cite` | d3 @ w=0.40 cb=1.5 (cite-only)        | 0.3911 | 0.6726 |
| `d3-all`  | d3 @ w=0.40 cb=2.0 (allpath)          | 0.3975 | 0.6655 |
| `d3-stk`  | d3 @ w=0.60 cb=1.3 (stock 009 knobs)  | 0.3616 | 0.6541 |

## d3-cite vs OpenAI / 009 — nDCG@5 deltas

- 65 queries total
-  12 queries d3 wins by ≥ 0.05 nDCG@5
-   8 queries d3 loses by ≥ 0.05 nDCG@5
-  45 queries within ±0.05 (effectively tied or 0/0)

### Where d3-cite *wins* by ≥ 0.05 nDCG@5

| qid | type | topic | path | OA | d3 | Δ nDCG@5 | Δ MRR | query |
|---|---|---|---|---:|---:|---:|---:|---|
| q038 | keyw | campai | fusion | 0.000 | 0.485 | **+0.485** | +0.857 | 85304 transfer ban candidate controlled committee slate mail |
| q056 | keyw | other | fusion | 0.170 | 0.515 | **+0.345** | +0.389 | section 89001 government newsletter mass mailing public expe |
| q049 | keyw | gifts_ | fusion | 0.470 | 0.718 | **+0.248** |   0.000 | 89518 campaign funds candidate self-compensation consulting  |
| q021 | fact | confli | bm25_onl | 0.131 | 0.339 | **+0.208** | +0.800 | A Half Moon Bay city council member owns a historic home loc |
| q029 | fact | confli | bm25_onl | 0.000 | 0.197 | **+0.197** | +0.281 | A city council member is also a licensed stockbroker. One of |
| q043 | natu | campai | fusion | 0.509 | 0.699 | **+0.190** |   0.000 | What are the requirements for a political committee to quali |
| q036 | fact | campai | fusion | 0.553 | 0.723 | **+0.170** |   0.000 | An engineering firm is named as a subcontractor in a contrac |
| q055 | fact | lobbyi | fusion | 0.773 | 0.930 | **+0.156** |   0.000 | Our lobbying firm assists school districts in applying for s |
| q044 | keyw | gifts_ | fusion | 0.424 | 0.555 | **+0.131** |   0.000 | 89501 89502 honorarium ban designated employee op-ed article |
| q040 | fact | campai | fusion | 0.619 | 0.699 | **+0.080** |   0.000 | A political party's county central committee has already con |
| q039 | natu | campai | fusion | 0.844 | 0.922 | **+0.078** |   0.000 | May a candidate-controlled committee make independent expend |
| q061 | keyw | other | fusion | 0.485 | 0.553 | **+0.068** |   0.000 | section 82041 definition local government agency Siegel four |

### Where d3-cite *loses* by ≥ 0.05 nDCG@5

| qid | type | topic | path | OA | d3 | Δ nDCG@5 | Δ MRR | query |
|---|---|---|---|---:|---:|---:|---:|---|
| q001 | keyw | confli | fusion | 0.339 | 0.000 | **-0.339** | -1.000 | Section 87103(a) disqualification business entity investment |
| q018 | keyw | confli | fusion | 0.339 | 0.000 | **-0.339** | -0.875 | post-employment revolving door one-year ban permanent ban Se |
| q046 | keyw | gifts_ | fusion | 0.740 | 0.544 | **-0.196** |   0.000 | Section 89503 gift limit exemption part-time governing board |
| q048 | natu | gifts_ | fusion | 1.000 | 0.830 | **-0.170** |   0.000 | Does Section 89506(a)(2) exempt travel payments from the gif |
| q041 | keyw | campai | fusion | 0.956 | 0.815 | **-0.140** |   0.000 | 84224 behested payment elected officer fundraising governmen |
| q060 | natu | other | bm25_onl | 0.339 | 0.214 | **-0.125** | -0.500 | Which entity is the code reviewing body for a joint powers a |
| q012 | keyw | confli | fusion | 0.229 | 0.140 | **-0.089** | -0.300 | Section 87103(e) personal financial effect official expenses |
| q054 | keyw | lobbyi | fusion | 0.670 | 0.597 | **-0.073** |   0.000 | placement agent contingency fee 86205(f) percentage of asset |

### Win/loss breakdown by topic and type

**By topic**:

| topic | n | wins | losses | flat | mean Δ nDCG@5 |
|---|---:|---:|---:|---:|---:|
| campaign_finance | 14 | 5 | 1 | 8 | +0.0702 |
| conflicts_of_interest | 29 | 2 | 3 | 24 | -0.0117 |
| gifts_honoraria | 7 | 2 | 2 | 3 | +0.0010 |
| lobbying | 5 | 1 | 1 | 3 | +0.0243 |
| other | 10 | 2 | 1 | 7 | +0.0262 |

**By type**:

| type | n | wins | losses | flat | mean Δ nDCG@5 |
|---|---:|---:|---:|---:|---:|
| fact_pattern | 17 | 5 | 0 | 12 | +0.0469 |
| keyword | 26 | 5 | 6 | 15 | +0.0086 |
| natural_language | 22 | 2 | 2 | 18 | +0.0006 |

**By routing path**:

| routing path | n | wins | losses | flat | mean Δ nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_only_noncite | 29 | 2 | 1 | 26 | +0.0110 |
| fusion | 36 | 10 | 7 | 19 | +0.0199 |

## COI deep-dive — where the remaining gap to OpenAI lives

- 29 COI queries; OpenAI mean nDCG@5 = 0.1892, d3-cite mean = 0.1775, mean Δ = -0.0117

### All 29 COI queries — d3-cite vs OpenAI
| qid | type | path | OA | d3 | Δ | query |
|---|---|---|---:|---:|---:|---|
| q001 | keyw | fusion | 0.339 | 0.000 | **-0.339** 🔻 | Section 87103(a) disqualification business entity investment |
| q018 | keyw | fusion | 0.339 | 0.000 | **-0.339** 🔻 | post-employment revolving door one-year ban permanent ban Se |
| q012 | keyw | fusion | 0.229 | 0.140 | **-0.089** 🔻 | Section 87103(e) personal financial effect official expenses |
| q024 | keyw | fusion | 0.170 | 0.146 | **-0.024** | Section 1091 1091.5 remote interest noninterest exception co |
| q002 | natu | bm25_onl | 0.000 | 0.000 | **  0.000** | Does a planning commissioner need to recuse if they provide  |
| q003 | fact | bm25_onl | 0.640 | 0.640 | **  0.000** | A city council member owns a home within a landscaping and l |
| q004 | keyw | fusion | 0.000 | 0.000 | **  0.000** | real property 500 feet proximity recusal Regulation 18702.2  |
| q005 | natu | bm25_onl | 0.723 | 0.723 | **  0.000** | Can a city council member vote on a street widening project  |
| q006 | fact | bm25_onl | 0.000 | 0.000 | **  0.000** | A planning commissioner owns a rental property approximately |
| q007 | keyw | fusion | 0.339 | 0.339 | **  0.000** | Section 87103(c) source of income disqualification employer  |
| q008 | natu | bm25_onl | 0.146 | 0.146 | **  0.000** | Is a school board member disqualified from voting on a contr |
| q009 | fact | bm25_onl | 0.363 | 0.363 | **  0.000** | A planning commissioner has a contract with a major technolo |
| q010 | keyw | fusion | 0.000 | 0.000 | **  0.000** | Section 87103(d) gift disqualification $250 threshold report |
| q011 | natu | bm25_onl | 0.000 | 0.000 | **  0.000** | Must a council member recuse from a redevelopment decision i |
| q013 | natu | bm25_onl | 0.363 | 0.363 | **  0.000** | Does a council member have a conflict of interest when votin |
| q014 | fact | bm25_onl | 0.339 | 0.339 | **  0.000** | A Fort Bragg city council member owns a small retail busines |
| q015 | keyw | fusion | 0.000 | 0.000 | **  0.000** | Government Code 1090 self-dealing financial interest contrac |
| q017 | fact | fusion | 0.000 | 0.000 | **  0.000** | A water district board member is also a partner in an engine |
| q019 | natu | bm25_onl | 0.229 | 0.229 | **  0.000** | Can a retired state agency deputy director work as a consult |
| q020 | keyw | fusion | 0.000 | 0.000 | **  0.000** | public generally exception Regulation 18703 significant segm |
| q022 | keyw | fusion | 0.000 | 0.000 | **  0.000** | legally required participation rule of necessity Regulation  |
| q023 | natu | bm25_onl | 0.000 | 0.000 | **  0.000** | Can a conflicted board member participate in a decision if t |
| q025 | natu | fusion | 0.592 | 0.592 | **  0.000** | Does a city council member's ownership of less than 3% of a  |
| q026 | keyw | fusion | 0.000 | 0.000 | **  0.000** | spouse community property income disqualification conflict o |
| q027 | natu | bm25_onl | 0.000 | 0.000 | **  0.000** | Is a school board member required to report and potentially  |
| q028 | keyw | bm25_onl | 0.000 | 0.000 | **  0.000** | common law conflict of interest incompatible offices fiducia |
| q016 | natu | bm25_onl | 0.544 | 0.592 | **+0.048** | Can a county board member who serves as an unpaid trustee of |
| q029 | fact | bm25_onl | 0.000 | 0.197 | **+0.197** ✅ | A city council member is also a licensed stockbroker. One of |
| q021 | fact | bm25_onl | 0.131 | 0.339 | **+0.208** ✅ | A Half Moon Bay city council member owns a historic home loc |

## d3-cite vs d3-allpath — what the routing extension changes

10 queries where allpath changes the result by ≥ 0.01 on nDCG@5 or MRR.

| qid | type | topic | cite-path → all-path | Δ nDCG@5 | Δ MRR | query |
|---|---|---|---|---:|---:|---|
| q003 | fact | confli | bm25_onl → fusion | **+0.214** |   0.000 | A city council member owns a home within a landsca |
| q045 | natu | gifts_ | bm25_onl → fusion | **+0.212** |   0.000 | Can an elected city council member who runs a spea |
| q053 | natu | lobbyi | bm25_onl → fusion | **+0.146** |   0.000 | Can a registered lobbying firm host a $2,500 congr |
| q047 | fact | gifts_ | bm25_onl → fusion | **+0.073** |   0.000 | A state agency is sponsoring a charity golf tourna |
| q002 | natu | confli | bm25_onl → fusion | **  0.000** | +0.104 | Does a planning commissioner need to recuse if the |
| q006 | fact | confli | bm25_onl → fusion | **  0.000** | +0.091 | A planning commissioner owns a rental property app |
| q011 | natu | confli | bm25_onl → fusion | **  0.000** | +0.056 | Must a council member recuse from a redevelopment  |
| q008 | natu | confli | bm25_onl → fusion | **-0.015** | -0.050 | Is a school board member disqualified from voting  |
| q016 | natu | confli | bm25_onl → fusion | **-0.048** |   0.000 | Can a county board member who serves as an unpaid  |
| q021 | fact | confli | bm25_onl → fusion | **-0.170** | -0.667 | A Half Moon Bay city council member owns a histori |

## d3 vs f2 (cite-only best) — ensemble candidate analysis

Per-query nDCG@5 for the two fine-tunes at their respective best configs.

- mean nDCG@5 (d3 best):  0.3911
- mean nDCG@5 (f2 best):  0.3893
- mean nDCG@5 (per-query MAX of d3, f2): **0.3938** (oracle upper bound for a query-router ensemble)
- mean nDCG@5 (OpenAI):   0.3752

2 queries where d3 and f2 differ by ≥ 0.10 nDCG@5 (ensemble would have most to gain on these).

| qid | type | topic | d3 | f2 | Δ |
|---|---|---|---:|---:|---:|
| q058 | keyw | other | 0.364 | 0.212 | -0.153 |
| q030 | keyw | campai | 0.214 | 0.339 | +0.125 |

