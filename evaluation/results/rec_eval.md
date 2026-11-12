# Recommender evaluation

Catalogue: data\catalogue\clothing_catalogue.csv (52 garments)  
Scenarios: evaluation\rec_scenarios.yaml (12)  
Labels: {'catalogue': 'data/catalogue/clothing_catalogue.csv (52 garments, cd_001-cd_052)', 'labeller': 'AI assistant (Claude), from images and names, before running the evaluation', 'date': datetime.date(2026, 11, 8)}


## attributes only

| Scenario | P@5 | nDCG@5 | top-5 |
|---|---|---|---|
| Interview, indoor, mild - tops | 1.00 | 1.00 | cd_002, cd_004, cd_036, cd_039, cd_007 |
| Interview, indoor, mild - bottoms | 0.60 | 1.00 | cd_027, cd_029, cd_045, cd_030, cd_031 |
| Casual outing, outdoor, hot - tops | 1.00 | 0.65 | cd_052, cd_032, cd_033, cd_034, cd_037 |
| Casual outing, outdoor, hot - bottoms | 0.80 | 0.95 | cd_044, cd_041, cd_042, cd_043, cd_028 |
| College, indoor, mild - tops | 0.60 | 0.39 | cd_001, cd_052, cd_023, cd_038, cd_032 |
| Wedding, indoor, mild - all | 1.00 | 0.47 | cd_014, cd_027, cd_029, cd_002, cd_004 |
| Party, indoor, mild - all | 0.40 | 0.28 | cd_006, cd_008, cd_051, cd_012, cd_038 |
| Sports, outdoor, hot - all | 1.00 | 0.86 | cd_042, cd_043, cd_028, cd_041, cd_046 |
| Travel, outdoor, rainy - tops | 0.40 | 0.68 | cd_023, cd_001, cd_052, cd_038, cd_015 |
| Casual outing, outdoor, cold - tops | 1.00 | 0.90 | cd_019, cd_020, cd_021, cd_001, cd_015 |
| Office, indoor, mild - tops | 0.80 | 0.87 | cd_007, cd_008, cd_036, cd_039, cd_038 |
| Office, indoor, mild - dresses | 0.60 | 0.95 | cd_011, cd_012, cd_013, cd_014, cd_009 |

## CLIP only

| Scenario | P@5 | nDCG@5 | top-5 |
|---|---|---|---|
| Interview, indoor, mild - tops | 0.60 | 0.72 | cd_002, cd_004, cd_007, cd_005, cd_001 |
| Interview, indoor, mild - bottoms | 0.60 | 1.00 | cd_045, cd_027, cd_029, cd_030, cd_044 |
| Casual outing, outdoor, hot - tops | 0.80 | 0.87 | cd_046, cd_047, cd_037, cd_049, cd_016 |
| Casual outing, outdoor, hot - bottoms | 0.80 | 0.78 | cd_041, cd_030, cd_044, cd_045, cd_042 |
| College, indoor, mild - tops | 0.60 | 0.37 | cd_001, cd_038, cd_018, cd_005, cd_017 |
| Wedding, indoor, mild - all | 0.40 | 0.31 | cd_014, cd_009, cd_012, cd_010, cd_007 |
| Party, indoor, mild - all | 0.60 | 0.53 | cd_014, cd_009, cd_012, cd_005, cd_007 |
| Sports, outdoor, hot - all | 0.60 | 0.68 | cd_050, cd_051, cd_043, cd_048, cd_052 |
| Travel, outdoor, rainy - tops | 1.00 | 1.00 | cd_023, cd_024, cd_016, cd_025, cd_022 |
| Casual outing, outdoor, cold - tops | 0.80 | 0.83 | cd_017, cd_003, cd_005, cd_020, cd_021 |
| Office, indoor, mild - tops | 0.80 | 0.87 | cd_007, cd_039, cd_004, cd_002, cd_001 |
| Office, indoor, mild - dresses | 0.40 | 0.33 | cd_014, cd_009, cd_012, cd_013, cd_010 |

## hybrid (default)

| Scenario | P@5 | nDCG@5 | top-5 |
|---|---|---|---|
| Interview, indoor, mild - tops | 1.00 | 1.00 | cd_002, cd_004, cd_007, cd_036, cd_039 |
| Interview, indoor, mild - bottoms | 0.60 | 1.00 | cd_027, cd_029, cd_045, cd_030, cd_044 |
| Casual outing, outdoor, hot - tops | 1.00 | 1.00 | cd_046, cd_047, cd_037, cd_052, cd_049 |
| Casual outing, outdoor, hot - bottoms | 1.00 | 1.00 | cd_044, cd_041, cd_042, cd_043, cd_030 |
| College, indoor, mild - tops | 1.00 | 0.69 | cd_001, cd_038, cd_046, cd_049, cd_017 |
| Wedding, indoor, mild - all | 1.00 | 0.62 | cd_014, cd_012, cd_029, cd_027, cd_002 |
| Party, indoor, mild - all | 0.40 | 0.20 | cd_012, cd_006, cd_038, cd_008, cd_005 |
| Sports, outdoor, hot - all | 0.80 | 0.93 | cd_043, cd_050, cd_048, cd_042, cd_052 |
| Travel, outdoor, rainy - tops | 0.80 | 0.92 | cd_023, cd_016, cd_025, cd_024, cd_001 |
| Casual outing, outdoor, cold - tops | 1.00 | 1.00 | cd_020, cd_021, cd_017, cd_026, cd_015 |
| Office, indoor, mild - tops | 0.80 | 0.83 | cd_007, cd_039, cd_038, cd_004, cd_002 |
| Office, indoor, mild - dresses | 0.60 | 0.78 | cd_012, cd_013, cd_014, cd_011, cd_009 |

Context sensitivity: mean Kendall tau over 45 scenario pairs sharing >= 5 garments = 0.01 (min -0.87, max 0.73). 1 would mean the ranking ignores context.

## Summary

| Configuration | mean P@5 | mean nDCG@5 |
|---|---|---|
| attributes only | 0.767 | 0.751 |
| CLIP only | 0.667 | 0.691 |
| hybrid (default) | 0.833 | 0.830 |