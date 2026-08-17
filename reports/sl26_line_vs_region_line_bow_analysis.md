# Swedish Lion 26: line vs. region+line dataset analysis

## Scope and method

- Compared `data/eval_swedish_lion_26/line_sl26_refit_bins_dataset.csv` (the **line** dataset) with `data/eval_swedish_lion_26/region_line_sl26_refit_bins_dataset.csv` (the **region+line** dataset).
- The requested absolute `/home/dgxuser/erik/projects/QualityPrediction/...` path was not present in this environment; the repository copy at `data/eval_swedish_lion_26/region_line_sl26_refit_bins_dataset.csv` was analyzed instead.
- Pages were joined one-to-one on `source_page_id`. Both files have unique `source_page_id` values.
- “Difference” is `region+line target_bow_f1 − line target_bow_f1`; ranking uses its absolute value. Ties are ordered by `source_page_id`.
- Aggregate scores are macro averages over pages, consistent with the evaluator's use of the arithmetic mean. SD is population standard deviation. Non-finite values are omitted per metric.

## Coverage

| Dataset / comparison | Pages |
|---|---:|
| Line dataset | 1727 |
| Region+line dataset | 1679 |
| Matched pages | 1679 |
| Only in line | 48 |
| Only in region+line | 0 |

All 1679 region+line pages match a line page. The line dataset has 48 additional pages, so both full-dataset and matched-page aggregates are shown.

## Aggregate scores: each full dataset

| Score | Line N | Line mean | Line median | Line SD | Region+line N | Region+line mean | Region+line median | Region+line SD | Mean delta (R−L) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BoW F1 | 1727 | 0.870671 | 0.897822 | 0.108645 | 1679 | 0.837505 | 0.891892 | 0.171153 | -0.033166 |
| BoW precision | 1727 | 0.845767 | 0.878289 | 0.126001 | 1679 | 0.839800 | 0.885417 | 0.152617 | -0.005967 |
| BoW recall | 1727 | 0.904265 | 0.924444 | 0.090357 | 1679 | 0.861238 | 0.918782 | 0.175803 | -0.043027 |
| line mAP@50 | 1727 | 0.961621 | 0.999301 | 0.112112 | 1679 | 0.053990 | 0.001035 | 0.133975 | -0.907630 |
| permutation CER | 1719 | 0.135195 | 0.083261 | 0.224511 | 1670 | 0.156737 | 0.093255 | 0.356585 | +0.021542 |

## Aggregate scores: matched pages only

| Score | Line N | Line mean | Line median | Line SD | Region+line N | Region+line mean | Region+line median | Region+line SD | Mean delta (R−L) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BoW F1 | 1679 | 0.870365 | 0.897881 | 0.109806 | 1679 | 0.837505 | 0.891892 | 0.171153 | -0.032860 |
| BoW precision | 1679 | 0.845254 | 0.878327 | 0.127347 | 1679 | 0.839800 | 0.885417 | 0.152617 | -0.005455 |
| BoW recall | 1679 | 0.904367 | 0.925000 | 0.091235 | 1679 | 0.861238 | 0.918782 | 0.175803 | -0.043129 |
| line mAP@50 | 1679 | 0.960836 | 0.999274 | 0.113574 | 1679 | 0.053990 | 0.001035 | 0.133975 | -0.906846 |
| permutation CER | 1671 | 0.136176 | 0.083179 | 0.227453 | 1670 | 0.156737 | 0.093255 | 0.356585 | +0.020561 |

## Paired BoW F1 summary on 1679 matched pages

| Statistic | Value |
|---|---:|
| Mean signed difference (R−L) | -0.032860 |
| Median signed difference (R−L) | -0.001989 |
| Mean absolute difference | 0.064526 |
| Median absolute difference | 0.016856 |
| Region+line higher | 768 |
| Equal | 16 |
| Region+line lower | 895 |

## 50 pages with the largest absolute BoW F1 difference

| Rank | `source_page_id` | Line BoW F1 | Region+line BoW F1 | Difference (R−L) | Absolute difference |
|---:|---|---:|---:|---:|---:|
| 1 | `Bergskollegium_Relationer_och_skrivelser_E3_10_1718-1727__40004031_00181__ff707494ef8f` | 1.000000 | 0.000000 | -1.000000 | 1.000000 |
| 2 | `Bergskollegium_Relationer_och_skrivelser_E3_5_1691-1692__40004026_00455__5d0a4edb6eb0` | 1.000000 | 0.000000 | -1.000000 | 1.000000 |
| 3 | `Bergskollegium_Relationer_och_skrivelser_E3_5_1691-1692__40004026_00459__d7f2c65853b7` | 1.000000 | 0.000000 | -1.000000 | 1.000000 |
| 4 | `SSA_Borgm_stare_och_R_d_f_re_1636__0384_dam_42__db1833f29cf5` | 1.000000 | 0.000000 | -1.000000 | 1.000000 |
| 5 | `SSA_S_dra_f_rstadens_k_mn_rsr_tt_1-8__0216_00220-scan_2021-03-02_10-48-29__1eb51dd171fd` | 1.000000 | 0.000000 | -1.000000 | 1.000000 |
| 6 | `Dalarna_18__B0001017_00405__16f54a920b8d` | 0.932584 | 0.021978 | -0.910606 | 0.910606 |
| 7 | `Norrland_20__B0001019_00013__228c03604f92` | 0.893300 | 0.000000 | -0.893300 | 0.893300 |
| 8 | `sterg_tland_5__B0001004_00218__165236e77129` | 0.893382 | 0.007246 | -0.886136 | 0.886136 |
| 9 | `SSA_Sollentuna_h_radsr_tt_9-16__0851_0275full__494973c6b013` | 0.894915 | 0.013605 | -0.881310 | 0.881310 |
| 10 | `VKaariainen_1600-luku_GT__0768_IMG_20180207_102352__7a98b042545e` | 0.000000 | 0.881029 | +0.881029 | 0.881029 |
| 11 | `Dalarna_19__B0001018_00047__72e53686588f` | 0.939597 | 0.075000 | -0.864597 | 0.864597 |
| 12 | `V_stmanland_17__B0001016_00095__54af3f050ce2` | 0.902597 | 0.038462 | -0.864136 | 0.864136 |
| 13 | `SSA_Politikollegiet_30-35__0315_Bfull_179__de6f0fec7bb2` | 0.886228 | 0.024096 | -0.862131 | 0.862131 |
| 14 | `Sm_land_9__B0001008_00116__8cd43ff9fa6e` | 0.861386 | 0.000000 | -0.861386 | 0.861386 |
| 15 | `SSA_Borgm_stare_och_R_d_f_re_1636__0015_00016-scan_2023-02-21_11-04-31__ab151a815fa0` | 0.857143 | 0.000000 | -0.857143 | 0.857143 |
| 16 | `VKaariainen_1600-luku_GT__0873_IMG_20180206_145322__badd95b2e163` | 0.000000 | 0.822171 | +0.822171 | 0.822171 |
| 17 | `Dalarna_18__B0001017_00496__7f0c9af2be13` | 0.876712 | 0.055556 | -0.821157 | 0.821157 |
| 18 | `sterg_tland_5__B0001004_00094__e877064c6cac` | 0.876404 | 0.063694 | -0.812710 | 0.812710 |
| 19 | `Gbg_poliskammare_2__30002048_00398__f04126c62e81` | 0.938480 | 0.138249 | -0.800231 | 0.800231 |
| 20 | `VKaariainen_1600-luku_GT__0847_IMG_20180207_113353__44db1da8a32f` | 0.000000 | 0.771930 | +0.771930 | 0.771930 |
| 21 | `Norrland_20__B0001019_00130__1baeceff2ad1` | 0.805556 | 0.036364 | -0.769192 | 0.769192 |
| 22 | `SSA_Sollentuna_h_radsr_tt_9-16__0858_0282full__907628a8eacc` | 0.801113 | 0.045584 | -0.755529 | 0.755529 |
| 23 | `Uppland_2__B0001001_00105__bc882e8ec0a3` | 0.866071 | 0.114754 | -0.751317 | 0.751317 |
| 24 | `sterg_tland_6__B0001005_00004__c1616c2c5a4d` | 0.899497 | 0.178404 | -0.721094 | 0.721094 |
| 25 | `V_stmanland_och_S_dermanland_16__B0001015_00215__1d364eea3c33` | 0.872131 | 0.174419 | -0.697713 | 0.697713 |
| 26 | `GT-FNA_1500__1541_0042__ebf1fb87a184` | 0.704082 | 0.011628 | -0.692454 | 0.692454 |
| 27 | `Bergskollegium_Relationer_och_skrivelser_E3_19_1748-1753__40006558_00122__d5cdd0e56088` | 0.333333 | 1.000000 | +0.666667 | 0.666667 |
| 28 | `SSA_Stadens_k_mn_rsr_tt_51-52__0750_00117-scan_2025-02-21_14-52-03__c2e97bd8bc28` | 0.666667 | 0.000000 | -0.666667 | 0.666667 |
| 29 | `SSA_Stockholms_Magistrat_och_R_dhusr_tt__0439_Mild_77__00d2ddf52c99` | 0.666667 | 0.000000 | -0.666667 | 0.666667 |
| 30 | `Norrland_Finland_och_hela_landet_21__B0001020_00066__ae4b998243cb` | 0.869565 | 0.211838 | -0.657727 | 0.657727 |
| 31 | `Uppland_8__A0052532_00077__67661e31cf46` | 0.813880 | 0.183784 | -0.630096 | 0.630096 |
| 32 | `SSA_Stockholms_domkapitel_11-19__2126_00340-scan_2020-02-20_19-28-52__297863be7852` | 0.899408 | 0.281407 | -0.618001 | 0.618001 |
| 33 | `SSA_Politikollegiet_16-22__1092_0002full__3d8facbb3f21` | 1.000000 | 0.400000 | -0.600000 | 0.600000 |
| 34 | `SSA_Stockholms_domkapitel_20-32__1624_0028full__31cc096909cd` | 0.600000 | 0.000000 | -0.600000 | 0.600000 |
| 35 | `sterg_tland_4__B0001003_00139__bc8b04335654` | 0.915254 | 0.342857 | -0.572397 | 0.572397 |
| 36 | `V_stmanland_15__B0001014_00048__13d41d96a2f1` | 0.897297 | 0.330435 | -0.566863 | 0.566863 |
| 37 | `SSA_S_dra_f_rstadens_k_mn_rsr_tt_9-16__2353_lck_42__8b320436040c` | 0.768595 | 0.206278 | -0.562317 | 0.562317 |
| 38 | `V_stmanland_15__B0001014_00349__edd8d16f6c9b` | 0.770701 | 0.219780 | -0.550920 | 0.550920 |
| 39 | `SSA_Stockholms_domkapitel_11-19__2118_00332-scan_2020-02-20_19-26-10__56e009d2863a` | 0.947846 | 0.406015 | -0.541831 | 0.541831 |
| 40 | `SSA_Sollentuna_h_radsr_tt_1-8__1521_0012full__4471db918ea0` | 0.824000 | 0.289655 | -0.534345 | 0.534345 |
| 41 | `SSA_Stadens_k_mn_rsr_tt_51-52__0742_00108-scan_2025-02-21_14-48-42__db51bf394ec0` | 0.526316 | 0.000000 | -0.526316 | 0.526316 |
| 42 | `SSA_Politikollegiet_30-35__0738_0340full__b3a598eadaaa` | 0.708333 | 0.190840 | -0.517494 | 0.517494 |
| 43 | `V_stmanland_och_S_dermanland_16__B0001015_00005__5852173823fe` | 0.897638 | 0.385542 | -0.512096 | 0.512096 |
| 44 | `SSA_Borgm_stare_och_R_d_f_re_1636__0809_00164-scan_2025-03-21_14-46-38__5de32e89d3ef` | 0.500000 | 0.000000 | -0.500000 | 0.500000 |
| 45 | `SSA_Norra_f_rstadens_v_stra_k_mn_rsr_tt_16-19__1041_0664full__85787eed52a7` | 0.848485 | 0.348485 | -0.500000 | 0.500000 |
| 46 | `Dalarna_18__B0001017_00009__d861cf21bc44` | 0.861446 | 0.363636 | -0.497809 | 0.497809 |
| 47 | `N_rke_o_V_rmland_1__A0052675_00038__0f0c06a9f1e0` | 0.772532 | 0.298611 | -0.473921 | 0.473921 |
| 48 | `Sm_land_10__B0001009_00262__8663fb81aaf5` | 0.950000 | 0.488889 | -0.461111 | 0.461111 |
| 49 | `Sm_land_22__A0052609_00008__6465aa16ea55` | 0.824176 | 0.364407 | -0.459769 | 0.459769 |
| 50 | `V_stmanland_17__B0001016_00279__1935864195b6` | 0.811594 | 0.352273 | -0.459321 | 0.459321 |

## Interpretation

The full-data macro BoW F1 is lower for region+line than for line. The matched-only table is the fairer pipeline comparison because it removes the 48 pages absent from region+line. Line mAP@50 changes much more dramatically than BoW F1, indicating that line localization behavior differs substantially between the two pipeline outputs; this report does not attempt to establish the cause.
