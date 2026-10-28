# Ablation: the same algorithm with and without the constraint information

Mean test AOCC over seeds. blind = f is the penalised value, g = 0 (what the penalty arm sees).
penalty arm = best algorithm of the penalty arm of the same repetition, for reference.

| problem | repetition | arm | algorithm | normal | blind | blind - normal | p (Wilcoxon) | penalty arm |
|---|---|---|---|---|---|---|---|---|
| cec/RC12 | haiku5 | free | AdaptiveMomentumDE | 0.794 | 0.518 | -0.276 | 6.1e-05 | 0.522 |
| cec/RC12 | haiku5 | informed | MixedFeasibilityJDEDense | 0.643 | 0.508 | -0.135 | 0.000122 | 0.522 |
| cec/RC12 | haiku5 | informed_v2 | DecayedEliteCrossoverDE | 0.788 | 0.512 | -0.276 | 6.1e-05 | 0.522 |
| cec/RC12 | haiku5-r2 | free | RefinedMixedFeasibleDE | 0.681 | 0.509 | -0.172 | 6.1e-05 | 0.518 |
| cec/RC12 | haiku5-r2 | informed_v2 | RefinedFeasBanditDE | 0.799 | 0.504 | -0.295 | 6.1e-05 | 0.518 |
| cec/RC12 | haiku5-r3 | free | EliteFeasPoolDE | 0.736 | 0.513 | -0.223 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r3 | informed_v2 | FRMIDEPlusV13 | 0.682 | 0.513 | -0.169 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r4 | free | FeasibilityIPOPSHADEMixed | 0.652 | 0.510 | -0.142 | 0.000183 | 0.511 |
| cec/RC12 | haiku5-r4 | informed_v2 | AdaptiveWeightedJADEPolish | 0.734 | 0.510 | -0.225 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r5 | free | FeasibilityRuleMixedDE | 0.729 | 0.509 | -0.219 | 6.1e-05 | 0.519 |
| cec/RC12 | haiku5-r5 | informed_v2 | ConstraintWeightedElitistMixedDE | 0.657 | 0.513 | -0.144 | 6.1e-05 | 0.519 |
| lens/original | haiku5 | free | EliteEpsilonJADEPlus | 0.685 | 0.608 | -0.077 | 6.1e-05 | 0.604 |
| lens/original | haiku5 | informed | ScreenedLogitRefineDE | 0.609 | 0.605 | -0.004 | 0.72 | 0.604 |
| lens/original | haiku5 | informed_v2 | ConstraintFrontierMIDE | 0.616 | 0.579 | -0.037 | 0.00836 | 0.604 |
| lens/original | haiku5-r2 | free | EliteExtrapMomentumDE | 0.705 | 0.617 | -0.088 | 0.000122 | 0.597 |
| lens/original | haiku5-r2 | informed_v2 | DominantRepairBisectionES2 | 0.623 | 0.573 | -0.050 | 0.000427 | 0.597 |
| lens/original | haiku5-r3 | free | RefinedArchivePatternES | 0.684 | 0.590 | -0.093 | 6.1e-05 | 0.638 |
| lens/original | haiku5-r3 | informed_v2 | ConstraintGradientCreditES | 0.620 | 0.585 | -0.035 | 0.0103 | 0.638 |
| lens/original | haiku5-r4 | free | EpsDiscountBanditLocalSHADE | 0.694 | 0.611 | -0.083 | 6.1e-05 | 0.601 |
| lens/original | haiku5-r4 | informed_v2 | ConstraintArchiveJADEPlus | 0.714 | 0.624 | -0.090 | 6.1e-05 | 0.601 |
| lens/original | haiku5-r5 | free | RefinedFeasibilityMIDEv7 | 0.695 | 0.611 | -0.084 | 6.1e-05 | 0.616 |
| lens/original | haiku5-r5 | informed_v2 | ConstraintSurrogateBanditEA | 0.631 | 0.581 | -0.050 | 0.00427 | 0.616 |

Per problem, over all algorithms: mean normal / blind / penalty arm
* cec/RC12: 0.718 / 0.511 / 0.517 (11 algorithms)
* lens/original: 0.661 / 0.599 / 0.610 (11 algorithms)
