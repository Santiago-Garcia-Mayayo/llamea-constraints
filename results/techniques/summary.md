# Constraint-handling techniques of the generated algorithms

9988 algorithms classified (claude-haiku-5-5). Share of algorithms; 'best' = the best algorithm of each run.

### Primary technique (what decides selection)

| technique | free (n=4992) | informed_v2 (n=4996) | free, best (n=50) | informed_v2, best (n=50) |
|---|---|---|---|---|
| penalty_static | 0% | 0% | 0% | 0% |
| penalty_adaptive | 0% | 12% | 0% | 16% |
| feasibility_rules | 87% | 84% | 86% | 80% |
| epsilon_constrained | 12% | 2% | 14% | 4% |
| stochastic_ranking | 1% | 0% | 0% | 0% |
| repair | 0% | 0% | 0% | 0% |
| per_constraint_guidance | 0% | 0% | 0% | 0% |
| constraint_surrogate | 0% | 0% | 0% | 0% |

### Any use of each technique (multi-label)

| technique | free (n=4992) | informed_v2 (n=4996) | free, best (n=50) | informed_v2, best (n=50) |
|---|---|---|---|---|
| penalty_static | 1% | 0% | 4% | 0% |
| penalty_adaptive | 1% | 53% | 2% | 64% |
| feasibility_rules | 99% | 98% | 98% | 98% |
| epsilon_constrained | 12% | 3% | 14% | 4% |
| stochastic_ranking | 3% | 1% | 4% | 0% |
| repair | 35% | 69% | 42% | 78% |
| feasible_archive | 55% | 63% | 72% | 76% |
| per_constraint_guidance | 30% | 80% | 42% | 90% |
| boundary_search | 7% | 19% | 14% | 24% |
| constraint_surrogate | 0% | 10% | 0% | 10% |
| ignores_constraints | 0% | 0% | 0% | 0% |
| other | 1% | 1% | 0% | 0% |

### Optimiser family

| family | free (n=4992) | informed_v2 (n=4996) |
|---|---|---|
| differential_evolution | 55% | 52% |
| evolution_strategy_cmaes | 15% | 22% |
| genetic_algorithm | 5% | 5% |
| local_search_pattern | 0% | 0% |
| hybrid | 24% | 20% |
| random_search | 0% | 0% |
| other | 0% | 0% |

Uses the individual constraint values (not only their sum): free 1%, informed_v2 78%

### Primary technique per problem (all candidates of both arms)

| problem | n | most common | 2nd | 3rd |
|---|---|---|---|---|
| lens/original | 999 | feasibility_rules 68% | epsilon_constrained 25% | penalty_adaptive 6% |
| lens/realistic_draft | 999 | feasibility_rules 73% | epsilon_constrained 15% | penalty_adaptive 12% |
| cec/RC12 | 1000 | feasibility_rules 87% | penalty_adaptive 8% | epsilon_constrained 4% |
| cec/RC13 | 1000 | feasibility_rules 91% | penalty_adaptive 7% | epsilon_constrained 1% |
| cec/RC15 | 999 | feasibility_rules 90% | penalty_adaptive 8% | epsilon_constrained 2% |
| cec/RC16 | 995 | feasibility_rules 86% | epsilon_constrained 8% | penalty_adaptive 6% |
| cec/RC18 | 1000 | feasibility_rules 88% | epsilon_constrained 9% | penalty_adaptive 3% |
| cec/RC21 | 998 | feasibility_rules 97% | penalty_adaptive 2% | epsilon_constrained 1% |
| cec/RC25 | 1000 | feasibility_rules 90% | epsilon_constrained 6% | penalty_adaptive 4% |
| cec/RC26 | 998 | feasibility_rules 88% | penalty_adaptive 6% | stochastic_ranking 3% |

### Mean training fitness by primary technique (techniques with >= 10 algorithms in the problem)

| problem | technique | n | mean fitness | best fitness |
|---|---|---|---|---|
| lens/original | epsilon_constrained | 249 | 0.690 | 0.741 |
| lens/original | penalty_adaptive | 61 | 0.616 | 0.692 |
| lens/original | feasibility_rules | 677 | 0.612 | 0.712 |
| lens/realistic_draft | epsilon_constrained | 146 | 0.677 | 0.745 |
| lens/realistic_draft | penalty_adaptive | 118 | 0.628 | 0.684 |
| lens/realistic_draft | feasibility_rules | 729 | 0.620 | 0.728 |
| cec/RC12 | penalty_adaptive | 85 | 0.663 | 0.817 |
| cec/RC12 | feasibility_rules | 866 | 0.654 | 0.863 |
| cec/RC12 | epsilon_constrained | 39 | 0.601 | 0.843 |
| cec/RC13 | penalty_adaptive | 74 | 0.918 | 0.971 |
| cec/RC13 | feasibility_rules | 911 | 0.905 | 0.975 |
| cec/RC13 | epsilon_constrained | 13 | 0.815 | 0.876 |
| cec/RC15 | penalty_adaptive | 77 | 0.927 | 0.963 |
| cec/RC15 | feasibility_rules | 902 | 0.878 | 0.972 |
| cec/RC15 | epsilon_constrained | 19 | 0.769 | 0.886 |
| cec/RC16 | feasibility_rules | 854 | 0.499 | 0.756 |
| cec/RC16 | epsilon_constrained | 78 | 0.484 | 0.557 |
| cec/RC16 | penalty_adaptive | 55 | 0.467 | 0.681 |
| cec/RC18 | penalty_adaptive | 28 | 0.646 | 0.729 |
| cec/RC18 | feasibility_rules | 880 | 0.577 | 0.758 |
| cec/RC18 | epsilon_constrained | 91 | 0.570 | 0.742 |
| cec/RC21 | penalty_adaptive | 17 | 0.889 | 0.967 |
| cec/RC21 | feasibility_rules | 968 | 0.884 | 0.986 |
| cec/RC21 | epsilon_constrained | 13 | 0.766 | 0.942 |
| cec/RC25 | penalty_adaptive | 37 | 0.517 | 0.592 |
| cec/RC25 | feasibility_rules | 898 | 0.508 | 0.580 |
| cec/RC25 | epsilon_constrained | 59 | 0.498 | 0.549 |
| cec/RC26 | penalty_adaptive | 62 | 0.498 | 0.532 |
| cec/RC26 | feasibility_rules | 881 | 0.490 | 0.578 |
| cec/RC26 | stochastic_ranking | 30 | 0.474 | 0.534 |
| cec/RC26 | epsilon_constrained | 22 | 0.433 | 0.502 |

### Jumps between techniques (child vs its parent)

9489 parent-child pairs; the child changes primary technique in 829 (9%).

| parent -> child | n | mean fitness change | share improving |
|---|---|---|---|
| feasibility_rules -> feasibility_rules | 7855 | -0.034 | 22% |
| epsilon_constrained -> epsilon_constrained | 473 | -0.033 | 21% |
| penalty_adaptive -> penalty_adaptive | 307 | -0.033 | 12% |
| feasibility_rules -> penalty_adaptive | 293 | -0.046 | 22% |
| feasibility_rules -> epsilon_constrained | 254 | -0.069 | 15% |
| penalty_adaptive -> feasibility_rules | 182 | -0.082 | 15% |
| feasibility_rules -> stochastic_ranking | 28 | -0.043 | 29% |
| stochastic_ranking -> stochastic_ranking | 22 | -0.036 | 14% |
| per_constraint_guidance -> feasibility_rules | 16 | -0.047 | 12% |
| feasibility_rules -> per_constraint_guidance | 8 | -0.016 | 38% |
| per_constraint_guidance -> penalty_adaptive | 7 | -0.035 | 14% |
| penalty_static -> penalty_adaptive | 7 | -0.012 | 14% |
| repair -> feasibility_rules | 7 | -0.031 | 14% |
| penalty_adaptive -> per_constraint_guidance | 5 | -0.033 | 0% |
| epsilon_constrained -> feasibility_rules | 5 | -0.024 | 0% |
| feasibility_rules -> repair | 5 | +0.000 | 60% |
| stochastic_ranking -> feasibility_rules | 3 | -0.005 | 33% |
| feasibility_rules -> constraint_surrogate | 3 | -0.032 | 0% |
| penalty_adaptive -> epsilon_constrained | 2 | -0.039 | 0% |
| repair -> repair | 2 | -0.012 | 50% |

### Primary technique early vs late in the runs (first vs last 50 algorithms of each run)

### Share of algorithms

| technique | free early (n=2493) | informed_v2 early (n=2499) | free late (n=2499) | informed_v2 late (n=2497) |
|---|---|---|---|---|
| penalty_static | 0% | 0% | 0% | 0% |
| penalty_adaptive | 0% | 8% | 0% | 16% |
| feasibility_rules | 91% | 89% | 83% | 80% |
| epsilon_constrained | 8% | 2% | 16% | 3% |
| stochastic_ranking | 1% | 1% | 0% | 0% |
| repair | 0% | 0% | 0% | 0% |
| per_constraint_guidance | 0% | 0% | 0% | 1% |
| constraint_surrogate | 0% | 0% | 0% | 0% |

