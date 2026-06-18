"""
Meal plan optimizer.
Greedy baseline + NSGA-II multi-objective optimization.
"""
import random
import numpy as np
from backend.schemas.user import UserProfileIn, NutrientTargets
from backend.utils.exceptions import NoFeasiblePlanError


class MealOptimizer:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def optimize(
        self,
        profile: UserProfileIn,
        daily_targets: NutrientTargets,
        scored_candidates: dict[str, list[dict]],
    ) -> list[dict]:
        meal_types = ["breakfast", "lunch", "dinner"]
        has_all_meals = all(scored_candidates.get(mt) for mt in meal_types)

        if has_all_meals:
            try:
                plan = self.nsga2_optimize(
                    profile, daily_targets, scored_candidates,
                    population_size=80, generations=40,
                )
                if plan:
                    return plan
            except Exception:
                pass

        meal_plans = self._greedy_optimize(
            profile, daily_targets, scored_candidates
        )

        if not meal_plans:
            raise NoFeasiblePlanError(
                "Không tìm được thực đơn phù hợp. Hãy thử nới lỏng sở thích hoặc tăng ngân sách."
            )

        return meal_plans

    def _greedy_optimize(
        self,
        profile: UserProfileIn,
        daily_targets: NutrientTargets,
        scored_candidates: dict[str, list[dict]],
    ) -> list[dict]:
        best_plan = None
        best_score = -float("inf")

        for _ in range(50):
            plan = []
            total_cal = 0
            total_sodium = 0
            total_sugar = 0
            used_ids = set()

            for meal_type in ["breakfast", "lunch", "dinner"]:
                candidates = scored_candidates.get(meal_type, [])
                available = [
                    c for c in candidates
                    if c["food_id"] not in used_ids
                ]

                if not available:
                    break

                remaining_cal = daily_targets.calories_kcal - total_cal
                remaining_meals = 3 - len(plan)
                ideal_cal = remaining_cal / max(remaining_meals, 1)

                scored = []
                for c in available:
                    cal = c.get("calories_kcal", 0)
                    cal_fit = max(0, 1.0 - abs(cal - ideal_cal) / max(ideal_cal, 1))
                    suit = c.get("suitability_score", 0.5)
                    combined = 0.5 * suit + 0.3 * cal_fit + 0.2 * random.random()
                    scored.append((combined, c))

                scored.sort(key=lambda x: x[0], reverse=True)
                top_choices = scored[:min(5, len(scored))]
                chosen_score, chosen = random.choice(top_choices)

                plan.append({**chosen, "meal_type": meal_type})
                used_ids.add(chosen["food_id"])
                total_cal += chosen.get("calories_kcal", 0)
                total_sodium += chosen.get("sodium_mg", 0)
                total_sugar += chosen.get("sugar_g", 0)

            if len(plan) == 3:
                cal_deviation = abs(total_cal - daily_targets.calories_kcal) / max(daily_targets.calories_kcal, 1)
                avg_suit = np.mean([p.get("suitability_score", 0.5) for p in plan])

                sodium_ok = total_sodium <= daily_targets.sodium_mg
                sugar_ok = total_sugar <= daily_targets.sugar_g

                plan_score = (
                    avg_suit * 0.4
                    + (1.0 - min(cal_deviation, 1.0)) * 0.3
                    + (0.15 if sodium_ok else 0.0)
                    + (0.15 if sugar_ok else 0.0)
                )

                if plan_score > best_score:
                    best_score = plan_score
                    best_plan = plan

        return best_plan or []

    def nsga2_optimize(
        self,
        profile: UserProfileIn,
        daily_targets: NutrientTargets,
        scored_candidates: dict[str, list[dict]],
        population_size: int = 100,
        generations: int = 50,
    ) -> list[dict]:
        meal_types = ["breakfast", "lunch", "dinner"]
        candidates_by_meal = {}
        for mt in meal_types:
            cands = scored_candidates.get(mt, [])
            if not cands:
                raise NoFeasiblePlanError(
                    f"Không có ứng viên cho bữa {mt}. Hãy thử thay đổi sở thích."
                )
            candidates_by_meal[mt] = cands

        try:
            from pymoo.core.problem import Problem
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize

            class MealPlanProblem(Problem):
                def __init__(self):
                    n_var = 3
                    xl = np.array([0, 0, 0])
                    xu = np.array([
                        len(candidates_by_meal[mt]) - 1 for mt in meal_types
                    ])
                    super().__init__(
                        n_var=n_var, n_obj=3, n_constr=3,
                        xl=xl, xu=xu, type_var=int,
                    )

                def _evaluate(self, x, out, *args, **kwargs):
                    f1, f2, f3 = [], [], []
                    g1, g2, g3 = [], [], []

                    for row in x:
                        total_cal = 0
                        total_sodium = 0
                        total_sugar = 0
                        total_cost = 0
                        total_suit = 0

                        used_ids = set()
                        has_duplicate = 0
                        for i, mt in enumerate(meal_types):
                            idx = int(row[i])
                            idx = min(idx, len(candidates_by_meal[mt]) - 1)
                            food = candidates_by_meal[mt][idx]
                            total_cal += food.get("calories_kcal", 0)
                            total_sodium += food.get("sodium_mg", 0)
                            total_sugar += food.get("sugar_g", 0)
                            total_cost += food.get("cost_estimate", 0)
                            total_suit += food.get("suitability_score", 0.5)

                            fid = food["food_id"]
                            if fid in used_ids:
                                has_duplicate += 1
                            used_ids.add(fid)

                        avg_suit = total_suit / 3
                        cal_dev = abs(total_cal - daily_targets.calories_kcal) / max(daily_targets.calories_kcal, 1)

                        f1.append(-avg_suit)
                        f2.append(cal_dev)
                        f3.append(total_cost / max(profile.budget_per_day or 200000, 1))

                        sodium_excess = max(0, total_sodium - daily_targets.sodium_mg) / max(daily_targets.sodium_mg, 1)
                        sugar_excess = max(0, total_sugar - daily_targets.sugar_g) / max(daily_targets.sugar_g, 1)
                        g1.append(sodium_excess)
                        g2.append(sugar_excess)
                        g3.append(float(has_duplicate))

                    out["F"] = np.column_stack([f1, f2, f3])
                    out["G"] = np.column_stack([g1, g2, g3])

            problem = MealPlanProblem()

            algorithm = NSGA2(
                pop_size=population_size,
                eliminate_duplicates=True,
            )

            res = minimize(
                problem, algorithm,
                termination=("n_gen", generations),
                seed=self.seed,
                verbose=False,
            )

            if res.X is None or (hasattr(res.X, 'ndim') and res.X.ndim == 2 and len(res.X) == 0):
                return self._greedy_optimize(profile, daily_targets, scored_candidates)

            if res.X.ndim == 2:
                F = res.F
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                distances = np.sqrt((F_norm ** 2).sum(axis=1))
                best_idx_arr = res.X[np.argmin(distances)]
            else:
                best_idx_arr = res.X

            plan = []
            for i, mt in enumerate(meal_types):
                idx = int(best_idx_arr[i])
                idx = min(idx, len(candidates_by_meal[mt]) - 1)
                food = candidates_by_meal[mt][idx]
                plan.append({**food, "meal_type": mt})

            used = set()
            has_dup = False
            for m in plan:
                if m["food_id"] in used:
                    has_dup = True
                used.add(m["food_id"])
            if has_dup:
                return self._greedy_optimize(profile, daily_targets, scored_candidates)

            return plan

        except ImportError:
            return self._greedy_optimize(profile, daily_targets, scored_candidates)
