"""
Disease Rule Engine.
Evaluates food items against disease-specific and allergy rules from YAML config.
Returns RuleDecision with pass/reject/warn status and trace.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "disease_rules.yaml"


class RuleStatus(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    WARN = "warn"
    UNKNOWN = "unknown"


@dataclass
class RuleDecision:
    rule_id: str
    status: RuleStatus
    observed_value: float | str | list | None = None
    threshold: float | str | list | None = None
    reason: str = ""
    severity: str = "hard"


@dataclass
class FoodRuleResult:
    food_id: str
    decisions: list[RuleDecision] = field(default_factory=list)
    is_safe: bool = True
    warnings: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


class DiseaseRuleEngine:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.rules = self._load_rules(config_path)
        self.version = self._get_version(config_path)

    def _load_rules(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("rules", [])

    def _get_version(self, path: Path) -> str:
        versions = set()
        for rule in self.rules if hasattr(self, "rules") else []:
            versions.add(rule.get("version", "unknown"))
        return ",".join(sorted(versions)) if versions else "unknown"

    def filter_food(
        self,
        food: dict,
        diseases: list[str],
        allergies: list[str],
        disliked_ingredients: list[str] | None = None,
        lang: str = "vi",
    ) -> FoodRuleResult:
        result = FoodRuleResult(food_id=food["food_id"])

        for rule in self.rules:
            applies_to = rule["applies_to"]
            if applies_to != "all" and applies_to not in diseases:
                continue

            decision = self._evaluate_rule(rule, food, allergies, disliked_ingredients, lang)
            result.decisions.append(decision)

            if decision.status == RuleStatus.REJECT:
                result.is_safe = False
                result.rejections.append(decision.reason)
            elif decision.status == RuleStatus.WARN:
                result.warnings.append(decision.reason)

        return result

    def filter_all(
        self,
        foods: list[dict],
        diseases: list[str],
        allergies: list[str],
        disliked_ingredients: list[str] | None = None,
        lang: str = "vi",
    ) -> tuple[list[dict], dict]:
        safe_foods = []
        rule_trace = {}

        for food in foods:
            result = self.filter_food(food, diseases, allergies, disliked_ingredients, lang)
            rule_trace[food["food_id"]] = {
                "is_safe": result.is_safe,
                "decisions": [
                    {
                        "rule_id": d.rule_id,
                        "status": d.status.value,
                        "reason": d.reason,
                    }
                    for d in result.decisions
                ],
                "warnings": result.warnings,
                "rejections": result.rejections,
            }
            if result.is_safe:
                safe_foods.append(food)

        return safe_foods, rule_trace

    def _evaluate_rule(
        self,
        rule: dict,
        food: dict,
        allergies: list[str],
        disliked_ingredients: list[str] | None,
        lang: str,
    ) -> RuleDecision:
        rule_id = rule["rule_id"]
        target_field = rule["target_field"]
        operator = rule["operator"]
        threshold = rule["value"]
        severity = rule.get("severity", "hard")
        unknown_policy = rule.get("unknown_policy", "reject")
        reason_templates = rule.get("reason_template", {})

        if target_field == "ingredients":
            return self._evaluate_ingredient_rule(
                rule_id, food, operator, threshold, allergies,
                disliked_ingredients, severity, reason_templates, lang,
            )

        observed = food.get(target_field)
        if observed is None:
            status = self._unknown_to_status(unknown_policy)
            return RuleDecision(
                rule_id=rule_id,
                status=status,
                observed_value=None,
                threshold=threshold,
                reason=f"Field '{target_field}' not found" if status != RuleStatus.PASS else "",
                severity=severity,
            )

        passed = self._compare(observed, operator, threshold)
        status = RuleStatus.PASS if passed else (
            RuleStatus.REJECT if severity == "hard" else RuleStatus.WARN
        )

        reason = ""
        if not passed:
            reason = self._format_reason(reason_templates, lang, observed, threshold, [])

        return RuleDecision(
            rule_id=rule_id,
            status=status,
            observed_value=observed,
            threshold=threshold,
            reason=reason,
            severity=severity,
        )

    def _evaluate_ingredient_rule(
        self,
        rule_id: str,
        food: dict,
        operator: str,
        threshold: str,
        allergies: list[str],
        disliked_ingredients: list[str] | None,
        severity: str,
        reason_templates: dict,
        lang: str,
    ) -> RuleDecision:
        ingredients = set(food.get("ingredients", []))
        matched = set()

        if threshold == "${user_allergies}":
            allergen_ingredients = self._get_allergen_ingredients(allergies)
            matched = ingredients & allergen_ingredients
        elif threshold == "${user_dislikes}":
            disliked = set(disliked_ingredients or [])
            matched = ingredients & disliked

        if operator == "no_intersection":
            passed = len(matched) == 0
        else:
            passed = True

        status = RuleStatus.PASS if passed else (
            RuleStatus.REJECT if severity == "hard" else RuleStatus.WARN
        )

        reason = ""
        if not passed:
            reason = self._format_reason(
                reason_templates, lang, None, None, sorted(matched)
            )

        return RuleDecision(
            rule_id=rule_id,
            status=status,
            observed_value=sorted(ingredients),
            threshold=threshold,
            reason=reason,
            severity=severity,
        )

    def _get_allergen_ingredients(self, allergies: list[str]) -> set:
        from ml.pipelines.preprocess_foods import ALLERGEN_INGREDIENTS
        result = set()
        for allergen in allergies:
            if allergen in ALLERGEN_INGREDIENTS:
                result.update(ALLERGEN_INGREDIENTS[allergen])
        return result

    def _compare(self, observed, operator: str, threshold) -> bool:
        try:
            if operator == "<=":
                return float(observed) <= float(threshold)
            elif operator == ">=":
                return float(observed) >= float(threshold)
            elif operator == "<":
                return float(observed) < float(threshold)
            elif operator == ">":
                return float(observed) > float(threshold)
            elif operator == "==":
                return observed == threshold
            elif operator == "not_in":
                if isinstance(threshold, list):
                    return observed not in threshold
                return str(observed) not in [str(t) for t in threshold]
            elif operator == "in":
                if isinstance(threshold, list):
                    return observed in threshold
                return str(observed) in [str(t) for t in threshold]
        except (ValueError, TypeError):
            return False
        return True

    def _unknown_to_status(self, policy: str) -> RuleStatus:
        return {
            "reject": RuleStatus.REJECT,
            "warn": RuleStatus.WARN,
            "pass": RuleStatus.PASS,
        }.get(policy, RuleStatus.UNKNOWN)

    def _format_reason(
        self,
        templates: dict,
        lang: str,
        observed,
        threshold,
        matched: list,
    ) -> str:
        template = templates.get(lang, templates.get("vi", ""))
        try:
            return template.format(
                observed=observed or 0,
                threshold=threshold or 0,
                matched=", ".join(matched),
            )
        except (KeyError, ValueError, IndexError):
            return template
