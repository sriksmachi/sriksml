
class Rule:
    def __init__(self, rule_id, rule_name, rule_description, rule_section) -> None:
        self.rule_id = rule_id
        self.rule_name = rule_name
        # This stores the prom
        self.rule_description = rule_description
        self.rule_sections = rule_section


class RulesEngine:
    def __init__(self) -> None:
        self.rules = []
        self.rules.append(Rule(1, "Rule 1", "Rule 1 Description", "Section 1"))

    def get_rules(self):
        return self.rules
