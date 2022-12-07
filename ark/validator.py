from ark.solver import Solver
from ark.cdg.cdg import CDG, CDGNode, CDGEdge
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValRule

class ArkValidator:

    def __init__(self, solver: Solver) -> None:
        self._solver = solver
        pass

    def validate(self, cdg: CDG, cdg_spec: CDGSpec):

        node: CDGNode
        edge: CDGEdge
        rule: ValRule

        rule_dict = cdg_spec.val_rule_dict

        for node in cdg.nodes:
            candidate_rules = rule_dict[node.cdg_type.type_name]
            print(node)
            print(candidate_rules)
            for rule in candidate_rules:
                rule.get_validation_matrix(node=node)
        return True