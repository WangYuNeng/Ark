from itertools import product
from ark.solver import Solver
from ark.cdg.cdg import CDG, CDGNode, CDGEdge
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValRule, ValPattern
from ark.specification.rule_keyword import Target

Connection = list[CDGEdge]

class ArkValidator:

    def __init__(self, solver: Solver) -> None:
        self._solver = solver

    def validate(self, cdg: CDG, cdg_spec: CDGSpec) -> bool:
        """
        Every possible connection of the node in the cdg should:

        - Satisfy at leat one of the accepted patterns of its base types
        - Satisfy none of the rejected patterns of its base types
        - Satisfy all the checking functions of its base types
        """

        node_type: NodeType

        rule_dict = cdg_spec.val_rule_dict

        for node in cdg.nodes:
            node_type = node.cdg_type
            possible_conns = self._enumerate_switch_branches(node=node)
            base_types = node_type.base_cdg_types()

            # Check all possible connections
            for conns in possible_conns:
                accepted = False

                # Enumerate the base types
                for n_t in base_types:
                    if n_t not in rule_dict:
                        continue
                    rule = rule_dict[n_t]
                    acc_pats, rej_pats, check_fns = rule.acc_pats, rule.rej_pats, rule.checking_fns
                    if acc_pats and not accepted:
                        acc_sat = self._check_satisfy(patterns=acc_pats, node=node, conns=conns)
                        if acc_sat:
                            accepted = True
                    if rej_pats:
                        rej_sat = self._check_satisfy(patterns=rej_pats, node=node, conns=conns)

                        if rej_sat:
                            raise SyntaxError(f'Node {node}: Connection {conns}\
                                            satisfies a rejected pattern {rej_pats}')

                    for check_fn in check_fns:
                        if not check_fn(node):
                            raise SyntaxError(f'Node {node}: Connection {conns}\
                                              does not satisfy {check_fn.__name__}')

                if not accepted:
                    raise SyntaxError(f'Node {node}: Connection {conns}\
                                      does not satisfy any of the accepted patterns')

        return True

    def _enumerate_switch_branches(self, node: CDGNode)-> list[list[CDGEdge]]:
        """Return all the possible connection for the node.
        
        Go through every edge in the node. 
        If the edge is a switch we consider both it being ON or OFF
        """

        fixed = node.get_non_switchable()
        switchable = list(node.get_switchable())
        n_conn_options = 2 ** len(switchable)

        possible_conns = [list(fixed) for _ in range(n_conn_options)]
        for i, conn in enumerate(possible_conns):
            for j, edge in enumerate(switchable):
                if i & (1 << j):
                    conn.append(edge)
        return possible_conns

    def _check_satisfy(self, patterns: list[ValPattern], node: CDGNode, conns: Connection) -> bool:
        """Check if the node satisfies the pattern."""
        matrix = self._get_validation_matrix(patterns=patterns, node=node, conns=conns)
        constraints = [pattern.deg_range for pattern in patterns]
        return self._solver.solve_validation_matrix(matrix=matrix, constraints=constraints)

    def _get_validation_matrix(self, patterns: list[ValPattern], node: CDGNode,
                               conns: Connection) -> list[list[bool]]:
        """Return a matrix representing the validation problem."""
        matrix = [[0 for _ in range(len(patterns))] for _ in range(len(conns))]
        for i, conn in enumerate(conns):
            tgt = node.which_tgt(conn)
            neighbor = node.get_neighbor(conn)
            for j, pattern in enumerate(patterns):
                matrix[i][j] = pattern.check_match(tgt=tgt, edge=conn, node=neighbor)

        return matrix
        