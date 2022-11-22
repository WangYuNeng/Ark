class Parser:

    def __init__(self) -> None:
        pass

    

''' TODO Write separate pasers for cdg, cdg_type, generation rule, and validation rule
Parse 'gen v: Edge := expr' to a (CDGEdge, CDGNode, CDGNode, bool, ast) tuple
The Boolean value denotes the target of the generation rule

Parse 'rule NodePattern { ConnRules* }' to a (CDGNode, list) pair
'''