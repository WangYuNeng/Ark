from types import FunctionType
from ark.visualize.latex_util import *
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.range import Range
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME

syn = lambda x: LatexPrettyPrinter.fmt(Terms.SYNTAX, x)
lit = lambda x: LatexPrettyPrinter.fmt(Terms.LITERAL, x)
expr = lambda x: LatexPrettyPrinter.fmt(Terms.EXPRESSION, x)
vari = lambda x: LatexPrettyPrinter.fmt(Terms.VARIABLE, x)
num = lambda x: LatexPrettyPrinter.fmt_number(x)
pct = lambda x: LatexPrettyPrinter.fmt_percent(x)

def special_variable(name):
    if name == "time":
        tok = syn("time")
    else:
        tok = f'{syn("var(")}{lit(name)}{syn(")")}'

    return tok

def format_variables(expr):
    print(expr)
    fn_subdict = {}
    for f in expr.atoms(sympy.Function):
        name = f.func.name
        assert("." in name)
        caller,fname = name.split(".")
        tok = lit(caller)
        tok += syn(".")
        tok += lit(fname)
        
        fn_subdict[f]  = sympy.Function(tok,real=True)(*f.args)
        

    fn_expr = expr.subs(fn_subdict)

    subdict = {}
    for sym in fn_expr.free_symbols:
        text = sym.name
        if "." in text:
            name,attr = text.split(".")
            tok = lit(name)
            tok += syn(".")
            tok += lit(attr)
        
        else:
            tok = special_variable(sym.name)

        subdict[sym] = sympy.Symbol(tok)
       
    for f in fn_expr.atoms(sympy.Float):
        subdict[f] = sympy.Symbol(num(f))



    sub_expr = fn_expr.subs(subdict)
    return sub_expr

def range_to_latex(range_,std=None,rstd=None, is_degree_range=False):
    assert(not ((std or rstd) and is_degree_range))
    tex = None
    if range_.is_exact():
        tex = lit(num(range_.exact)) + syn(",") + " " + lit(num(range_.exact))
    elif range_.is_upper_bound():
        tex = syn("-inf") + syn(",") + " " + lit(num(range_.max))
    elif range_.is_lower_bound():
        tex = lit(num(range_.min)) + syn(",") + " " + syn("inf")
    elif range_.is_interval_bound():
        tex = lit(num(range_.min)) + syn(",") + " " + lit(num(range_.max))
    assert(not tex is None)
    if not is_degree_range:
        tex = f'{syn("[")}{tex}{syn("]")}'

        if not std is None:
            tex = f'{tex} {syn("mismatch(")}{lit(num(std))}{syn(",")} {lit(num(0))}{syn(")")}'
        elif not rstd is None:
            tex = f'{tex} {syn("mismatch(")}{lit(num(0))}{syn(",")} {lit(num(rstd))}{syn(")")}'
    
    return tex



def attr_to_latex(attr):
    tex = f'{syn("attr")} {lit(attr.name)} {syn("=")} '
    if attr.type == FunctionType:
        tex += syn("lambd")
    elif attr.type == float:
        tex += syn("real")
    elif attr.type == int:
        tex += syn("int")
    else:
        raise Exception("unhandled attribute type <%s>" % attr.type)

    if not attr.valid_range is None:
        if isinstance(attr,AttrDefMismatch):
            assert(not attr.std is None or not attr.rstd is None)
            assert(attr.std is None or attr.rstd is None)
            tex += range_to_latex(attr.valid_range,std=attr.std,rstd=attr.rstd)
        else:
            tex += range_to_latex(attr.valid_range)

    return tex
        



def type_spec_to_latex(cdglang):
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM


    tab = LatexVerbatim()
    tab.add_verbatim_directive("fontsize=\\footnotesize")
    tab.add_verbatim_directive("fontseries=b")
    
    q = lambda x: tab.add_token(x)

    def attr_block(attrs):
        q(syn("\{"))
        for attr in attrs:
            q(attr_to_latex(attr))
            tab.delim(",")
        tab.eat_delim()
        q(syn("\}"))
        # tab.newline()

    def inherit_block(cdg_type):
        base_types = cdg_type.base_cdg_types()
        if len(base_types) > 1:
            q(syn("inherit"))
            # Only show the immediate parent type
            q(lit(base_types[1].name))
        tab.newline()

    for node in cdglang.node_types():
        node_type_def = f'{syn("node-type(")}{lit(num(node.order))}{syn(",")} {syn(node.reduction.name)}{syn(")")}'
        q(node_type_def)
        q(lit(node.name))
        attr_block(node.attr_def.values())
        inherit_block(node)
        

    for edge in cdglang.edge_types():
        q(syn("edge-type"))
        q(lit(edge.name))
        attr_block(edge.attr_def.values())
        inherit_block(edge)

    latex_text = tab.to_latex()
    print("---- type spec [%s] ----" % cdglang.name)
    print(latex_text)
    write_file(cdglang.filename("types","tex"),latex_text)



def production_rules_to_latex(cdglang):

    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.TABULAR
    varfmt = None

    tab = LatexTabular("lccl")
    global syn, lit
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    # vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)

    tab.add_preprocessing_directive("\setlength{\\tabcolsep}{1pt}")
    tab.add_preprocessing_directive("\\footnotesize")

    for rule in cdglang.production_rules():
        edge_type = rule.identifier.et.name
        src = rule.identifier.src_nt.name
        dest = rule.identifier.dst_nt.name
        tgt = rule.identifier.gen_tgt
        expr = rule.fn_sympy
        latex_expr = latex(format_variables(expr), mul_symbol="dot", fold_short_frac=True)

        if not tgt == SELF:
            second_arg_name = lit('t')
            tgt_name = lit(str(tgt)[0])
        else:
            second_arg_name = lit('s')
            tgt_name = lit('s')
        prod_head = f'{syn("prod(")}{lit("e")}{syn(":")} {lit(edge_type)}{syn(",")} '
        prod_head += f'{lit("s")}{syn(":")} {lit(src)}{syn("->")}{second_arg_name}{syn(":")} {lit(dest)}{syn(")")}'
        tab.add_cell(prod_head)
        tab.add_cell(tgt_name)


        tab.add_cell(syn("<="))
        tab.add_cell(LatexPrettyPrinter.math_formula(latex_expr))
        tab.end()

    latex_text = tab.to_latex()
    print("---- relations [%s] ----" % cdglang.name)
    print(latex_text)
    print("")
    write_file(cdglang.filename("relations","tex"),latex_text)

def validation_rules_to_latex(cdglang):

    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.TABULAR
    K = 3
    tab = LatexTabular("lll")
    global syn, lit
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    # vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)

    tab.add_preprocessing_directive("\setlength{\\tabcolsep}{1pt}")
    tab.add_preprocessing_directive("\\footnotesize")

    def add_patterns_to_cell(target_type, pats):
        # fill in empty cells to align
        for pat in pats:
            for _ in range(tab.col_count, 2):
                tab.add_cell("")
            targ = pat.target # source or destination
            edge_type = lit(pat.edge_type.name)
            node_types = f'[{(syn(",") + " ").join(list(map(lambda nt: lit(nt.name), pat.node_types)))}]'
            cell_text = syn("match(")
            deg_range = pat.deg_range
            cell_text += f'{range_to_latex(deg_range, is_degree_range=True)}{syn(",")} {edge_type}{syn(",")} '
            if targ == SRC:
                cell_text += f'{target_type}{syn("->")}{node_types}{syn(")")}'
            elif targ == DST:
                cell_text += f'{node_types}{syn("->")}{target_type}{syn(")")}'
            elif targ == SELF:
                cell_text += f'{target_type}{syn(")")}'
            tab.add_cell(cell_text)
            tab.end()

    def extern_func_to_latex(fns):
        raise NotImplementedError

    for rule in cdglang.validation_rules():
        target_type = lit(rule.tgt_node_type.name)
        tab.add_multicell(K, 'l', f'{syn("cstr")} {target_type}'+" \{")
        tab.end()
        tex_snippet = []
        if rule.acc_pats != []:
            tab.add_cell("")
            tab.add_cell(syn("acc"))
            add_patterns_to_cell(target_type, rule.acc_pats)
        if rule.rej_pats != []:
            tab.add_cell("")
            tab.add_cell(syn("rej"))
            add_patterns_to_cell(target_type, rule.rej_pats)
        if rule.checking_fns != []:
            tab.add_cell("")
            tab.add_cell(syn("extern-func"))
            extern_func_to_latex(rule.checking_fns)
        tab.add_multicell(K, 'l', "\}")
        tab.end()

    latex_text = tab.to_latex()
    print("----- validation rules [%s] ------" % (cdglang.name))
    print(latex_text)
    print("")
    write_file(cdglang.filename("validation","tex"),latex_text)

def gen_func_example(func_name, lang_name, v_node_type, i_node_type, edge_type):
    """
    func branch_tline (..., sw: int[0, 1], 
    r_v10: real[0, inf], ...) uses tln-lang {
        ...
        node n_v10: IdealV
        node n_i11: IdealI
        node n_ibr: IdealI
        edge<n_v10, n_i11> e_10_11: IdealE
        edge<n_i11, n_ibr> e_10_br: IdealE_sw
        set-attr n_v10.r = r_v10
        set-switch n_ibr when sw==1
        ...
    }
    """
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM


    tab = LatexVerbatim()
    tab.add_verbatim_directive("fontsize=\\footnotesize")
    tab.add_verbatim_directive("fontseries=b")
    
    q = lambda x: tab.add_token(x)

    sw_range = Range(0,1)

    func_def = f'{syn("func")} {lit(func_name)} {syn("(")}'
    func_def += f'{lit("sw")}{syn(":")} {syn("int")}{range_to_latex(sw_range)}{syn(")")}'
    q(func_def)
    # tab.newline()
    # func_def = f'{lit("g_v10")}{syn(":")} {syn("real")}{range_to_latex(Range(0))}{syn(",")}...{syn(")")}'
    # q(func_def)
    tab.newline()
    func_def = f'{syn("uses")} {lit(lang_name)}'
    func_def += syn("\{")
    q(func_def)
    tab.add_indent()
    tab.newline()
    q('...')
    tab.newline()
    q(f'{syn("node")} {lit("n_v10")}{syn(":")} {lit(v_node_type)}')
    tab.newline()
    q(f'{syn("node")} {lit("n_i11")}{syn(":")} {lit(i_node_type)}')
    tab.newline()
    q(f'{syn("node")} {lit("n_ibr")}{syn(":")} {lit(i_node_type)}')
    tab.newline()
    q(f'{syn("edge<")}{lit("n_v10")}{syn(",")} {lit("n_i11")}{syn(">")} {lit("e_10_11")}{syn(":")} {lit(edge_type)}')
    tab.newline()
    q(f'{syn("edge<")}{lit("n_i11")}{syn(",")} {lit("n_ibr")}{syn(">")} {lit("e_10_br")}{syn(":")} {lit(edge_type)}{lit("_sw")}')
    tab.newline()
    q(f'{syn("set-switch")} {lit("n_ibr")} {syn("when")} {expr("sw==1")}')
    # tab.newline()
    # q(f'{syn("set-attr")} {lit("n_v10.g")} {syn("=")} {lit("g_v10")}')
    tab.newline()
    q('...')
    tab.remove_indent()
    tab.newline()
    q(syn("\}"))
    tab.newline()


    latex_text = tab.to_latex()
    print(f"---- func example [{func_name}, {lang_name}] ----")
    print(latex_text)
    print("")
    write_file(f"func-example-{func_name}-{lang_name}.tex",latex_text)



if __name__ == "__main__":
    gen_func_example("br-tln", "tln-lang", "V", "I", "E")
    gen_func_example("br-tln-Nm", "hwtln-lang", "Vm", "Im", "E")
    gen_func_example("br-tln-Em", "hwtln-lang", "V", "I", "Em")


"""
func fc_osc (switch_6: int [0, 1], switch_7: int [0, 1], switch_8: int [0, 1],) uses osc-lang {
    node 0: Osc,
    node 1: Osc,
    node 2: Osc,
    edge<0, 1> 3: Coupling_sw,
    edge<1, 2> 4: Coupling_sw,
    edge<2, 0> 5: Coupling_sw,
    switch 6,
    switch 7,
    switch 8,
    set-switch 3 when sw_6==1,
    set-switch 4 when sw_7==1,
    set-switch 5 when sw_8==1
}
"""
