from ark.visualize.latex_util import *

def special_variable(name):
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)

    if name == "TIME":
        tok = lit("t")
    elif name == "DST":
        tok = vari("dst")
    elif name == "SRC":
        tok = vari("src")
    elif name == "EDGE":
        tok = vari("edge")
    else:
        tok = vari(name)

    return tok

def format_variables(expr):
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)

    print(expr)
    fn_subdict = {}
    for f in expr.atoms(sympy.Function):
        name = f.func.name
        assert("." in name)
        caller,fname = name.split(".")
        tok = special_variable(caller)
        tok += syn(".")
        tok += lit(fname)
        
        fn_subdict[f]  = sympy.Function(tok,real=True)(*f.args)
        

    fn_expr = expr.subs(fn_subdict)

    subdict = {}
    for sym in fn_expr.free_symbols:
        text = sym.name
        if "." in text:
            name,attr = text.split(".")
            tok = vari(name)
            tok += syn(".")
            tok += lit(attr)
        
        else:
            tok = special_variable(sym.name)

        subdict[sym] = sympy.Symbol(tok)
       
    for f in fn_expr.atoms(sympy.Float):
        fmt = "\\num{{{0:.2g}}}".format(f)
        subdict[f] = sympy.Symbol(fmt)



    sub_expr = fn_expr.subs(subdict)
    return sub_expr


def type_spec_to_latex(cdglang):
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM

    tab = LatexVerbatim()
    tab.add_verbatim_directive("fontsize=\\footnotesize")
    tab.add_verbatim_directive("fontseries=b")
    syn = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.SYNTAX, x))
    lit = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.LITERAL, x))
    vari = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.VARIABLE, x))

    def attr_to_latex(attr):
        syn("attr")
        vari(attr.name)
        syn(":")
        if callable(attr.type):
            lit("func")
        elif isinstance(attr.type, float):
            lit("real")
        else:
            raise Exception("unhandled attribute type <%s>" % attr.type)

        assert(attr.valid_range is None)

    def attr_block(attrs):
        syn("{")
        tab.newline()
        for attr in attrs:
            attr_to_latex(attr)
            tab.comma()
        tab.eat_comma()
        syn("}")
        tab.newline()

    for node in cdglang.node_types():
        syn("node-type")
        vari(node.name)
        syn("order(").gobble()
        vari(str(node.order)).gobble()
        syn(")")
        syn("reduce(").gobble()
        vari(node.reduction.name).gobble()
        syn(")")
        attr_block(node.attr_def.values())
        

    for edge in cdglang.edge_types():
        syn("edge-type")
        vari(edge.name)
        attr_block(edge.attr_def.values())

    latex_text = tab.to_latex()
    print("---- type spec [%s] ----" % cdglang.name)
    print(latex_text)
    write_file(cdglang.filename("types","tex"),latex_text)

def production_rules_to_latex(cdglang):
    def src_is_target(prodrule):
        return prodrule.identifier.gen_tgt.name == "SRC"

    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.TABULAR
    varfmt = None

    tab = LatexTabular("rrclllll")
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)

    tab.add_preprocessing_directive("\setlength{\\tabcolsep}{1pt}")
    tab.add_preprocessing_directive("\\footnotesize")

    by_target = {}
    for prodrule in cdglang.production_rules():
        targ_node = prodrule.identifier.src_nt if src_is_target(prodrule) else prodrule.identifier.dst_nt
        if not targ_node in by_target:
            by_target[targ_node] = []
        by_target[targ_node].append(prodrule)

    for targ_node,rules in by_target.items():
        for rule in rules:
            edge_type = rule.identifier.et.name
            src = rule.identifier.src_nt.name
            dest = rule.identifier.dst_nt.name

            if src_is_target(rule):
                target = "src"
            else:
                target = "dst"

            expr = rule.fn_sympy
            latex_expr = latex(format_variables(expr))
            tab.add_cell(syn("rule(")+vari("edge")+syn(":")+lit(edge_type))
            tab.add_cell(vari("src")+syn(":")+lit(src))
            tab.add_cell(LatexPrettyPrinter.math_formula("\\rightarrow"))
            tab.add_cell(vari("dst")+syn(":")+lit(dest)+syn(")"))
            tab.add_cell(syn("produces"))
            tab.add_cell(lit(target))
            tab.add_cell(syn("<="))
            tab.add_cell(LatexPrettyPrinter.math_formula(latex_expr))
            tab.end()

    latex_text = tab.to_latex()
    print("---- relations [%s] ----" % cdglang.name)
    print(latex_text)
    write_file(cdglang.filename("relations","tex"),latex_text)

def validation_rules_to_latex(cdglang):
    raise Exception("not implemented")