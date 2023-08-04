from ark.visualize.latex_util import *
from ark.specification.attribute_def import AttrDef, AttrDefMismatch

syn = lambda x: LatexPrettyPrinter.fmt(Terms.SYNTAX, x)
lit = lambda x: LatexPrettyPrinter.fmt(Terms.LITERAL, x)
vari = lambda x: LatexPrettyPrinter.fmt(Terms.VARIABLE, x)
num = lambda x: LatexPrettyPrinter.fmt_number(x)
pct = lambda x: LatexPrettyPrinter.fmt_percent(x)

def special_variable(name):
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
        subdict[f] = sympy.Symbol(num(f))



    sub_expr = fn_expr.subs(subdict)
    return sub_expr

def range_to_latex(range_,std=None,rstd=None):
    tex = None
    if range_.is_exact():
        tex = lit(num(range_.exact))
    elif range_.is_upper_bound():
        tex = syn("[") + syn("*") + syn(",")+lit(num(range_.max)) + syn("]")
    elif range_.is_lower_bound():
        tex = syn("[") +lit(num(range_.min)) + syn(",") + syn("*")+ syn("]")
    elif range_.is_interval_bound():
        tex = syn("[") +lit(num(range_.min)) + syn(",") + lit(num(range_.max))+ syn("]")
    assert(not tex is None)

    if not std is None:
        tex = syn("N(") + tex + syn(",") + num(std) + ")"
    elif not rstd is None:
        tex = syn("N(") + tex + syn(",") + pct(rstd) + syn("%") + syn(")")
    
    return tex



def attr_to_latex(attr):
    tex = syn("attr")
    tex += vari(attr.name)
    tex += syn(":")
    if callable(attr.type):
        tex += lit("func")
    elif isinstance(attr.type, float):
        tex += lit("real")
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
        q(syn("{"))
        tab.newline()
        for attr in attrs:
            q(attr_to_latex(attr))
            tab.comma()
        tab.eat_comma()
        q(syn("}"))
        tab.newline()

    for node in cdglang.node_types():
        q(syn("node-type"))
        q(vari(node.name))
        q(syn("order(")).gobble()
        q(vari(str(node.order))).gobble()
        q(syn(")"))
        q(syn("reduce(")).gobble()
        q(vari(node.reduction.name)).gobble()
        q(syn(")"))
        attr_block(node.attr_def.values())
        

    for edge in cdglang.edge_types():
        q(syn("edge-type"))
        q(vari(edge.name))
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
    print("")
    write_file(cdglang.filename("relations","tex"),latex_text)

def validation_rules_to_latex(cdglang):

    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.TABULAR
    K = 3
    tab = LatexTabular("rcl")
    syn = lambda x: LatexPrettyPrinter.fmt_code(Terms.SYNTAX, x)
    lit = lambda x: LatexPrettyPrinter.fmt_code(Terms.LITERAL, x)
    vari = lambda x: LatexPrettyPrinter.fmt_code(Terms.VARIABLE, x)


    for rule in cdglang.validation_rules():
        target_type = rule.tgt_node_type.name
        tab.add_cell(syn("rule(targ:")+vari(target_type)+syn(")"))
        tab.add_cell("=")
        tex_snippet = []
        for pat in rule.acc_pats:
            targ = pat.target # source or destination
            edge_type = lit(pat.edge_type.name)
            node_types = syn("|").join(list(map(lambda nt: lit(nt.name), pat.node_types)))
            deg_range = pat.deg_range
            if targ.is_source():
                src = vari("targ")
                dest = vari("node") + syn(":") + node_types
            elif targ.is_dest():
                src = vari("node") + syn(":") + node_types
                dest = vari("targ")

            base_args = [edge_type,src+syn("=>")+dest]
            if deg_range.is_exact():
                args = [syn(str(deg_range.exact))] + base_args
                name = "exactly" 
            elif deg_range.is_lower_bound():
                args = [syn(str(deg_range.min))] + base_args
                name = "at-least"
            elif deg_range.is_upper_bound():
                args = [syn(str(deg_range.max))] + base_args
                name = "at-most"
            elif deg_range.is_interval_bound():
                args = [syn(str(deg_range.min)),syn(str(deg_range.max))] + base_args
                name = "between"

            
            tex = syn(name)+syn("(")+syn(",").join(args)+syn(")")
            tex_snippet.append(tex)

        for i,tex in enumerate(tex_snippet):
            if i != 0:
                tab.add_cell("")
                tab.add_cell("")
            tab.add_cell(tex)
            tab.end()

        # if len(tex_snippet) % K != 0: 
        #     tab.fill_and_end()
        print("")

    latex_text = tab.to_latex()
    print("----- validation rules [%s] ------" % (cdglang.name))
    print(latex_text)
    print("")
    write_file(cdglang.filename("validation","tex"),latex_text)