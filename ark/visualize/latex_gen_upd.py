from types import FunctionType
from ark.visualize.latex_util import *
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.range import Range
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from pylatexenc.latex2text import LatexNodes2Text

syn = lambda x: LatexPrettyPrinter.fmt(Terms.SYNTAX, x)
lit = lambda x: LatexPrettyPrinter.fmt(Terms.LITERAL, x)
expr = lambda x: LatexPrettyPrinter.fmt(Terms.EXPRESSION, x)
vari = lambda x: LatexPrettyPrinter.fmt(Terms.VARIABLE, x)
kw = lambda x: LatexPrettyPrinter.fmt(Terms.KEYWORD, x)
num = lambda x: LatexPrettyPrinter.fmt_number(x)
pct = lambda x: LatexPrettyPrinter.fmt_percent(x)

MAXLINEWIDTH = 25


def range_to_latex(range_,std=None,rstd=None, is_degree_range=False):
    assert(not ((std or rstd) and is_degree_range))
    tex = None
    if range_.is_exact():
        tex = lit(num(range_.exact)) + syn(",") + " " + lit(num(range_.exact))
    elif range_.is_upper_bound():
        tex = kw("-inf") + syn(",") + " " + lit(num(range_.max))
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


def attr_to_latex(attr):
    tex = f'{syn("attr")} {lit(attr.name)} {syn("=")} '
    if attr.type == FunctionType:
        tex += kw("lambd")
    elif attr.type == float:
        tex += kw("real")
    elif attr.type == int:
        tex += kw("int")
    else:
        raise Exception("unhandled attribute type <%s>" % attr.type)

    if not attr.valid_range is None:
        if isinstance(attr,AttrDefMismatch):
            assert(not attr.std is None or not attr.rstd is None)
            assert(attr.std is None or attr.rstd is None)
            tex += range_to_latex(attr.valid_range,std=attr.std,rstd=attr.rstd)
        else:
            tex += range_to_latex(attr.valid_range)

    if attr.type == FunctionType:
        tex += syn("(")
        tex += LatexPrettyPrinter.delimited(syn(","), map(lambda i: vari("a%d" % i), range(attr.nargs)))
        tex += syn(")")
    return tex
        

def language_to_latex(cdglang):
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM
    tab = LatexVerbatim(linewidth=MAXLINEWIDTH)

    q = lambda x: tab.add_token(x)
    qs = lambda x: tab.add_token(x+" ")


    tab.add_verbatim_directive("fontsize=\\footnotesize")
    tab.add_verbatim_directive("fontseries=b")
    qs(kw("lang"))
    qs(vari(cdglang.name))
    if not cdglang.inherits is None:
        qs(kw("inherits"))
        q(vari(cdglang.inherits.name))
    q(syn("{"))
    tab.indent()    
    tab.newline()

    type_spec_to_latex(tab,cdglang)
    production_rules_to_latex(tab,cdglang)
    validation_rules_to_latex(tab,cdglang)
    tab.unindent()
    tab.linebreak(" ")    
    q(syn("}"))
    tab.newline()

    latex_text = tab.to_latex()
    print("---- lang [%s] ----" % cdglang.name)
    txt = LatexNodes2Text().latex_to_text(latex_text)
    print(txt)
    
    write_file(cdglang.filename("language","tex"),latex_text)



def type_spec_to_latex(tab,cdglang):
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM

    q = lambda x: tab.add_token(x)
    qs = lambda x: tab.add_token(x+" ")

    def attr_block(attrs):
        q(syn("{"))
        tab.indent()
        for idx,attr in enumerate(attrs):
            q(attr_to_latex(attr))
            if idx < len(attrs) - 1:
                q(syn(","))
            tab.linebreak()
        tab.unindent()
        q(syn("}"))

    def inherit_block(cdg_type):
        base_types = cdg_type.base_cdg_types()
        if len(base_types) > 1:
            q(syn("inherit"))
            # Only show the immediate parent type
            q(vari(base_types[1].name))

    for node in cdglang.node_types(inherited=False):
        q(kw("node-type"))
        q(syn("("))
        q(lit(num(node.order)))
        q(syn(","))
        q(syn(node.reduction.name))
        qs(syn(")"))
        q(vari(node.name))
        inherit_block(node)
        tab.linebreak(" ")
        attr_block(node.attr_def.values())
        q(syn(";"))
        tab.newline()
        

    for edge in cdglang.edge_types(inherited=False):
        qs(syn("edge-type"))
        q(vari(edge.name))
        inherit_block(edge)
        tab.linebreak(" ")
        if len(list(edge.attr_def.values())) == 0:
            q("{}")
        else:
            attr_block(edge.attr_def.values())
        q(syn(";"))
        tab.newline()

    
def production_rules_to_latex(tab,cdglang):
    q = lambda x: tab.add_token(x)
    qs = lambda x: tab.add_token(x+" ")

    for rule in cdglang.production_rules():
        dest = rule.identifier.dst_nt.name
        tgt = rule.identifier.gen_tgt
        expr = rule.fn_sympy
        latex_expr = latex(format_variables(expr), mul_symbol="dot", fold_short_frac=True)

        if not tgt == SELF:
            tgt_name = str(tgt)[0]
        else:
            tgt_name = 's'

        q(syn("prod("))
        q(lit("e"))
        q(syn(":"))
        q(lit(rule.identifier.et.name))
        qs(",")
        q(lit("s"))
        q(syn(":"))
        q(lit(rule.identifier.src_nt.name))
        if not rule.identifier.gen_tgt == SELF:
            qs(syn(","))
            qs(syn("->"))
            q(lit("t"))
            q(syn(":"))
            q(lit(rule.identifier.dst_nt.name))
        q(syn(")"))
        tab.indent() 
        tab.linebreak(" ")
        q(syn(tgt_name))
        q(syn("<="))
        q(LatexPrettyPrinter.math_formula(latex_expr))
        q(syn(";"))
        tab.unindent()
        tab.newline()


def validation_rules_to_latex(tab,cdglang):
    q = lambda x: tab.add_token(x)
    qs = lambda x: tab.add_token(x+" ")


    def add_patterns_to_verbatim(target_type, pats):
        # fill in empty cells to align
        tab.indent()
        tab.linebreak()
        for idx,pat in enumerate(pats):
            targ = pat.target # source or destination
            deg_range = pat.deg_range
            edge_type = pat.edge_type.name
            node_types = LatexPrettyPrinter.delimited(syn(","), map(lambda nt: vari(nt.name), pat.node_types))
            q(kw("match"))
            q(syn("("))
            q(range_to_latex(deg_range, is_degree_range=True))
            q(syn(","))
            q(lit(edge_type))
            q(syn(","))
            if targ == SRC:
                q(vari(target_type))
                q(syn("->"))
                q(syn("["))
                q(node_types)
                q(syn("]"))
            elif targ == DST:
                q(syn("["))
                q(node_types)
                q(syn("]"))
                q(syn("->"))
                q(vari(target_type))
            elif targ == SELF:
                q(vari(target_type))
            q(syn(")"))
            if idx < len(pats) - 1:
                q(syn(","))
            tab.linebreak()

        tab.unindent()

    def extern_func_to_latex(fns):
        raise NotImplementedError

    for rule in cdglang.validation_rules():
        target_type = rule.tgt_node_type.name
        qs(kw("cstr"))
        qs(lit(target_type))
        q(syn("{"))
        tab.indent()
        tab.linebreak()
        tex_snippet = []
        if rule.acc_pats != []:
            q(kw("acc"))
            q(syn("["))
            add_patterns_to_verbatim(target_type, rule.acc_pats)
            q(syn("]"))
        if rule.rej_pats != []:
            q(kw("rej"))
            q(syn("["))
            add_patterns_to_verbatim(target_type, rule.rej_pats)
            q(syn("]"))
        if rule.checking_fns != []:
            q(syn("extern-func"))
            extern_func_to_latex(rule.checking_fns)
        tab.unindent()
        q(syn("}"))
        tab.newline()
 
@dataclass
class SwitchArg:
    edge : str;
    expr : str;

def gen_function(fname,lang,cdg,args,first_k=None):
    LatexPrettyPrinter.ESCAPE_STYLE = EscapeStyle.VERBATIM
    tab = LatexVerbatim(linewidth=MAXLINEWIDTH)

    q = lambda x: tab.add_token(x)
    qs = lambda x: tab.add_token(x+" ")

    def gen_attrs(n,a,v):
        val = num(v)
        qs(kw("set-attr"))
        q(vari(n.name))
        q(syn("."))
        q(lit(a))
        q(syn("="))
        q(val)
        q(syn(";"))
        tab.linebreak(" ")
 
    def ellipses():
        tab.newline()
        q(syn("..."))
        tab.newline()

    tab.add_verbatim_directive("fontsize=\\footnotesize")
    tab.add_verbatim_directive("fontseries=b")
    qs(kw("func"))
    qs(vari(fname))
    qs(syn("("))
    n_args = len(list(args.values()))

    edge_map = {}
    for idx,(name,arg) in enumerate(args.items()):
        q(vari(name))
        q(syn(":"))
        if isinstance(arg, SwitchArg):
            edge_map[arg.edge] = name
            q(lit("int"))
            q(syn("[0,1]"))
        if idx < n_args-1:
            q(syn(","))

    qs(syn(")"))
    qs(kw("uses"))
    q(vari(lang.name))
    q(syn("{"))
    tab.indent()    
    tab.newline()


    n_nodes = 0
    for n in cdg.nodes:
        qs(kw("node"))
        q(vari(n.name))
        q(syn(":"))
        q(lit(n.cdg_type.name))
        q(syn(";"))
        tab.linebreak(" ")
        n_nodes += 1
        if not first_k is None and n_nodes >= first_k:
            ellipses() 
            break
                   
    tab.newline()

    n_edges = 0
    for e in cdg.edges:
        qs(kw("edge"))
        q(syn("<"))
        q(vari(e.src.name))
        qs(syn(","))
        q(vari(e.dst.name))
        q(syn(">"))
        q(vari(e.name))
        q(syn(":"))
        q(lit(e.cdg_type.name))
        q(syn(";"))
        tab.linebreak(" ")
        n_edges += 1
        if not first_k is None and n_edges >= first_k:
            ellipses() 
            break
           
        
    for edge in edge_map:
        sw = edge_map[edge]
        qs(kw("set-switch"))
        qs(vari(edge))
        qs(kw("when"))
        q(expr(sw))
        q(syn(";"))
        tab.linebreak(" ")
            
    n_attrs = 0
    stop = False
    for n in cdg.nodes:
        for a,v in n.attrs.items():
            if stop:
                continue

            gen_attrs(n,a,v)
            n_attrs += 1
            stop = not first_k is None and n_attrs>= first_k
            if stop:
                ellipses() 
                break 

    n_attrs = 0
    for e in cdg.edges:
        for a,v in n.attrs.items():
            if stop:
                continue

            gen_attrs(e,a,v)
            n_attrs += 1
            stop = not first_k is None and n_attrs>= first_k
            if stop:
                ellipses() 
                break
       


    tab.newline()

    tab.unindent()
    tab.linebreak(" ")    
    q(syn("}"))
    tab.newline()

    latex_text = tab.to_latex()
    print("---- function [%s] ----" % fname)
    txt = LatexNodes2Text().latex_to_text(latex_text)
    write_file(lang.filename(fname,"tex"),latex_text)

