from enum import Enum
import re
from sympy import latex
import sympy
import os


class Terms(Enum):
    LITERAL = "literal"
    VARIABLE = "variable"
    HIGHLIGHT = "highlight"
    SYNTAX = "syntax"

class TextStyle(Enum):
    BOLD = "bold"

    def formatted(self,text):
        if self == TextStyle.BOLD:
            return "\\textbf{" + text + "}"
        else:
            raise NotImplementedError
            
class EscapeStyle(Enum):
    NONE = "none"
    VERBATIM = "verbatim"
    TABULAR = "tabular"
    def default_get_escape_characters(self):
        return {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
            '<': r'\textless{}',
            '>': r'\textgreater{}',
        }

    def get_escape_characters(self):
        if self == EscapeStyle.TABULAR:
            dflt = dict(self.default_get_escape_characters())
            dflt["\\"] = "\\"
            return dflt
        else:
            return self.default_get_escape_characters()


class LatexPrettyPrinter:
    COLORS = {}
    STYLES = {}

    ESCAPE_STYLE = EscapeStyle.NONE

    @classmethod
    def set_color(cls,enum,color):
        LatexPrettyPrinter.COLORS[enum] = color

    @classmethod
    def set_style(cls,enum,style):
        assert(isinstance(style,TextStyle))
        LatexPrettyPrinter.STYLES[enum] = style


    @classmethod
    def esc(cls,text):
        conv = cls.ESCAPE_STYLE.get_escape_characters()
        regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
        result = regex.sub(lambda match: conv[match.group()], text)
        return result 

    @classmethod
    def fmt(cls,type_,text_):
        text = LatexPrettyPrinter.esc(text_)
        if type_ in LatexPrettyPrinter.STYLES:
            text = LatexPrettyPrinter.STYLES[type_].formatted(text_)

        if type_ in LatexPrettyPrinter.COLORS:
            color = LatexPrettyPrinter.COLORS[type_]
            text = "\\textcolor{%s}{%s}" % (color,text)

        return text

    @classmethod
    def fmt_code(cls,type_,text_):
        tex = LatexPrettyPrinter.fmt(type_,text_)
        return "\\textbf{\\texttt{%s}}" % tex

    @classmethod
    def math_formula(cls,expr):
        if isinstance(expr,sympy.Expr):
            latex_formula = latex(expr)
        else:
            latex_formula = str(expr)

        if LatexPrettyPrinter.ESCAPE_STYLE == EscapeStyle.NONE or \
            LatexPrettyPrinter.ESCAPE_STYLE == EscapeStyle.TABULAR:
            return "$%s$" % latex_formula
        else:
            raise NotImplementedError

class LatexTabular:

    def __init__(self,align):
        self.ctx = []
        self.rowbuf = []
        self.col_count = 0

        self.preproc = []
        self.ncols = len(align) 
        self.align = align
        assert(len(self.align) == self.ncols)

    def add_preprocessing_directive(self,direc):
        self.preproc.append(direc)

    def add_cell(self,text):
        self.rowbuf.append(text)
        self.col_count += 1 
        assert(len(self.rowbuf) <= self.ncols)
        return self

    def add_multicell(self,k,a,text):
        text = "\\multicolumn{%d}{%s}{%s}" % (k,a,text)
        self.rowbuf.append(text)
        self.col_count += k 
        assert(self.rowbuf <= self.ncols)
        return text


    def end(self):
        assert(len(self.rowbuf) == self.ncols)
        self.ctx.append(self.rowbuf)
        self.rowbuf  = []
        self.col_count = 0

    def to_latex(self):
        stmts = []
        def q(txt):
            stmts.append(txt)
        
        def qrow(els):
            q(" & ".join(els)+"\\\\")
        q("{")
        for pre in self.preproc:
            q(pre)
        q("\\begin{tabular}{%s}" % self.align)
        for row in self.ctx:
            qrow(row)
        q("\\end{tabular}")
        q("}")
        return "\n".join(stmts)

class LatexVerbatim:

    def __init__(self):
        self.ctx = []
        self.row = []
        self._insert_comma = False
        self.preproc = []
        self.directives = []
        self.do_gobble = False
        self.add_verbatim_directive("commandchars=\\\\\{\}")
        self.add_verbatim_directive("codes={\catcode`$=3\catcode`_=8}")


    def add_verbatim_directive(self,direc):
        self.directives.append(direc)


    def add_preprocessing_directive(self,direc):
        self.preproc.append(direc)

    def gobble(self):
        self.do_gobble = True

    def add_token(self,text):
        assert(isinstance(text,str))
        if self._insert_comma:
            self.row.append(",")
            self._insert_comma = False 

        if not self.do_gobble:
            self.row.append(text)
        else:
            self.row[-1] += text
            self.do_gobble = False
        return self
     
    def eat_comma(self):
        self._insert_comma = False


    def comma(self):
        self._insert_comma = True

    def newline(self):
        self.ctx.append(" ".join(self.row)+"\n")
        self.row = []

    def to_latex(self):
        stmts = []
        def q(txt):
            stmts.append(txt)

        incantation = ",".join(self.directives) 

        #"codes={\catcode‘$=3\catcode‘^=7"
        q("{")
        for pre in self.preproc:
            q(pre)
        q("\\begin{Verbatim}[%s]\n" % incantation)
        for st in self.ctx:
            q(st)
        q("\\end{Verbatim}\n")
        q("}")

        return "".join(stmts)



LatexPrettyPrinter.set_color(Terms.LITERAL, "deepred")
LatexPrettyPrinter.set_color(Terms.VARIABLE, "deepblue")
LatexPrettyPrinter.set_color(Terms.SYNTAX, "black")
LatexPrettyPrinter.set_style(Terms.SYNTAX, TextStyle.BOLD)

def write_file(filename,text):
    direc = "tex-outputs"
    if not os.path.exists(direc):
        os.mkdir(direc)

    path = "%s/%s" % (direc,filename)
    with open(path,"w") as fh:
        fh.write(text)

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
    print(latex_text)
    input()
    write_file(cdglang.filename("types","tex"),latex_text)

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

    write_file(cdglang.filename("relations","tex"),tab.to_latex())