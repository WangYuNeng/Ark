from enum import Enum
import re

class Terms(Enum):
    LITERAL = "literal"
    VARIABLE = "variable"
    HIGHLIGHT = "highlight"
    SYNTAX = "syntax"

class TextStyle(Enum):
    BOLD = "bold"

    def formatted(self,text):
        if self == TextStyle.BOLD:
            templ = "\\textbf{%s}"
        else:
            raise NotImplementedError
            
        return templ % str(text)

class EscapeStyle(Enum):
    NONE = "none"
    VERBATIM = "verbatim"

    def get_escape_characters(self):
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

class LatexPrettyPrinter:
    COLORS = {}
    STYLES = {}

    ESCAPE_STYLE = EscapeStyle.NONE

    @classmethod
    def set_color(cls,enum,color):
        LatexPrettyPrinter.COLORS[enum] = color

    @classmethod
    def esc(cls,text):
        conv = cls.ESCAPE_STYLE.get_escape_characters()
        regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
        return regex.sub(lambda match: conv[match.group()], text)

    @classmethod
    def fmt(cls,type_,text_):
        text = LatexPrettyPrinter.esc(text_)
        if type_ in LatexPrettyPrinter.STYLES:
            text = LatexPrettyPrinter.STYLES[type_].formatted(type_)

        if type_ in LatexPrettyPrinter.COLORS:
            color = LatexPrettyPrinter.COLORS[type_]
            text = "\\textcolor{%s}{%s}" % (color,text)

        return text

    @classmethod
    def math_formula(cls,expr):
        latex_formula = latex(expr)
        if LatexPrettyPrinter.ESCAPE_STYLE == EscapeStyle.NONE:
            return "$%s$" % latex_formula
        else:
            raise NotImplementedError

class LatexTabular:

    def __init__(self,ncols,align):
        self.ctx = []
        self.rowbuf = []
        self.col_count = 0

        self.ncols = ncols
        self.align = align
        assert(len(self.align) == ncols)

    def add_cell(self,text):
        self.rowbuf.append(text)
        self.col_count += 1 
        assert(self.rowbuf <= self.ncols)
        return self

    def add_multicell(self,k,a,text):
        text = "\\multicolumn{%d}{%s}{%s}" % (k,a,text)
        self.rowbuf.append(text)
        self.col_count += k 
        assert(self.rowbuf <= self.ncols)
        return text


    def end(self):
        assert(self.rowbuf == self.ncols)
        self.ctx.append(self.rowbuf)
        self.rowbuf  = []
        self.col_count = 0

    def to_latex(self):
        stmts = []
        def q(txt):
            stmts.append(txt)
        
        def qrow(els):
            q(" & ".join(els)+"\\")

        q("\\begin\{tabular\}\{%s\}" % align)
        for row in ctx:
            qrow(row)
        q("\\end\{tabular\}")
        return "\n".join(stmts)

class LatexVerbatim:

    def __init__(self):
        self.ctx = []
        self.row = []
        self._insert_comma = False

    def add_token(self,text):
        assert(isinstance(text,str))
        if self._insert_comma:
            self.row.append(",")
            self._insert_comma = False 

        self.row.append(text)
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
        
        incantation = "commandchars=\\\\\{\}"
        #"codes={\catcode‘$=3\catcode‘^=7"
        q("\\begin{Verbatim}[%s]\n" % incantation)
        for st in self.ctx:
            q(st)
        q("\\end{Verbatim}\n")

        return "".join(stmts)



LatexPrettyPrinter.set_color(Terms.LITERAL, "cyan")
LatexPrettyPrinter.set_color(Terms.VARIABLE, "grey")
LatexPrettyPrinter.set_color(Terms.SYNTAX, "black")

def type_spec_to_latex(cdglang):
    tab = LatexVerbatim()
    syn = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.SYNTAX, x))
    lit = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.LITERAL, x))
    vari = lambda x: tab.add_token(LatexPrettyPrinter.fmt(Terms.VARIABLE, x))

    def attr_to_latex(attr):
        lit("attr")
        vari(attr.name)

    for node in cdglang.node_types():
        lit("node-type")
        vari(node.name)
        lit("order")
        syn("=")
        vari(str(node.order))
        lit("reduction")
        syn("=")
        vari(node.reduction.name)
        syn("{")
        tab.newline()
        for attr in node.attr_def.values():
            attr_to_latex(attr)
            tab.comma()
        tab.eat_comma()
        syn("}")
        tab.newline()
        

    print(tab.to_latex())
    input("continue")

def production_rules_to_latex(cdglang):
    tab = LatexTabular(3,"rcl")
    