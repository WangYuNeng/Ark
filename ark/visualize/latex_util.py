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
    def fmt_integer(cls,integer):
        return cls.fmt(None,str(integer))
       
    @classmethod
    def fmt_decimal(cls,decimal):
        return cls.fmt(None,str(decimal))
  
    @classmethod
    def fmt_number(cls,number):
        if isinstance(number, int):
            return cls.fmt_integer(number)
        else:
            return cls.fmt_decimal(number)

    @classmethod
    def fmt_percent(cls, frac):
        x = cls.fmt_number(frac*100.0)
        x += cls.fmt(None,"%")
        return x

    @classmethod
    def fmt(cls,type_,text_):
        assert(isinstance(text_,str)) 
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


    def fill_and_end(self):
        n_empty = self.ncols - len(self.rowbuf)
        for _ in range(n_empty):
            self.add_cell("")
        self.end()

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
