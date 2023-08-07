from enum import Enum
import re
from sympy import latex
import sympy
import os
from pylatexenc.latex2text import LatexNodes2Text
from dataclasses import dataclass


class Terms(Enum):
    LITERAL = "literal"
    VARIABLE = "variable"
    HIGHLIGHT = "highlight"
    SYNTAX = "syntax"
    EXPRESSION = "expression"
    KEYWORD = "keyword"

class TextStyle(Enum):
    BOLD = "bold"

    def formatted(self,text):
        if self == TextStyle.BOLD:
            return "\\textbf{" + text+ "}"
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
    def delimited(cls,delim,arggen):
        args = list(arggen)
        tex = ""
        for i in range(len(args)-1):
            tex += args[i]
            tex += delim
        tex += args[len(args)-1]
        return tex


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
        esc = text = LatexPrettyPrinter.esc(text_)
        if type_ in LatexPrettyPrinter.STYLES:
            text = LatexPrettyPrinter.STYLES[type_].formatted(text)

        print(esc,text)
        if type_ in LatexPrettyPrinter.COLORS:
            color = LatexPrettyPrinter.COLORS[type_]
            tex = "\\textcolor{%s}" % (color)
            tex += "{"+text+"}"
            text = tex
        print(esc,text)

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
            LatexPrettyPrinter.ESCAPE_STYLE == EscapeStyle.TABULAR or \
            LatexPrettyPrinter.ESCAPE_STYLE == EscapeStyle.VERBATIM:
            return "$%s$" % latex_formula
        else:
            raise  Exception("unknown escape style %s" % LatexPrettyPrinter.ESCAPE_STYLE)

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
        assert(self.col_count <= self.ncols)
        return self

    def add_multicell(self,k,a,text):
        text = "\\multicolumn{%d}{%s}{%s}" % (k,a,text)
        self.rowbuf.append(text)
        self.col_count += k 
        assert(self.col_count <= self.ncols)
        return text


    def fill_and_end(self):
        n_empty = self.ncols - self.col_count
        for _ in range(n_empty):
            self.add_cell("")
        self.end()

    def end(self):
        assert(self.col_count == self.ncols)
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

    @dataclass
    class LineBreak:
        indent_style : str
        n_indent : int
        pad : str
        force: bool


    def __init__(self,linewidth=None):
        self.ctx = []
        self.row = []
        self._delim = None
        self.preproc = []
        self.directives = []
        self.do_gobble = False
        self.indent_style = "  "
        self.n_indent = 0
        self.linewidth = linewidth 
        self.add_verbatim_directive("commandchars=\\\\\{\}")
        self.add_verbatim_directive("codes={\catcode`$=3\catcode`_=8}")

  
    def eat_delim(self):
        self._delim = None


    def delim(self,txt):
        self._delim = txt

    def add_verbatim_directive(self,direc):
        self.directives.append(direc)


    def add_preprocessing_directive(self,direc):
        self.preproc.append(direc)

    def gobble(self):
        self.do_gobble = True

    def add_token(self,text):
        print(text)
        assert(isinstance(text,str))
        if not self._delim is None:
            self.ctx.append(self._delim)
            self._delim = None

        if not self.do_gobble:
            self.ctx.append(text)
        else:
            self.ctx[-1] += text
            self.do_gobble = False
        return self
    
    def add_space(self):
        self.add_token(" ")

    def indent(self):
        self.n_indent += 1

    def unindent(self):
        self.n_indent -= 1

    def linebreak(self,pad=""):
        lb = LatexVerbatim.LineBreak(indent_style=self.indent_style,n_indent=self.n_indent,pad=pad, force=False)
        self.ctx.append(lb)

    def newline(self):
        lb = LatexVerbatim.LineBreak(indent_style=self.indent_style,n_indent=self.n_indent,pad="",force=True)
        self.ctx.append(lb)

    def to_latex(self):
        stmts = []
        def q(txt):
            assert(isinstance(txt,str))
            stmts.append(txt)
        
        lb_ctx = []
        buf = []
        nchars = 0
        for tok in self.ctx:
            if isinstance(tok, LatexVerbatim.LineBreak):
                lb_ctx.append((nchars,tok,buf))
                buf = []
                nchars = 0
            else:
                txt = LatexNodes2Text().latex_to_text(tok)
                nchars += len(txt)
                buf.append(tok)

        if not(len(buf) == 0):
            raise Exception("must end verbatim in a linebreak or newline")

        incantation = ",".join(self.directives)


        #"codes={\catcode‘$=3\catcode‘^=7"
        q("{")
        q("\n")
        for pre in self.preproc:
            q(pre)
            q("\n")

        q("\\begin{Verbatim}[%s]\n" % incantation)
        lw = 0
        for idx,(width,lb,buf) in enumerate(lb_ctx):
            for b in buf:
                print(b)
                q(b)

            if self.linewidth is None \
                or idx == len(lb_ctx)-1 \
                or lb_ctx[idx+1][0]+width+lw >= self.linewidth \
                or lb.force:
                q("\n")
                q(lb.indent_style*lb.n_indent)
                lw = 0
            else:
                q(lb.pad)
            lw += width

        q("\\end{Verbatim}")
        q("\n")
        q("}")


        return "".join(stmts)




LatexPrettyPrinter.set_color(Terms.LITERAL, "burntorange")
LatexPrettyPrinter.set_color(Terms.VARIABLE, "deepblue")
LatexPrettyPrinter.set_color(Terms.KEYWORD, "ultraviolet")
LatexPrettyPrinter.set_color(Terms.SYNTAX, "black")
LatexPrettyPrinter.set_color(Terms.EXPRESSION, "green")
LatexPrettyPrinter.set_style(Terms.SYNTAX, TextStyle.BOLD)
LatexPrettyPrinter.set_style(Terms.KEYWORD, TextStyle.BOLD)

def write_file(filename,text):
    direc = "tex-outputs"
    if not os.path.exists(direc):
        os.mkdir(direc)

    path = "%s/%s" % (direc,filename)
    with open(path,"w") as fh:
        fh.write(text)
