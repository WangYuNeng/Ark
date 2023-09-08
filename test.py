from ark.specification.specification import *
from ark.cdg.cdg_lang import *
from ark.cdg.cdg import *
from examples.cnn import *
# Example input function
def pulse(t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1):
    """Trapezoidal pulse function"""
    t = (t - delay) % period
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0

def make_cnn_lang():
    # Ark specification
    lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
    w_range = Range(min=0.5, max=2)

    tln_lang = CDGLang("tln")

    # Ideal implementation
    # Parallel capacitor(c=capacitance) and resistor(g=conductance)
    IdealV = NodeType(name='IdealV', order=1,
                    reduction=SUM,
                    attr_def=[AttrDef('c', attr_type=float, attr_range=lc_range),
                            AttrDef('g', attr_type=float, attr_range=gr_range)
                            ])
    # Series inductor(l=inductance) and resistor(r=resistance)
    IdealI = NodeType(name='IdealI', order=1,
                    reduction=SUM,
                    attr_def=[AttrDef('l', attr_type=float, attr_range=lc_range),
                            AttrDef('r', attr_type=float, attr_range=gr_range)
                            ])

    IdealE = EdgeType(name='IdealE',
                    attr_def=[AttrDef('ws', attr_type=float,attr_range=w_range),
                            AttrDef('wt', attr_type=float,attr_range=w_range)
                            ])

    # Voltage source in Thevenin equivalent
    InpV = NodeType(name='InpV',
                    order=0,
                    attr_def=[AttrDef('fn', attr_type=FunctionType,nargs=1),
                        AttrDef('r', attr_type=float, attr_range=gr_range)
                        ])

    # Current source in Thevenin equivalent
    InpI = NodeType(name='InpI',
                    order=0,
                    attr_def=[AttrDef('fn', attr_type=FunctionType,nargs=1),
                            AttrDef('g', attr_type=float, attr_range=gr_range)
                            ])
    tln_lang.add_types(IdealV, IdealI, IdealE, InpV, InpI)




    # Production rules
    _v2i = ProdRule(IdealE, IdealV, IdealI, SRC, -VAR(DST)/SRC.c)
    v2_i = ProdRule(IdealE, IdealV, IdealI, DST, VAR(SRC)/DST.l)
    _i2v = ProdRule(IdealE, IdealI, IdealV, SRC, -VAR(DST)/SRC.l)
    i2_v = ProdRule(IdealE, IdealI, IdealV, DST, VAR(SRC)/DST.c)
    vself = ProdRule(IdealE, IdealV, IdealV, SELF, -VAR(SRC)*SRC.g/SRC.c)
    iself = ProdRule(IdealE, IdealI, IdealI, SELF, -VAR(SRC)*SRC.r/SRC.l)
    inpv2_v = ProdRule(IdealE, InpV, IdealV, DST, (SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
    inpv2_i = ProdRule(IdealE, InpV, IdealI, DST, (SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
    inpi2_v = ProdRule(IdealE, InpI, IdealV, DST, (SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
    inpi2_i = ProdRule(IdealE, InpI, IdealI, DST, (SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
    prod_rules = [_v2i, v2_i, _i2v, i2_v, vself, iself, inpv2_v, inpv2_i, inpi2_v, inpi2_i]

    tln_lang.add_production_rules(*prod_rules)

    # Validation rules
    v_val = ValRule(IdealV, [ValPattern(SRC, IdealE, IdealI, Range(min=0)),
                        ValPattern(DST, IdealE, IdealI, Range(min=0)),
                        ValPattern(DST, IdealE, InpV, Range(min=0)),
                        ValPattern(DST, IdealE, InpI, Range(min=0)),
                        ValPattern(SELF, IdealE, IdealV, Range(exact=1))])
    i_val = ValRule(IdealI, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                        ValPattern(DST, IdealE, [IdealV, InpV, InpI], Range(min=0, max=1)),
                        ValPattern(SELF, IdealE, IdealI, Range(exact=1))])
    inpv_val = ValRule(InpV, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                            ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1))])
    inpi_val = ValRule(InpI, [ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
                            ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1))])
    val_rules = [v_val, i_val, inpv_val, inpi_val]
    tln_lang.add_validation_rules(*val_rules)


    hw_tln_lang = CDGLang("gmc-tln",inherits=tln_lang)
    # Nonideal implementation with 10% random variation
    MmV = NodeType(name='Vm', base=IdealV,
                attr_def=[AttrDefMismatch('c', attr_type=float, attr_range=lc_range, rstd=0.1)])
    MmI = NodeType(name='Im', base=IdealI,
                attr_def=[AttrDefMismatch('l', attr_type=float, attr_range=lc_range, rstd=0.1)])
    MmE = EdgeType(name='Em', base=IdealE,
                attr_def=[AttrDefMismatch('ws', attr_type=float, attr_range=w_range, rstd=0.1),
                            AttrDefMismatch('wt', attr_type=float, attr_range=w_range, rstd=0.1)])
    hw_tln_lang.add_types(MmV, MmI, MmE)

    _v2i_mm = ProdRule(MmE, IdealV, IdealI, SRC, -EDGE.ws*VAR(DST)/SRC.c)
    v2_i_mm = ProdRule(MmE, IdealV, IdealI, DST, EDGE.wt*VAR(SRC)/DST.l)
    _i2v_mm = ProdRule(MmE, IdealI, IdealV, SRC, -EDGE.ws*VAR(DST)/SRC.l)
    i2_v_mm = ProdRule(MmE, IdealI, IdealV, DST, EDGE.wt*VAR(SRC)/DST.c)
    inpv2_v_mm = ProdRule(MmE, InpV, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.c/SRC.r)
    inpv2_i_mm = ProdRule(MmE, InpV, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.r)/DST.l)
    inpi2_v_mm = ProdRule(MmE, InpI, IdealV, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST)*SRC.g)/DST.c)
    inpi2_i_mm = ProdRule(MmE, InpI, IdealI, DST, EDGE.wt*(SRC.fn(TIME)-VAR(DST))/DST.l/SRC.g)
    hw_prod_rules = [_v2i_mm, v2_i_mm, _i2v_mm, i2_v_mm, inpv2_v_mm, inpv2_i_mm, inpi2_v_mm, inpi2_i_mm]
    hw_tln_lang.add_production_rules(*hw_prod_rules)

    prod_rules += hw_prod_rules

    cdg_types = [IdealV, IdealI, IdealE, InpV, InpI, MmV, MmI, MmE]
    help_fn = [pulse]
    spec = CDGSpec(cdg_types, prod_rules, val_rules)


    validator = ArkValidator(solver=SMTSolver())
    compiler = ArkCompiler(rewrite=RewriteGen())
    return spec, validator, compiler, help_fn


def analyze_dynamics(graph,cutpoint):
    raise NotImplementedError


def saturation(sig):
    """Saturate the value at 1"""
    return 0.5 * (abs(sig + 1) - abs(sig - 1))

spec,validator,compiler,help_fn = make_cnn_lang()

II_T = spec.cdg_type("IdealI")
IV_T = spec.cdg_type("IdealV")
E_T = spec.cdg_type("IdealE")

g1 = CDG()

c = 1e-8
i1 = II_T(l=1.0*c,r=1.0*c)
v1 = IV_T(c=1.0*c,g=0.0)
v2 = IV_T(c=1.0*c,g=1.0*c)

g1.connect(E_T(wt=1.0,ws=1.0), v1, v1)
g1.connect(E_T(wt=1.0,ws=1.0), v1, i1)
g1.connect(E_T(wt=1.0,ws=1.0), i1, i1)
g1.connect(E_T(wt=1.0,ws=1.0), v2, v2)
g1.connect(E_T(ws=1.0,wt=1.0), i1, v2)
is_valid, hints = validator.validate(cdg=g1, cdg_spec=spec)
if is_valid != 0:
    raise Exception("CDG Invalid code=%d <%s>" % (is_valid, str(hints)))

compiler.compile_to_sympy(cdg=g1, cdg_spec=spec, help_fn=help_fn)
