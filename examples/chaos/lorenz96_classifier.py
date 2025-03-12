import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from classifier_parser import args
from diffrax import Tsit5
from spec import build_lorenz96_sys, lorenz96_spec

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import TrainableMgr

N_STATE_VAR = args.n_state_var
N_STEPS = args.n_steps
BATCH_SIZE = args.batch_size
LR = args.lr

READOUT_TIME = args.readout_time
N_TIME_POINTS = args.n_time_points
FORCING = args.forcing

trainable_mgr = TrainableMgr()
time_points = np.linspace(0, READOUT_TIME, N_TIME_POINTS, endpoint=True)
time_info = TimeInfo(
    t0=0,
    t1=READOUT_TIME,
    dt0=READOUT_TIME / N_TIME_POINTS,
    saveat=time_points,
)


if __name__ == "__main__":
    np.random.seed(0)
    lorenz_sys_cdg, state_vars = build_lorenz96_sys(
        n_state_var=N_STATE_VAR, init_F=FORCING, trainable_mgr=trainable_mgr
    )
    lorenz_sys_cls = OptCompiler().compile(
        prog_name="lorenz_96",
        cdg=lorenz_sys_cdg,
        cdg_spec=lorenz96_spec,
        trainable_mgr=trainable_mgr,
        readout_nodes=state_vars,
        vectorize=True,
        normalize_weight=False,
    )

    lorenz_sys: BaseAnalogCkt = lorenz_sys_cls(
        init_trainable=trainable_mgr.get_initial_vals(),
        is_stochastic=False,
        solver=Tsit5(),
    )

    initial_state = jnp.array(np.random.randn(N_STATE_VAR))
    trace = lorenz_sys(
        time_info=time_info,
        initial_state=initial_state,
        switch=[],
        args_seed=0,
        noise_seed=0,
    )
    plt.plot(time_points, trace)
    plt.show()
