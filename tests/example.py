import ipopt
import matplotlib.pyplot as plt
import numpy as np
from pyomeca import Analogs3d

from static_optim.dynamic_models import (
    ClassicalStaticOptimization,
    ClassicalOptimizationLinearConstraints,
    LocalOptimizationLinearConstraints,
)

setup = {
    "model": "template/arm26.osim",
    "mot": "data/arm26_InverseKinematics.mot",
    "filter_param": None,
    "muscle_physiology": False,
}

# static optimization model

# model = ClassicalStaticOptimization(setup["model"], setup["mot"], setup["filter_param"])

# model = ClassicalOptimizationLinearConstraints(
#     setup["model"],
#     setup["mot"],
#     setup["filter_param"],
#     muscle_physiology=setup["muscle_physiology"],
# )

model = LocalOptimizationLinearConstraints(
    setup["model"],
    setup["mot"],
    setup["filter_param"],
    muscle_physiology=setup["muscle_physiology"],
)

# optimization options
activation_initial_guess = np.zeros([model.n_muscles])
lb, ub = model.get_bounds()

# problem
problem = ipopt.problem(
    n=model.n_muscles,  # Nb of variables
    lb=lb,  # Variables lower bounds
    ub=ub,  # Variables upper bounds
    m=model.n_dof,  # Nb of constraints
    cl=np.zeros(model.n_dof),  # Lower bound constraints
    cu=np.zeros(model.n_dof),  # Upper bound constraints
    problem_obj=model,  # Class that defines the problem
)
problem.addOption("tol", 1e-7)
problem.addOption("print_level", 0)

# optimization
activations = []
for iframe in range(model.n_frame):
    model.upd_model_kinematics(iframe)

    try:
        x, info = problem.solve(activation_initial_guess)
    except RuntimeError:
        print(f"Error while computing the frame #{iframe}")

    # the answer is the initial guess for next frame
    activation_initial_guess = x
    activations.append(x)

    print(
        f'time: {model.get_time(iframe)} | perf: {info.get("obj_val")} | violation: {np.linalg.norm(info.get("g"))}'
    )

# compare with gui
data = {
    "from_us": Analogs3d(np.array(activations)),
    "from_gui": Analogs3d.from_csv(
        "data/arm26_StaticOptimization_activation.sto",
        delimiter="\t",
        time_column=0,
        header=7,
        first_column=1,
        first_row=8,
    ),
}

data["from_us"].plot()
data["from_gui"].plot()
(data["from_us"] - data["from_gui"]).plot()

plt.show()
