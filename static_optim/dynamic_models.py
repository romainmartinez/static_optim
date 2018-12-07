"""
Dynamic models.

These are the actual classes to send to IPOPT.
"""

from static_optim.kinematic import KinematicModel
from static_optim.forces import ResidualForces, ExternalForces
from static_optim.objective import ObjMinimizeActivation
from static_optim.constraints import ConstraintAccelerationTarget

import opensim as osim
import numpy as np


class ClassicalStaticOptimization(
    KinematicModel,
    ObjMinimizeActivation,
    ConstraintAccelerationTarget,
    ResidualForces,
    ExternalForces,
):
    """
    Computes the muscle activations in order to minimize them while targeting the acceleration from inverse kinematics.
    This is the most classical approach to Static Opimization.
    """

    # TODO: write tests
    def __init__(
        self,
        model,
        mot,
        filter_param=None,
        activation_exponent=2,
        residual_actuator_xml=None,
        external_load_xml=None,
    ):
        KinematicModel.__init__(self, model, mot, filter_param)
        ObjMinimizeActivation.__init__(self, activation_exponent)
        ResidualForces.__init__(self, residual_actuator_xml)
        ExternalForces.__init__(self, external_load_xml)

    def forward_dynamics(self, x):
        # set residual forces
        fs = self.model.getForceSet()
        for i in range(self.n_muscles, fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.setOverrideActuation(self.state, x[i])

        # update muscles
        muscle_activation = x[: self.n_muscles]
        for m in range(self.n_muscles):
            self.muscle_actuators.get(m).setActivation(self.state, muscle_activation[m])
        self.model.equilibrateMuscles(self.state)
        self.model.realizeAcceleration(self.state)
        return self.state.getUDot()


# ClassicalOptimizationLinearConstraints intends to mimic the classical approach but with the constraints linearized.
# It makes the assumption that muscle length is constant at a particular position and velocity, whatever the muscle
# activation.
class ClassicalOptimizationLinearConstraints(
    KinematicModel,
    ObjMinimizeActivation,
    ConstraintAccelerationTarget,
    ResidualForces,
    ExternalForces,
):
    """
    Intends to mimic the classical approach but with the constraints linearized.
    It makes the assumption that muscle length is constant at a particular position and velocity, whatever the muscle
    activation.
    """

    # TODO: write tests
    def __init__(
        self,
        model,
        mot,
        filter_param=None,
        activation_exponent=2,
        residual_actuator_xml=None,
        external_load_xml=None,
        muscle_physiology=True,
    ):
        self.muscle_physiology = muscle_physiology

        KinematicModel.__init__(self, model, mot, filter_param)
        ObjMinimizeActivation.__init__(self, activation_exponent)
        ResidualForces.__init__(self, residual_actuator_xml)
        ExternalForces.__init__(self, external_load_xml)

        # prepare linear constraints variables
        self.optimal_forces = []
        self.constraint_vector = []
        self.constraint_matrix = []
        self.jacobian_matrix = []  # Precomputed jacobian
        self._prepare_constraints()

    def forward_dynamics(self, x):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.setOverrideActuation(self.state, x[i] * self.optimal_forces[i])

        self.model.realizeAcceleration(self.state)
        return self.state.getUDot()

    def upd_model_kinematics(self, frame):
        super().upd_model_kinematics(frame)
        self._prepare_constraints()

    def _prepare_constraints(self):
        self.model.realizeVelocity(self.state)

        forces = self.model.getForceSet()
        self.optimal_forces = []
        for i in range(forces.getSize()):
            muscle = osim.Muscle.safeDownCast(forces.get(i))
            if muscle:
                if self.muscle_physiology:
                    self.model.setAllControllersEnabled(True)
                    self.optimal_forces.append(
                        muscle.calcInextensibleTendonActiveFiberForce(self.state, 1.0)
                    )
                    self.model.setAllControllersEnabled(False)
                else:
                    self.optimal_forces.append(muscle.getMaxIsometricForce())
            coordinate = osim.CoordinateActuator.safeDownCast(forces.get(i))
            if coordinate:
                self.optimal_forces.append(coordinate.getOptimalForce())

        self.linear_constraints()

    def constraints(self, x, idx=None):
        x_tp = x.reshape((x.shape[0], 1))
        x_mul = np.array((self.constraint_matrix.shape[0], x_tp.shape[1]))
        x_constraint = x_mul.ravel()
        c = x_constraint + self.constraint_vector
        if idx:
            return c[idx]
        else:
            return c

    def linear_constraints(self):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.overrideActuation(self.state, True)

        p_vector = np.zeros(self.n_actuators)
        self.constraint_vector = np.array(super().constraints(p_vector))

        self.constraint_matrix = np.zeros((self.n_dof, self.n_actuators))

        for p in range(self.n_actuators):
            p_vector[p] = 1
            self.constraint_matrix[:, p] = (
                np.array(super().constraints(p_vector)) - self.constraint_vector
            )
            p_vector[p] = 0

    def jacobian(self, x):
        return self.constraint_matrix
