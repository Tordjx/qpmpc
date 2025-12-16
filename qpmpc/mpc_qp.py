#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""MPC problem represented as a quadratic program."""

import logging

import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix
from scipy.linalg import block_diag
from .exceptions import ProblemDefinitionError
from .mpc_problem import MPCProblem


class MPCQP:
    r"""MPC problem represented as a quadratic program.

    This class further stores intermediate matrices used to recompute cost and
    linear inequality vectors.
    """

    G: np.ndarray
    P: np.ndarray
    Phi: np.ndarray
    Psi: np.ndarray
    h: np.ndarray
    phi_last: np.ndarray
    psi_last: np.ndarray
    q: np.ndarray

    def __init__(self, mpc_problem: MPCProblem, sparse: bool = False) -> None:
        """Create a new QP representation.

        Args:
            mpc_problem: Model predictive control problem to cast as a QP.
            sparse: If set, use sparse matrix representation.
        """
        input_dim = mpc_problem.input_dim
        state_dim = mpc_problem.state_dim
        stacked_input_dim = mpc_problem.input_dim * mpc_problem.nb_timesteps
        if mpc_problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        initial_state: np.ndarray = mpc_problem.initial_state

        phi = np.eye(state_dim)
        psi = np.zeros((state_dim, stacked_input_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        e_list, C_list = [], []
        for k in range(mpc_problem.nb_timesteps):
            # Loop invariant: x == psi * U + phi * x_init
            phi_list.append(phi)
            psi_list.append(psi)
            A_k = mpc_problem.get_transition_state_matrix(k)
            B_k = mpc_problem.get_transition_input_matrix(k)
            C_k = mpc_problem.get_ineq_state_matrix(k)
            D_k = mpc_problem.get_ineq_input_matrix(k)
            e_k = mpc_problem.get_ineq_vector(k)
            if k == mpc_problem.nb_timesteps -1 :
                term_C_k = mpc_problem.get_terminal_ineq_state_matrix(k)
                term_D_k = mpc_problem.get_terminal_ineq_input_matrix(k)
                term_e_k = mpc_problem.get_terminal_ineq_vector(k)
                C_k = np.concatenate((C_k,term_C_k))
                D_k = np.concatenate((D_k,term_D_k))
                e_k = np.concatenate((e_k,term_e_k))
            G_k = np.zeros((e_k.shape[0], stacked_input_dim))
            h_k = (
                e_k
                if C_k is None
                else e_k - np.dot(C_k.dot(phi), initial_state)  # type: ignore
            )
            input_slice = slice(k * input_dim, (k + 1) * input_dim)
            if D_k is not None:
                # we rely on G == 0 to avoid a slower +=
                G_k[:, input_slice] = D_k
            if C_k is not None:
                G_k += C_k.dot(psi)  # type: ignore
            if k == 0 and D_k is None and np.any(h_k < 0.0):
                # in this case, the initial state constraint is violated and
                # cannot be compensated by any input (D_k is None)
                logging.warning(
                    "initial state is unfeasible: "
                    f"G_0 * x <= h_0 with G_0 == 0 and min(h_0) == {min(h_k)}"
                )
            G_list.append(G_k)
            h_list.append(h_k)
            phi = A_k.dot(phi)
            psi = A_k.dot(psi)
            psi[:, input_slice] = B_k
            e_list.append(e_k)
            C_list.append(C_k)
        
        G: np.ndarray = np.vstack(G_list, dtype=float)
        h: np.ndarray = np.hstack(h_list, dtype=float)
        Phi = np.vstack(phi_list, dtype=float)
        Psi = np.vstack(psi_list, dtype=float)
        if C_list != []:
            C = block_diag(*C_list)
        else : 
            C = None
        e = np.hstack(e_list, dtype=float)
        P: np.ndarray = np.kron(np.eye(mpc_problem.nb_timesteps), mpc_problem.stage_input_cost_weight)

        if mpc_problem.terminal_cost_weight is not None: 
            P += psi.T@mpc_problem.terminal_cost_weight@psi

        if mpc_problem.stage_state_cost_weight is not None:
            P += Psi.T@np.kron(np.eye(mpc_problem.nb_timesteps),mpc_problem.stage_state_cost_weight)@Psi
        q: np.ndarray = np.zeros(stacked_input_dim)

        self.G = csc_matrix(G) if sparse else G
        self.P = csc_matrix(P) if sparse else P
        self.Phi = Phi
        self.Psi = Psi
        self.h = h
        self.phi_last = phi
        self.psi_last = psi
        self.q = q  # initialized below
        self.e = e
        self.C = C
        self.CPhi = self.C@self.Phi
        self.PsiT = self.Psi.T
        #
        try:
            self.update_cost_vector(mpc_problem)
        except ProblemDefinitionError:
            pass

    @property
    def problem(self) -> qpsolvers.Problem:
        """Get quadratic program to call a QP solver."""
        return qpsolvers.Problem(self.P, self.q, self.G, self.h)
    
    def update_cost_vector(self, mpc_problem: MPCProblem) -> None:
        if mpc_problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")

        x0 = mpc_problem.initial_state
        self.q[:] = 0.0
        # ===== Stage cost =====
        if mpc_problem.has_stage_state_cost:
            Q = mpc_problem.stage_state_cost_weight
            x_star = mpc_problem.target_states

            nx = mpc_problem.state_dim

                        
            err = self.Phi @ x0 - x_star.reshape(-1)     # (N*nx,)
            err = err.reshape(-1, nx)                # (N, nx)

            # Apply Q to each stage (no block-diag)
            Qerr = err @ Q.T                        # (N, nx)
            Qerr = Qerr.reshape(-1)                 # (N*nx,)

            self.q[:] = self.PsiT @ Qerr                   # (N*nu,)



        # ===== Terminal cost =====
        if mpc_problem.has_terminal_cost:
            cT = self.phi_last @ x0 - mpc_problem.goal_state
            self.q += self.psi_last.T @ (mpc_problem.terminal_cost_weight @ cT)
    
    def update_constraint_vector(self, mpc_problem: MPCProblem) -> None:
        """Update the inequality constraint vector `h`.

        Args:
            mpc_problem: Updated MPC problem with a new initial state.
        """
        
        if mpc_problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        if self.C is not None:
            h= self.e - self.CPhi@mpc_problem.initial_state
            self.h = h.reshape(-1)
