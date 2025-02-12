import torch
import torch.nn as nn
import torch.nn.functional as F




# Contractive REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
# Matrix notations are same as the paper above.

class R_REN(nn.Module):
        def __init__(self, n_inputs, n_outputs, n_states, n_nl, filter_index, n_sys_inputs, bias = False,
                                                                                        gamma= 1e-3, q = 10, beta=1e5):
            super().__init__()
            self.n_inputs   = n_inputs
            self.n_states   = n_states
            self.n_nl       = n_nl
            self.n_outputs  = n_outputs
            s = min(n_inputs, n_outputs)                                                                                #size of X3,Y3 and Z3 depend on s but not ecplicitly mentioed in paper
            # # # # # # # # # IQC specification # # # # # # # # #
            m                   = self.n_inputs - n_sys_inputs
            self.Q              = -q * torch.eye(self.n_outputs)
            R_diag              = torch.cat((beta * torch.ones(n_sys_inputs),
                                                gamma * torch.ones(filter_index-1),
                                                    torch.tensor([beta]),
                                                        gamma * torch.ones(m-filter_index)),dim = 0)

            self.R              = torch.diag(R_diag)
            # self.S              = torch.zeros(self.n_inputs, self.outputs)
            # # # # # # # # # Training parameters # # # # # # # # #
            # Auxiliary matrices:
            std         = 0.1
            self.X      = nn.Parameter((torch.randn(2 * n_states + n_nl, 2 * n_states + n_nl) * std))
            self.Y1     = nn.Parameter((torch.randn(n_states, n_states) * std))
            self.X3     = nn.Parameter((torch.randn(s, s) * std))
            self.Y3     = nn.Parameter((torch.randn(s, s) * std))
            self.Z3     = nn.Parameter((torch.randn(abs(n_inputs- n_outputs), s) * std))
            # NN state dynamics:
            self.B2     = nn.Parameter((torch.randn(n_states, n_inputs) * std))
            # NN output:
            self.C2     = nn.Parameter((torch.randn(n_outputs, n_states) * std))
            self.D21    = nn.Parameter((torch.randn(n_outputs, n_nl) * std))
            # v signal:
            self.D12    = nn.Parameter((torch.randn(n_nl, n_inputs) * std))
            if bias:
                    self.bx = nn.Parameter(torch.randn(n_states))
                    self.bv = nn.Parameter(torch.randn(n_nl))
                    self.bu = nn.Parameter(torch.randn(n_outputs))
            else:
                    self.bx = torch.zeros(n_states)
                    self.bv = torch.zeros(n_nl)
                    self.bu = torch.zeros(n_outputs)
            # # # # # # # # # Non-trainable parameters # # # # # # # # #
            # Auxiliary elements
            self.epsilon    = 0.001
            self.F          = torch.zeros(n_states, n_states)
            self.B1         = torch.zeros(n_states, n_nl)
            self.E          = torch.zeros(n_states, n_states)
            self.Lambda     = torch.ones(n_nl)
            self.C1         = torch.zeros(n_nl, n_states)
            self.D11        = torch.zeros(n_nl, n_nl)
            self.D22        = torch.zeros(n_outputs, n_inputs)
            # self.P = torch.zeros(n_states, n_states)
            # self.P_cal = torch.zeros(n_states, n_states)
            self.set_model_param()

        def set_model_param(self):
            n_states, n_nl  = self.n_states, self.n_nl
            Q, R            = self.Q, self.R


            # Calculate D22:
            M   = (torch.matmul(self.X3.T, self.X3) + self.Y3 - self.Y3.T
                                                + torch.matmul(self.Z3.T,self.Z3)
                                                                        + self.epsilon * torch.eye(self.X3.shape[0]))
            s   = min(self.n_inputs, self.n_outputs)                                                                    #size of X3,Y3 and Z3 depend on s but not ecplicitly mentioed in paper
            if  self.n_outputs >= self.n_inputs:
                N = torch.cat((torch.matmul(torch.eye(s) - M, torch.inverse(torch.eye(s) + M)),
                                                    torch.matmul(-2*self.Z3, torch.inverse(torch.eye(s) + M))), dim=0)
            else:
                N = torch.cat((torch.matmul(torch.inverse(torch.eye(s) + M),torch.eye(s) - M),
                               torch.matmul(-2 *torch.inverse(torch.eye(s) + M), self.Z3.T)), dim=1)
            Lq          = torch.linalg.cholesky(-self.Q).T
            Lr          = torch.linalg.cholesky(self.R).T
            self.D22    =  torch.matmul(torch.inverse(Lq), torch.matmul(N, Lr))

            # Calculate psi_r:     term added to H of R_REN  with R
            R_cal       = R + torch.matmul(self.D22.T,torch.matmul(Q, self.D22))
            R_cal_inv   = torch.linalg.inv(R_cal)                                                                                                                                   # what is differnce of this with inverse?
            C2_cal      = torch.matmul(torch.matmul(self.D22.T, self.Q), self.C2).T
            D21_cal     = torch.matmul(torch.matmul(self.D22.T, self.Q), self.D21).T - self.D12
            vec_r       = torch.cat((C2_cal, D21_cal, self.B2), dim=0)
            psi_r       = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))

            # Calculate psi_q:  term added to H of R_REN with Q
            vec_q       = torch.cat((self.C2.T, self.D21.T, torch.zeros(self.n_states, self.n_outputs)), dim=0)
            psi_q       = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))


            # Create H matrix:
            H               = (torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_states + n_nl)
                                                                                                      + psi_r - psi_q)

            h1, h2, h3      = torch.split(H, [n_states, n_nl, n_states], dim=0)
            H11, H12, H13   = torch.split(h1, [n_states, n_nl, n_states], dim=1)
            H21, H22, _     = torch.split(h2, [n_states, n_nl, n_states], dim=1)
            H31, H32, H33   = torch.split(h3, [n_states, n_nl, n_states], dim=1)
            P               = H33
            # NN state dynamics:
            self.F          = H31
            self.B1         = H32
            # NN output:
            self.E          = 0.5 * (H11 + P + self.Y1 - self.Y1.T)
            # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
            self.Lambda     = 0.5 * torch.diag(H22)
            self.D11        = -torch.tril(H22, diagonal=-1)
            self.C1         = -H21

        def forward(self, inpt, state):

            if self.n_nl != 0:
                vec = torch.zeros(self.n_nl)
                vec[0] = 1
                w = torch.zeros(self.n_nl)
                v = F.linear(state, self.C1[0, :]) + F.linear(inpt, self.D12[0, :])+ self.bv[0]
                w = w + vec * torch.tanh(v / self.Lambda[0])
                for i in range(1, self.n_nl):
                    vec = torch.zeros(self.n_nl)
                    vec[i] = 1
                    v = F.linear(state, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(inpt, self.D12[i,
                                                                                                      :])+self.bv[i]
                    w = w + vec * torch.tanh(v / self.Lambda[i])
            else:
                w = torch.zeros(self.n_nl)
            E_x_ = F.linear(state, self.F) + F.linear(w, self.B1) + F.linear(inpt, self.B2) + self.bx
            state_plus = F.linear(E_x_, self.E.inverse())
            output = F.linear(state, self.C2) + F.linear(w, self.D21) + F.linear(inpt, self.D22) + self.bu
            return output, state_plus


class C_REN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_states, n_nl, bias = True):
        super().__init__()
        self.n_inputs   = n_inputs
        self.n_states   = n_states
        self.n_nl       = n_nl
        self.n_outputs  = n_outputs
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std             = 0.1                                                                                           # should be tuned so not stuck in bad locall minumum, but easy to tune - mainly small numbers work great
        self.X          = nn.Parameter((torch.randn(2 * n_states + n_nl, 2 * n_states + n_nl) * std))
        self.Y1         = nn.Parameter((torch.randn(n_states, n_states) * std))
        # NN state dynamics:
        self.B2         = nn.Parameter((torch.randn(n_states, n_inputs) * std))
        # NN output:
        self.C2         = nn.Parameter((torch.randn(n_outputs, n_states) * std))
        self.D21        = nn.Parameter((torch.randn(n_outputs, n_nl) * std))
        self.D22        = nn.Parameter((torch.randn(n_outputs, n_inputs) * std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(n_nl, n_inputs) * std))
        # bias:
        if bias:
            self.bx = nn.Parameter(torch.randn(n_states))
            self.bv = nn.Parameter(torch.randn(n_nl))
            self.bu = nn.Parameter(torch.randn(n_outputs))
        else:
            self.bx = torch.zeros(n_states)
            self.bv = torch.zeros(n_nl)
            self.bu = torch.zeros(n_outputs)
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon    = 0.001
        self.F          = torch.zeros(n_states, n_states)
        self.B1         = torch.zeros(n_states, n_nl)
        self.E          = torch.zeros(n_states, n_states)
        self.Lambda     = torch.ones(n_nl)
        self.C1         = torch.zeros(n_nl, n_states)
        self.D11        = torch.zeros(n_nl, n_nl)
        self.set_model_param()                                                                                          #non_trainable parameters should be here.

    # set parameters of NN to ensure built-in properties of REN:
    def set_model_param(self):
        n_states        = self.n_states
        n_nl            = self.n_nl
        H               = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_states + n_nl)
        h1, h2, h3      = torch.split(H, (n_states, n_nl, n_states), dim=0)
        H11, H12, H13   = torch.split(h1, (n_states, n_nl, n_states), dim=1)
        H21, H22, _     = torch.split(h2, (n_states, n_nl, n_states), dim=1)
        H31, H32, H33   = torch.split(h3, (n_states, n_nl, n_states), dim=1)
        P = H33
        # NN state dynamics:
        self.F          = H31
        self.B1         = H32
        # NN output:  (alpha bar, upper bound of contracting rate is assumed to be 1)
        self.E          = 0.5 * (H11 + P + self.Y1 - self.Y1.T)
        # v signal:  [Change the following 2 lines if you don't want a strictly acyclic REN!]
        self.Lambda     = 0.5*torch.diag(H22)
        self.D11        = - torch.tril(H22, diagonal=-1)
        self.C1         = - H21

    def forward(self, inpt, state):
        if self.n_nl != 0:
            vec         = torch.zeros(self.n_nl)
            vec[0]      = 1
            w           = torch.zeros(self.n_nl)
            v           = F.linear(state, self.C1[0, :]) + F.linear(inpt, self.D12[0, :])  # + self.bv[0]
            w           = w + vec * torch.tanh(v / self.Lambda[0])
            for i in range(1, self.n_nl):
                vec         = torch.zeros(self.n_nl)
                vec[i]      = 1
                v           = F.linear(state, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(inpt, self.D12[i, :]) # self.bv[i]
                w           = w + vec * torch.tanh(v / self.Lambda[i])
        else:
            w = torch.zeros(self.n_nl)
        E_x_        = F.linear(state, self.F) + F.linear(w, self.B1) + F.linear(inpt, self.B2)  # + self.bxi
        state_plus  = F.linear(E_x_, self.E.inverse())
        output      = F.linear(state, self.C2) + F.linear(w, self.D21) + F.linear(inpt, self.D22)  # + self.bu
        return output, state_plus

    # Robust REN implementation in the acyclic version and a specific selection of IQC appropriate for FD
    # edit here names
    # check line by line relations
    # check comments in C_REN --- I think you should revise some part
    # make consistent with C_REN
