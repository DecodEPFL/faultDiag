import torch

from src.utils import set_params, get_data_set, generate_faulty_training_data, generate_faulty_test_data
from src.plots import plot_results
from src.plots_all_in1 import plot_all_results
from src.models import R_REN

# def main():
torch.manual_seed(1)
# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params()
t_end, _, \
_, _,  \
_, n_traj, data, _, _, _, _    = params

epochs          = 400  # specifications of the model want to load 500
beta            = 1e2  # 1e5
gamma           = 1e-4 # 1e-5
q               = 10   # 10
FD_n_states     = 8    # 2
FD_n_nl         = 2    # 4
t_init          = 1    # 2
# # # # # # # # Get system data # # # # # # # #
sys_output, sys_input, fault_signal = get_data_set(data, 0, n_traj)
FD_n_inputs = sys_output.shape[0] + sys_input.shape[0]
FD_n_outputs = 1
# # # # # # # # Define models # # # # # # # #
FDs = [R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 1, sys_input.shape[0], gamma=gamma, beta= beta, q= q),
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 2, sys_input.shape[0], gamma=gamma, beta=beta, q=q),  # for other good result this is gamma e-2 and beta e4 , you need to change saving load below
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 3, sys_input.shape[0], gamma=gamma, beta= beta, q= q),
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 4, sys_input.shape[0], gamma=gamma, beta= beta, q= q)]
# # # # # # # # Load models # # # # # # # #
gamma_save = gamma*1e8

for i, FD in enumerate(FDs):
    saved_dict_FD = torch.load("results/ok_results/" + "FD_ %i" % i + "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save +
               "_q_ %i" % q + "_beta_ %i" % beta + "_t_init_ %i" % t_init +
               "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl + ".pt")
    FDs[i].load_state_dict(saved_dict_FD)
    FDs[i].set_model_param()

# # # # # # # # Plot results for one training data # # # # # # # #

sys_input, sys_output_f, r_labels   = generate_faulty_training_data(data, 12, n_traj)
rs                                  = [torch.zeros(fault_signal.shape[0], t_end) for _ in range(4)]
states                              = [torch.zeros(FD.n_states) for FD in FDs]
for t in range(t_end):
    FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
    for i, FD in enumerate(FDs):
        r, states[i] = FD(FD_input, states[i])
        rs[i][:, t]       = r.detach()

# for i, FD in enumerate(FDs):
#     if i > 1:
#         ylebel = '[m]'
#     else:
#         ylebel = '[m/s]'
#     plot_results(t_end, r_labels[i], rs[i], 'Residual of FD #%i'% (i+1), ylebel,
#                  "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save + "_q_ %i" % q + "_beta_ %i" % beta
#                  + "_t_init_ %i" % t_init + "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl
#                  + "_trained.pt", save=False)


# thresholds = compute_thresholds(data, FDs)
# print("Computed thresholds:", thresholds)
#
#
# conf_matrix = compute_confusion_matrix(test_data, 1000, thresholds, FDs)
# print(conf_matrix)
#
# TDR, FAR = compute_detection_and_alarm_rates(conf_matrix)


# # # # # # # # Plot results for one test data # # # # # # # #
test_data = torch.load('data/test_data.pt')
sys_input, sys_output_f, r_labels           = generate_faulty_test_data(test_data, 900, 1000,
                                                                            'first sensor fault')
t_end           = sys_output_f.shape[1]
rs              = [torch.zeros(fault_signal.shape[0], t_end) for _ in range(4)]
states          = [torch.zeros(FD.n_states) for FD in FDs]

for t in range(t_end):
    FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
    for i, FD in enumerate(FDs):
        r, states[i] = FD(FD_input, states[i])
        rs[i][:, t] = r.detach()
# for i, FD in enumerate(FDs):
#     if i > 1:
#         ylebel = '[m]'
#     else:
#         ylebel = '[m/s]'
#     plot_results(t_end, r_labels[i], rs[i], 'Residual of FD #%i' % (i + 1), ylebel,
#                      "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save + "_q_ %i" % q + "_beta_ %i" % beta
#                      + "_t_init_ %i" % t_init + "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl
#                      + "_trained.pt", save=False)


plot_all_results(t_end, r_labels, rs, 'Residuals of FDs',
                     "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save + "_q_ %i" % q + "_beta_ %i" % beta
                     + "_t_init_ %i" % t_init + "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl
                     + "_trained.pt", save=True)


input("Press Enter to close the plot and end the script...")

