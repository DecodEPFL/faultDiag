import torch

from src.models import R_REN
from src.loss_functions import loss_fun
from src.utils import set_params, get_data_set, generate_faulty_training_data, generate_faulty_test_data
from src.plots import plot_results

# def main():
if __name__ == '__main__':
    torch.manual_seed(1)
    # # # # # # # # Parameters, hyperparameters # # # # # # # #
    params                      = set_params()
    t_end, learning_rate, \
    epochs, FD_n_states, \
    FD_n_nl, n_traj, data, gamma, q, beta, t_init     = params
    # # # # # # # # Get system ata # # # # # # # #d
    sys_output, sys_input, fault_signal     = get_data_set(data, 0, n_traj)
    FD_n_inputs                             = sys_output.shape[0] + sys_input.shape[0]
    FD_n_outputs                            = 1
    # # # # # # # # Define models # # # # # # # #
    # note that above you have same settings for all FDs but below you modify tuning of each. howverver, later in the name
    # of save you use unified setting above, which is bad. Good news is that for the ok results you got you put
    # settings correctly in the ff code.
    FDs = [R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 1, sys_input.shape[0], gamma=gamma, beta= beta, q= q),
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 2, sys_input.shape[0], gamma=gamma, beta= beta, q=q),
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 3, sys_input.shape[0], gamma=gamma, beta= beta, q= q),
           R_REN(FD_n_inputs, FD_n_outputs, FD_n_states, FD_n_nl, 4, sys_input.shape[0], gamma=gamma, beta= beta, q= q)]
    # # # # # # # # Define optimizer and parameters # # # # # # # #
    optimizers   = [torch.optim.Adam(index.parameters(), lr=learning_rate) for index in FDs]
    # # # # # # # # Training # # # # # # # #
    print("------------ Begin training ------------")
    print("Fault detection problem -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate +
          " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj)
    print("REN info -- Four FDs (RENs) -- number of states: %i" % FD_n_states +
          " -- number of nonlinearities: %i" % FD_n_nl)
    print("--------- --------- ---------  ---------")

    # stopped = [False] * len(FDs)  # Track which models have stopped training
    stopped = [False, False, False, False]

    for epoch in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad()
        losses = [0] * len(FDs)

        for kk in range(n_traj):
            sys_input, sys_output_f, r_labels = generate_faulty_training_data(data, kk, n_traj)
            states = [1e-5 * torch.ones(FD.n_states) for FD in FDs]
            for t in range(t_end):
                FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
                for i, FD in enumerate(FDs):
                    if stopped[i]:
                        continue  # Skip the model that has already met the criterion
                    r, states[i] = FD(FD_input, states[i])
                    if t >= t_init:
                        losses[i] += loss_fun(r_labels[i][:, t], r)

        losses_str = " --- ".join([f"Loss{i + 1}: {loss:.4f}" for i, loss in enumerate(losses)])
        print(f"Epoch: {epoch} --- {losses_str} ---")

        for i, FD in enumerate(FDs):
            if not stopped[i]:  # Only update models that haven't stopped
                if losses[i] < 5:
                    print(f"FD {i + 1} stopping as loss is below 8 at epoch {epoch}")
                    stopped[i] = True  # Mark this model as stopped
                else:
                    losses[i].backward()  # retain_graph=True
                    optimizers[i].step()
                    FD.set_model_param()

        # If all models have stopped, break the loop
        if all(stopped):
            print(f"All FDs have stopped training by epoch {epoch}")
            break
    # # # # # # # # Save trained model # # # # # # # #
    gamma_save = gamma*1e8
    for i, FD in enumerate(FDs):
        torch.save(FD.state_dict(), "results/"+"FD_ %i" % i+"epochs_ %i" % epochs+"_gamma_ %i" % gamma_save+
                                        "_q_ %i" % q+"_beta_ %i" % beta+"_t_init_ %i" % t_init+
                                        "_state_size_ %i" % FD_n_states+"_nl_size_ %i" % FD_n_nl+".pt")
    # # # # # # # # Plot results # # # # # # # #
    sys_input, sys_output_f, r_labels   = generate_faulty_training_data(data, 12, n_traj)
    rs                                  = [torch.zeros(fault_signal.shape[0], t_end) for _ in range(4)]
    states                              = [torch.zeros(FD.n_states) for FD in FDs]
    for t in range(t_end):
        FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
        for i, FD in enumerate(FDs):
            r, states[i] = FD(FD_input, states[i])
            rs[i][:, t]       = r.detach()
    for i, FD in enumerate(FDs):
        if i>1:
            ylebel = '[m]'
        else:
            ylebel = '[m/s]'
        plot_results(t_end, r_labels[i], rs[i], 'Residual of FD #%i' % (i + 1),ylebel,
                     "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save + "_q_ %i" % q + "_beta_ %i" % beta
                     + "_t_init_ %i" % t_init + "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl
                     + "_trained.pt", save=False)

    # # # # # # # # Plot results for test data # # # # # # # #
    test_data                                   = torch.load('data/test_data.pt')
    sys_input, sys_output_f, r_labels           = generate_faulty_test_data(test_data, 500, 1000,
                                                                            'first sensor fault')
    t_end           = sys_output_f.shape[1]
    rs              = [torch.zeros(fault_signal.shape[0], t_end) for _ in range(4)]
    states          = [torch.zeros(FD.n_states) for FD in FDs]

    for t in range(t_end):
        FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
        for i, FD in enumerate(FDs):
            r, states[i] = FD(FD_input, states[i])
            rs[i][:, t] = r.detach()
    for i, FD in enumerate(FDs):
        if i > 1:
            ylebel = '[m]'
        else:
            ylebel = '[m/s]'
        plot_results(t_end, r_labels[i], rs[i], 'Residual of FD #%i' % (i + 1), ylebel,
                     "epochs_ %i" % epochs + "_gamma_ %i" % gamma_save + "_q_ %i" % q + "_beta_ %i" % beta
                     + "_t_init_ %i" % t_init + "_state_size_ %i" % FD_n_states + "_nl_size_ %i" % FD_n_nl
                     + "_trained.pt", save=False)

    input("Press Enter to close the plot and end the script...")

# Press the green button in the gutter to run the script.

    # Run main
    # main()

