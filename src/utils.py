import torch
import random


def set_params():
    # # # # # # # # Parameters # # # # # # # #
    n_traj              = 20            # number of trajectories collected at each step of the learning
    data                = torch.load('data/training_data.pt')
    sys_output, _, _    = get_data_set(data,0, n_traj)
    t_end               = sys_output.shape[1]
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate  = 5e-3  #2e-3   5e-3
    epochs         = 400   #400
    beta           = 1e2   #1e4
    gamma          = 1e-4  #1e-2
    q              = 10    #10
    FD_n_states    = 8     #4
    FD_n_nl        = 2     #2
    t_init         = 1     #1                            #initial sample for loss compution
    return t_end, learning_rate, epochs, FD_n_states, FD_n_nl, n_traj, data, gamma, q, beta, t_init


def get_data_set(d, n_data_set,n_traj):
    u1       = d[n_data_set,:].unsqueeze(0)
    u2       = d[n_data_set+n_traj,:].unsqueeze(0)
    y1       = d[n_data_set+2*n_traj,:].unsqueeze(0)
    y2       = d[n_data_set+3*n_traj,:].unsqueeze(0)
    y3       = d[n_data_set+4*n_traj,:].unsqueeze(0)
    y4       = d[n_data_set+5*n_traj,:].unsqueeze(0)
    f        = d[n_data_set+6*n_traj,:].unsqueeze(0)

    sys_input    = torch.cat((u1,u2),dim = 0)
    sys_output   = torch.cat((y1,y2,y3,y4),dim = 0)
    fault_signal = f

    return sys_output, sys_input, fault_signal


def generate_faulty_training_data(data, data_set, n_traj):
    class_size = round(n_traj / 4, 0)
    if data_set < class_size:
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * 0
        r2_label = fault_signal * 0
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    elif data_set < class_size*2:
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal
        r2_label = fault_signal * 0
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    elif data_set < class_size*3:
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * 0
        r2_label = fault_signal #0.15*torch.ones(fault_signal.shape)
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    else:
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal
        r2_label = fault_signal
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    return sys_input, sys_output_f, r_labels

def generate_faulty_test_data(data, data_set, n_traj, fault_type):
    if fault_type == 'healthy':
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * 0
        r2_label = fault_signal * 0
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    elif fault_type == 'first sensor fault':
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * torch.cat((torch.zeros(1,30), torch.ones(1,fault_signal.shape[1]-30)), dim=1)
        r2_label = fault_signal * 0
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    elif fault_type == 'second sensor fault':
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * 0
        r2_label = fault_signal * torch.cat((torch.zeros(1,30), torch.ones(1,fault_signal.shape[1]-30)), dim=1)   #0.15*torch.ones(fault_signal.shape)  #5*torch.ones(fault_signal.shape[0], 121)
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    elif fault_type == 'both sensors fault':
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * torch.cat((torch.zeros(1,30), torch.ones(1,fault_signal.shape[1]-30)), dim=1)
        r2_label = fault_signal * torch.cat((torch.zeros(1,30), torch.ones(1,fault_signal.shape[1]-30)), dim=1)
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    else:                                                                                   # default healthy
        sys_output, sys_input, fault_signal = get_data_set(data, data_set, n_traj)
        r1_label = fault_signal * 0
        r2_label = fault_signal * 0
        r3_label = fault_signal * 0
        r4_label = fault_signal * 0
        r_labels = [r1_label, r2_label, r3_label, r4_label]
        sys_output_f = sys_output + torch.cat((r1_label, r2_label, r3_label, r4_label), dim=0)
    return sys_input, sys_output_f, r_labels


# def compute_thresholds(training_data, FDs):
#     n_traj = 20
#     class_size = round(n_traj / 4, 0)
#     rs_max          = [0] * 4  # Store max values for each fault signal
#
#     for i in range(n_traj):
#         sys_input, sys_output_f, r_labels   = generate_faulty_training_data(training_data, i, n_traj)
#         t_end                               = sys_output_f.shape[1]
#         rs                                  = [torch.zeros(sys_output_f.shape[0], t_end) for _ in range(4)]
#         states                              = [torch.zeros(FD.n_states) for FD in FDs]
#
#         for t in range(t_end):
#             FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
#             for j, FD in enumerate(FDs):
#                 r, states[j]    = FD(FD_input, states[j])
#                 rs[j][:, t]     = r.detach()
#
#         for j in range(4):
#             if (i < class_size or class_size*2 <= i < class_size*3) and j == 0:
#                 rs_max[j] = max(rs_max[j], rs[j].max().item())
#
#             if i < 2*class_size and j == 1:
#                 rs_max[j] = max(rs_max[j], rs[j].max().item())
#
#             if j in [2, 3]:  # Always compute thresholds for r[2] and r[3] over all data
#                 rs_max[j] = max(rs_max[j], rs[j].max().item())
#
#     return rs_max
#
#
# def compute_confusion_matrix(test_data, n_traj, thresholds, FDs):
#     # Initialize confusion matrix (4x4)
#     confusion_matrix = torch.zeros((4, 4), dtype=torch.int)
#
#     fault_types = ['healthy', 'first sensor fault', 'second sensor fault', 'both sensors fault']
#
#     # Generate 1000 random test data samples
#     for i in range(1000):
#         # Randomly select fault type for each sample
#         fault_type = random.choice(fault_types)
#
#         # Generate faulty test data
#         sys_input, sys_output_f, r_labels = generate_faulty_test_data(test_data, i, 1000, fault_type)
#         t_end = sys_output_f.shape[1]
#         rs = [torch.zeros(sys_output_f.shape[0], t_end) for _ in range(4)]
#         states = [torch.zeros(FD.n_states) for FD in FDs]
#
#         for t in range(t_end):
#             FD_input = torch.cat((sys_input[:, t], sys_output_f[:, t]))
#             for j, FD in enumerate(FDs):
#                 r, states[j] = FD(FD_input, states[j])
#                 rs[j][:, t] = r.detach()
#
#         # Predict the fault class based on threshold comparison
#         for j in range(4):
#             # Compute the prediction based on the threshold for each fault signal
#             predicted_label = (rs[j].max() > thresholds[j]).int()  # 1 if fault detected, 0 otherwise
#             true_label      = ((r_labels[j].max())>0).int()  # True labels for the fault
#
#             if predicted_label == true_label:
#                 confusion_matrix[true_label, predicted_label] += 1
#             else:
#                 confusion_matrix[true_label, predicted_label] += 1
#
#             # Update confusion matrix based on predicted vs true labels
#             for true_label, predicted_label in zip(true_labels, predicted_labels):
#                 confusion_matrix[true_label, predicted_label] += 1
#
#     return confusion_matrix
#
# def compute_detection_and_alarm_rates(confusion_matrix):
#     # Total number of samples
#     total_samples = confusion_matrix.sum().item()
#
#     # Initialize Total Detection Rate (TDR) and False Alarm Rate (FAR)
#     total_detection_rate = 0.0
#     false_alarm_rate = 0.0
#
#     # Loop through each fault class (row in confusion matrix)
#     for i in range(4):
#         true_positives = confusion_matrix[i, i].item()  # Diagonal element
#         false_negatives = confusion_matrix[i, :].sum().item() - true_positives  # Sum of the row - TP
#         false_positives = confusion_matrix[:, i].sum().item() - true_positives  # Sum of the column - TP
#         true_negatives = total_samples - (true_positives + false_negatives + false_positives)
#
#         # Calculate TDR and FAR for each class
#         tdr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
#         far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
#
#         # Accumulate TDR and FAR
#         total_detection_rate += tdr
#         false_alarm_rate += far
#
#     # Average TDR and FAR over all fault classes
#     avg_detection_rate = total_detection_rate / 4
#     avg_alarm_rate = false_alarm_rate / 4
#
#     return avg_detection_rate, avg_alarm_rate

'''
# convert data from .mat to .pt:
import torch
from scipy.io import loadmat
data     = loadmat('training_data.mat')  # Choose the training set data file to be used
d        = data['training_data']
d        = torch.tensor(d)
d        = d.float()
torch.save(d, 'training_data.pt')
data     = loadmat('test_data.mat')  # Choose the training set data file to be used
d        = data['test_data']
d        = torch.tensor(d)
d        = d.float()
torch.save(d, 'test_data.pt')
'''