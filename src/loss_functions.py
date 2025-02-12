from torch import sqrt, matmul

# remove this and use mse torch.nn.MSELoss

def loss_fun(sys_output, nn_output):
    delta = sys_output - nn_output
    return sqrt(matmul(delta, delta))