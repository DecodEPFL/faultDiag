import torch
import matplotlib.pyplot as plt


def plot_results(t_end, r_label, r_FD, filename = 'Residual of first sensor filter',ylabel= '[m]', text ='', save=False):
    t = torch.linspace(0, t_end - 1, t_end)
    plt.figure()
    plt.ion()
    plt.plot(t, r_label[0, :], label='Actual fault')
    plt.plot(t, r_FD[0, :], label= filename)
    plt.xlabel('Sample')
    plt.ylabel(ylabel)
    # plt.ylim(-16, 16)
    # plt.ylim(-0.07, 0.07)
    # plt.xlim(0, 80)
    plt.legend(loc='upper right')

    if save:
        plt.savefig('results/figs/' + filename  +text + '.svg', format='svg',transparent=True)
        plt.savefig('results/figs/' + filename  +text + '.eps', format='eps')
        plt.savefig('results/figs/' + filename  +text + '.png', format='png')
    else:
        plt.show(block=False)
        plt.pause(1)

