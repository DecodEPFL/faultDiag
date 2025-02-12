import torch
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt

def plot_all_results(t_end, r_labels, r_FDs, filename='Residual of first sensor filter', text='', save=False):
    t = torch.linspace(0, t_end - 1, t_end)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.ion()

    font_size_title = 20
    font_size_labels = 20
    font_size_ticks = 20
    line_width = 3

    for ax in axs.flatten():  # Apply grid and box settings to all subplots
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Dashed grid lines
        ax.spines['top'].set_visible(True)  # Ensure the top border is visible
        ax.spines['right'].set_visible(True)  # Ensure the right border is visible

    axs[0, 0].plot(t, r_labels[0][0, :], label='Actual fault', linewidth=line_width)
    axs[0, 0].plot(t, r_FDs[0][0, :], label=filename, linewidth=line_width)
    axs[0, 0].set_title('First sensor', fontsize=font_size_title)
    axs[0, 0].set_xlabel('Sample', fontsize=font_size_labels)
    axs[0, 0].set_ylabel('[m]', fontsize=font_size_labels)
    axs[0, 0].set_ylim(-0.07, 0.07)
    axs[0, 0].set_xlim(20, 80)
    axs[0, 0].tick_params(labelsize=font_size_ticks)

    axs[0, 1].plot(t, r_labels[1][0, :], label='Actual fault', linewidth=line_width)
    axs[0, 1].plot(t, r_FDs[1][0, :], label=filename, linewidth=line_width)
    axs[0, 1].set_title('Second sensor', fontsize=font_size_title)
    axs[0, 1].set_xlabel('Sample', fontsize=font_size_labels)
    axs[0, 1].set_ylabel('[m]', fontsize=font_size_labels)
    axs[0, 1].set_ylim(-0.07, 0.07)
    axs[0, 1].set_xlim(20, 80)
    axs[0, 1].tick_params(labelsize=font_size_ticks)
    axs[0, 1].legend(loc='upper right', fontsize=font_size_labels)

    axs[1, 0].plot(t, r_labels[2][0, :], label='Actual fault', linewidth=line_width)
    axs[1, 0].plot(t, r_FDs[2][0, :], label=filename, linewidth=line_width)
    axs[1, 0].set_title('Third sensor', fontsize=font_size_title)
    axs[1, 0].set_xlabel('Sample', fontsize=font_size_labels)
    axs[1, 0].set_ylabel('[m/s]', fontsize=font_size_labels)
    axs[1, 0].set_ylim(-0.07, 0.07)
    axs[1, 0].set_xlim(20, 80)
    axs[1, 0].tick_params(labelsize=font_size_ticks)

    axs[1, 1].plot(t, r_labels[3][0, :], label='Actual fault', linewidth=line_width)
    axs[1, 1].plot(t, r_FDs[3][0, :], label=filename, linewidth=line_width)
    axs[1, 1].set_title('Fourth sensor', fontsize=font_size_title)
    axs[1, 1].set_xlabel('Sample', fontsize=font_size_labels)
    axs[1, 1].set_ylabel('[m/s]', fontsize=font_size_labels)
    axs[1, 1].set_ylim(-0.07, 0.07)
    axs[1, 1].set_xlim(20, 80)
    axs[1, 1].tick_params(labelsize=font_size_ticks)

    plt.tight_layout()

    if save:
        plt.savefig(f'results/figs/{filename}{text}.svg', format='svg', transparent=True)
        plt.savefig(f'results/figs/{filename}{text}.eps', format='eps')
        plt.savefig(f'results/figs/{filename}{text}.png', format='png')
    else:
        plt.show(block=False)
        plt.pause(1)
