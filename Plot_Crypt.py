import numpy as np
import matplotlib.pyplot as plt


def plot_CPA_KPA():
    Terms = ['CPA Attack', 'KPA Attack']
    Classifier = ['DES', 'AES', 'ECC', 'RSA', 'MA-ABPRE']
    marker = ['^', 's', 'o', 'd', 'x']
    lines = ['-', '--', '-.', ':', '-']
    Block_size = [1, 2, 3, 4, 5]  # Hidden neuron counts

    for m in range(len(Terms)):
        Eval = np.load('Eval_attack.npy', allow_pickle=True)[m]

        # Create figure and axis with reduced size
        fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size

        for s in range(Eval.shape[0]):
            ax.plot(Block_size, Eval[:, s], marker=marker[s], label=Classifier[s], color='black',
                    linestyle=lines[s])

        ax.set_xticks(Block_size)
        ax.set_xlabel('Block Size', fontsize=12)
        ax.set_ylabel(Terms[m], fontsize=12)

        # Legend inside the overall box
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=10, ncol=len(Classifier),
                           frameon=False)

        # Show only y-axis grid
        ax.grid(axis='y')

        # Remove only top and right box lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Keep left and bottom box lines with custom styling
        ax.spines['left'].set_edgecolor('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(1.5)

        # Apply tight layout to minimize extra space
        plt.tight_layout(pad=1.5)

        # Add an external bounding box
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2.5)  # Thickness of the extra bounding box

        # Adjust subplot layout to make space for legend
        plt.subplots_adjust(top=0.85)  # Move the content slightly downward

        path = "./Results/%s.png" % (Terms[m])
        plt.savefig(path, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def plot_ENC_DEC_CONS_MEM_1():
    Terms = ['Total Consumption time', 'Memory Size', 'Encryption time', 'Decryption time']
    Classifier = ['DES', 'AES', 'ECC', 'RSA', 'MA-ABPRE']
    marker = ['^', 's', 'o', 'd', 'x']
    lines = ['-', '--', '-.', ':', '-']
    Block_size = [1, 2, 3, 4, 5]  # Hidden neuron counts

    for m in range(len(Terms)):
        Eval = np.load('Eval_time.npy', allow_pickle=True)[m]

        # Create figure and axis with reduced size
        fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size

        for s in range(Eval.shape[0]):
            ax.plot(Block_size, Eval[:, s], marker=marker[s], label=Classifier[s], color='black',
                    linestyle=lines[s])

        ax.set_xticks(Block_size)
        ax.set_xlabel('Block Size', fontsize=12)
        ax.set_ylabel(Terms[m], fontsize=12)

        # Legend inside the overall box
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=10, ncol=len(Classifier),
                           frameon=False)

        # Show only y-axis grid
        ax.grid(axis='y')

        # Remove only top and right box lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Keep left and bottom box lines with custom styling
        ax.spines['left'].set_edgecolor('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['bottom'].set_linewidth(1.5)

        # Apply tight layout to minimize extra space
        plt.tight_layout(pad=1.5)

        # Add an external bounding box
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2.5)  # Thickness of the extra bounding box

        # Adjust subplot layout to make space for legend
        plt.subplots_adjust(top=0.85)  # Move the content slightly downward

        path = "./Results/%s.png" % (Terms[m])
        plt.savefig(path, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

def Plot_Security():
    Terms = ['Transactions per Second ', 'Transaction Latency', ' Security', ' Cost per Transaction']

    Classifier = ['EB', 'HFB', 'PB', 'DAG', 'PRB']

    # Colors for each category
    colors = ['lawngreen', 'hotpink', 'burlywood', 'cyan', 'DarkKhaki']
    for m in range(len(Terms)):
        Eval = np.load('Eval_Security.npy', allow_pickle=True)[m]
        bar_width = 0.15
        plt.figure(figsize=(7, 6))

        x = np.arange(Eval.shape[0])

        # Plot first graph (Algorithms)
        for i in range(Eval.shape[0]):
            bars =  plt.bar(x + i * bar_width, Eval[:, i], width=bar_width, label=Classifier[i], color=colors[i])
            # Add text annotations inside the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height - (0.05 * height),
                         f"{str(np.round(height, 2))}", ha='center', va='top', fontsize=10, color='k', rotation=90,
                         fontweight='bold')


        # Customizations
        plt.xticks(x + bar_width * 2, ['1', '2', '3', '4', '5'], fontsize=10)
        plt.xlabel('Block Size', fontsize=12)
        plt.ylabel(Terms[m], fontsize=12)

        # Remove axes outline
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)


        # Legend
        dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                       in
                       colors]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10, frameon=False,
                   ncol=len(Classifier))



        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        path =  "./Results/%s.png" % (Terms[m])
        plt.savefig(path, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


if __name__ == '__main__':
    # plot_ENC_DEC_CONS_MEM_1()
    # plot_CPA_KPA()
    Plot_Security()