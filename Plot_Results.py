from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix




def ROC_curve():
    lw = 2
    cls = ['SVM', 'KNN', 'DNN', 'STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN']
    # Create a figure and axis
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Roc Curve')
    Actual = np.load('Authendication_Target.npy', allow_pickle=True).astype('int')
    per = round(Actual.shape[0] * 0.75)
    Actual = Actual[per:, :]
    colors = cycle(["#fe2f4a", "#8b88f8", "#fc824a", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()




def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['CWO-STA-ABi-RNN', 'TOT-STA-ABi-RNN', 'WOA-STA-ABi-RNN', 'SOA-STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN']
    color = ['Deeppink', 'm', 'r', 'y', 'k']
    markerfacecolor = ['red', 'green', 'cyan', 'magenta', 'black']
    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("Statistical", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('--------------------------------------------- Statistical Analysis ----------------------------------------')
    print(Table)

    fig = plt.figure(facecolor='#FFF1F1')
    fig.canvas.manager.set_window_title('Convergence')
    iteration = np.arange(conv.shape[1])
    for m in range(conv.shape[0]):
        plt.plot(iteration, conv[m, :], color=color[m], linewidth=3, marker='*',
                 markerfacecolor=markerfacecolor[m], markersize=12, label=Algorithm[m])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path = "./Results/Conv.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_table():
    Eval = np.load('Eval_All_act.npy', allow_pickle=True)
    Graph_Term = np.array([0]).astype(int)
    Comparison = ['Algorithm', 'Classifier']
    Networks = [['Activation Function', 'CWO-STA-ABi-RNN', 'TOT-STA-ABi-RNN', 'WOA-STA-ABi-RNN', 'SOA-STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN'],
                ['Activation Function', 'SVM', 'KNN', 'DNN', 'STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN']]
    variation = ['Linear', 'Relu', 'Tanh', 'Sigmoid', 'Softmax']
    value = Eval[:, :, 4:]
    for m in range(len(Comparison)):
        Table = PrettyTable()
        Table.add_column(Networks[m][0], variation)
        for j in range(len(Networks[m]) - 1):
            if m == 1:  # for Network
                k = len(Networks[m]) + j - 1
            else:
                k = j
            Table.add_column(Networks[m][j + 1],
                             [item for sublist in value[:, k, Graph_Term] for item in sublist])
        print(
            '----------------------------------- Activation Function  -' +
            Comparison[m] + ' Comparison - Accuracy  -------------------------------')
        print(Table)



def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.17, 0.8, 0.8])

    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))

    # Change colormap to 'coolwarm', 'Blues', 'viridis', or any other suitable map
    sns.heatmap(cm, annot=True, fmt='g', cmap="coolwarm", ax=ax)

    ax.xaxis.set_ticklabels(['Non Authenticated', 'Authenticated'])
    ax.yaxis.set_ticklabels(['Non Authenticated', 'Authenticated'])

    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    path = "./Results/Confusion.png"
    plt.savefig(path, dpi=300)  # Increase resolution
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Results():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score', 'MCC',
             'FOR', 'pt', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR +', 'LR -', 'DOR', 'Prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    Eval = np.load('Eval_Hidden.npy', allow_pickle=True)

    for m in range(len(Graph_Term)):
        Graph = Eval[:, :, Graph_Term[m] + 4]

        fig = plt.figure(figsize=(12, 6))
        fig.canvas.manager.set_window_title('Steps Per Epochs - Method')
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

        bar_width = 0.2  # Bar width
        Algo = ['CWO-STA-ABi-RNN', 'TOT-STA-ABi-RNN', 'WOA-STA-ABi-RNN', 'SOA-STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN']
        colors_alg = ['#D2B48C', '#7E1E9C', '#00FFFF', '#FF00FF', '#FAC205']

        # Introduce spacing by increasing X values
        num_groups = Graph.shape[0]
        X = np.arange(num_groups) * (bar_width * 6)  # Adding space after each group

        for j in range(5):  # Loop over methods
            bars = ax.bar(X + j * bar_width, Graph[:, 5 + j], color=colors_alg[j], width=bar_width, label=Algo[j])

            # Add rounded integer values on top of bars
            for bar in bars:
                height = int(round(bar.get_height()))  # Convert to integer
                ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5, f'{height}',
                        ha='center', va='center', fontsize=10, color="black", fontweight="bold",
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.xticks(X + (2 * bar_width), ['100', '200', '300', '400', '500'])  # Correct x-tick alignment
        plt.xlabel('Hidden Neuron Count')
        plt.ylabel(Terms[Graph_Term[m]])

        # Remove left and top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
        Path = f"./Results/{Terms[Graph_Term[m]]}_ALG.png"
        plt.savefig(Path)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        fig = plt.figure(figsize=(12, 6))
        fig.canvas.manager.set_window_title('Steps Per Epochs - Method')
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

        bar_width = 0.2  # Bar width
        methods = ['SVM', 'KNN', 'DNN', 'STA-ABi-RNN', 'UAV-SOA-STA-ABi-RNN']
        colors_met = ['#DC143C', '#AAFF32', '#0000FF', '#9A0EEA', '#FAC205']

        # Introduce spacing by increasing X values
        num_groups = Graph.shape[0]
        X = np.arange(num_groups) * (bar_width * 6)  # Adding space after each group

        for j in range(5):  # Loop over methods
            bars = ax.bar(X + j * bar_width, Graph[:, 5 + j], color=colors_met[j], width=bar_width, label=methods[j])

            # Add rounded integer values on top of bars
            for bar in bars:
                height = int(round(bar.get_height()))  # Convert to integer
                ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5, f'{height}',
                        ha='center', va='center', fontsize=10, color="black", fontweight="bold",
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.xticks(X + (2 * bar_width), ['100', '200', '300', '400', '500'])  # Correct x-tick alignment
        plt.xlabel('Hidden Neuron Count')
        plt.ylabel(Terms[Graph_Term[m]])

        # Remove left and top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
        Path = f"./Results/{Terms[Graph_Term[m]]}_Method.png"
        plt.savefig(Path)
        plt.show(block=False)
        plt.pause(2)
        plt.close()


if __name__ == '__main__':
    ROC_curve()
    plot_results_conv()
    Plot_table()
    Plot_Confusion()
    Plot_Results()
