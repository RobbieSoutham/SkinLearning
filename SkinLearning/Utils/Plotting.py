from matplotlib import pyplot as plt

def plotParameterBars(df, file_name=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
    df.plot.bar(ax=ax)
    
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylabel("Architecture", fontsize=30)
    ax.set_xlabel("Accuracy (%)", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),  prop={'size': 25})
    ax.set_title("Average percent correctness 100-MAPE", size=40, y=1.08)
    ax.set_ylim(60, 100)

    if file_name:
        fig.savefig(f"../Results/{file_name}", bbox_inches='tight')
