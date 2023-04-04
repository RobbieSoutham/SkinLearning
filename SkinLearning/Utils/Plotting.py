from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np

"""
    Plots a bar chart for 100-MAPE of all parameters and overall
    for the given models in the dataframe
"""
def plot_parameter_bars(df, fname=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
    df.plot.bar(ax=ax)
    
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylabel("Architecture", fontsize=30)
    ax.set_xlabel("Accuracy (%)", fontsize=30)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),  prop={'size': 25})
    ax.set_title("Average percent correctness 100-MAPE", size=40, y=1.08)
    ax.set_ylim(60, 100)

    if fname:
        fig.savefig(f"../Results/{fname}", bbox_inches='tight')


"""
    Plots train/validation loss curves for the given models
"""
def print_curves(model_names, train_losses, val_losses, epochs=550, fname=None):
    assert len(model_names) == len(train_losses) == len(val_losses)
    
    num_plots = len(model_names)
    num_cols = 2 if num_plots >= 2 else 1
    num_rows = math.ceil(num_plots / num_cols)
    figsize = (20 * num_cols, 15 * num_rows)
    lines = []
    labels = []
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)
    
    if num_rows == 1 and num_cols == 1:
        axes = [axes]

    for idx, ax in enumerate(axes.flatten()):
        if idx < num_plots:
            model_name = model_names[idx]
            train_loss = train_losses[idx]
            val_loss = val_losses[idx]
            epoch_range = range(0, epochs)

            line1, = ax.plot(epoch_range, train_loss, label='Train')
            line2, = ax.plot(epoch_range, val_loss, label='Validation')

            ax.set_title(model_name, fontsize=40)
            ax.tick_params(axis='both', labelsize=35)
            
                        
            if idx == 0:
                lines.extend([line1, line2])
                labels.extend(['Train', 'Validation'])
                
        else:
            ax.axis('off')

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle('Epoch', fontsize=50, x=0.5, y=0.02, horizontalalignment='center', verticalalignment='bottom')
    fig.supylabel('MAE', fontsize=50, x=0.02, y=0.5, horizontalalignment='left', verticalalignment='center', rotation='vertical')
    fig.legend(lines, labels, prop={'size': 50})
    
    if not os.path.exists("../Results/"):
        os.makedirs("../Results/")
    
    if fname:
        plt.savefig(f"../Results/{fname}.svg", format='svg', bbox_inches='tight')
        
    plt.show()

"""
    Plots and save a given dataframe
"""
def save_df(df, fname):
    fig = plt.figure(figsize=(5,5), frameon=False, constrained_layout=True)
    df = df.astype(float)
    df = df.round(2)

    labels = np.concatenate(([df.index.name], df.columns))
    text = np.hstack((df.index.values.reshape(-1, 1), df.values))
    
    table = plt.table(cellText = text, 
                      colLabels = labels,
                      loc='center',
                     )

    plt.axis('off')
    plt.grid('off')
    
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        
        cell.set_edgecolor("none")

            
        if row %2 != 0:
            cell.set_facecolor('gainsboro')

        

    table.auto_set_font_size(False)
    table.set_fontsize(25)
    table.scale(2.5, 2.5)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.savefig(f'../Results/{fname}.svg', bbox_inches="tight" )
    
    plt.show()