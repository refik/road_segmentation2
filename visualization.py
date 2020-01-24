import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_epoch_stats(epoch_stats):
    """Return loss, f1, accuracy plots for training and validation."""
    
    # Formatting data for the plot
    stats_df = (pd.DataFrame(epoch_stats)
                .melt(id_vars=['phase', 'epoch'], 
                      value_vars=['loss', 'f1', 'acc']))
    
    # First epochs may be too bad and break the scales of plot
    stats_df = stats_df.query('epoch > 2')
    
    # Prepare the facet grid
    g = sns.FacetGrid(stats_df, row="variable", hue="phase", 
                      sharey=False, aspect=3)
    
    # Plot and return
    return g.map(plt.plot, "epoch", "value").add_legend()
