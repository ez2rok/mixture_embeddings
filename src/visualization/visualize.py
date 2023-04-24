import os
import matplotlib.pyplot as plt

def plot_edit_distance_approximation(y_pred, y_true, model_name, dataset, percent_rmse, args, outdir='reports/figures'):
    
    fig, ax = plt.subplots()
    
    plt.hist2d(y_pred, y_true, bins=100, cmap='Blues')
    plt.plot([0, 1], [0, 1], linewidth=2, color='red')
    
    title = 'Edit distance approximation\n{} %RMSE={:.4f} ({})'.format(model_name, percent_rmse, dataset)
    plt.title(title)
    ax.set(xlabel='Predicted distance', ylabel='Real distance', xlim=[0, 0.3], ylim=[0, 0.3])
   
    filename = '{}/{}_{}_{}_{}.png'.format(outdir, model_name.lower(), args.distance, args.embedding_size, dataset)
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(filename)
    return fig