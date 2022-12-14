import os
import matplotlib.pyplot as plt

def plot_edit_distance_approximation(y_pred, y_true, model_name, dataset, percent_rmse):
    
    fig, ax = plt.subplots()
    
    plt.hist2d(y_pred, y_true, bins=150, cmap='Blues')
    plt.plot([0, 1], [0, 1], transform=ax.transAxes, color='red', linewidth=2)
    
    plt.title('Edit distance approximation\n{} %RMSE={:.4f} ({})'.format(model_name, percent_rmse, dataset))
    ax.set(xlabel='Predicted distance', ylabel='Real distance')
   
    filename = 'reports/figures/{}_{}.png'.format(model_name, dataset)
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(filename)