import numpy as np
#import matplotlib as mpl
#mpl.use('pgf')
#mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

class Color:
    #white background for screenshots
    background = "#ffffff"
    text = "#000000"
    signal1 = "#77ba9b"
    signal2 = "#82c19d"
    signal3 = "#008e8c"
    zero = "#fb5f5f"

class Visualizer_base:
    def __init__(self):
        super().__init__()
        self.fig = plt.figure(figsize=(25, 75), facecolor=Color.background)
        self.ax_dict = {}

    @staticmethod
    def consecutive(data, stepsize: int = 1):
        """
        Check for consecutive numbers in a 1d array

        :param data: data array
        :param stepsize: The step size to check
        :return: list of ndarrays with consecutive numbers
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def save_pgf(self, name):
        plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'  # load times roman font
        plt.rcParams['font.family'] = 'serif'  # use serif font as default
        plt.rcParams['text.usetex'] = True  # enable LaTeX rendering globally
        self.fig.savefig(f'{name}.pgf', format='pgf')

    def save_tikz(self, name):
        import tikzplotlib
        tikzplotlib.save(f"{name}.tex")

    def save_fig(self, name):
        self.fig.savefig(f"{name}.png", facecolor=self.fig.get_facecolor(), edgecolor='none')

    def save_np(self, name, data, s, c):
        file = f"{name}.npy"
        with open(file, 'wb') as f:
            for k in self.ax_dict.keys():
                ts = data[k][s:s+1024, c[k]]
                np.save(f, ts)

    def show_plot(self):
        """Show plot"""
        plt.show()

    def draw_plot(self):
        """Draw the figure"""
        plt.draw()
        plt.pause(0.01)

    def close_plot(self):
        """Close the plot"""
        plt.close(self.fig)

    def clear_axes(self):
        """Clear the axes"""
        for k, ax in self.ax_dict.items():
            ax.clear()

    def plot_legend(self):
        for _, ax in self.ax_dict.items():
            ax.legend(facecolor=Color.background, edgecolor=Color.text, labelcolor=Color.text)


class VisualizerCorrelation(Visualizer_base):
    def __init__(self, n_plots=1):
        super().__init__()
        self.create_figure_subplots(n_plots)

    def create_figure_subplots(self, n_plots: int):
        """
        Creating a figure with subplots
        :param n_plots: number of plots that should be plotted in the figure
        """
        if n_plots > 1:
            n_rows = 4
            n_cols = n_plots//n_rows
        else:
            n_rows = 1
            n_cols = 1
        for i in range(n_plots):
            self.ax_dict[f"entity_{i}"] = self.fig.add_subplot(n_rows, n_cols, i+1)

        for _, ax in self.ax_dict.items():
            ax.set_facecolor(Color.background)
            ax.set_xticks([64,128,256, 512, 1024])
            #ax.set_yticks([])
            ax.spines['bottom'].set_color(Color.text)
            ax.spines['top'].set_color(Color.text)
            ax.spines['left'].set_color(Color.text)
            ax.spines['right'].set_color(Color.text)


    def plot_mean_acf(self, i, data, legend_entry):
        ax = self.ax_dict[f"entity_{i}"]
        ax.plot(data, label=legend_entry)
