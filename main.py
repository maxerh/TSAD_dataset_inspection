import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import visualization
import matplotlib.pyplot as plt

import datainspection
import tikz_printer.box_plots as box_plotter
from data_loader import DataLoader



def get_dataset_entities():
    entities = {"SMD": [], "PSM": ["psm"], "SWaT": ["swat"], "WADI": ["wadi"], "MSL": [], "SMAP": []}

    entities["SMD"] += ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]
    entities["SMD"] += ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9"]
    entities["SMD"] += ["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    entities["SMD"] = [f"machine-{e}" for e in entities["SMD"]]

    entities["MSL"] += ["C-1", "C-2", "D-14", "D-15", "D-16", "F-4", "F-5", "F-7", "F-8"]
    entities["MSL"] += ["M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7", "P-10", "P-11", "P-14", "P-15"]
    entities["MSL"] += ["S-2", "T-4", "T-5", "T-8", "T-9", "T-12", "T-13"]

    entities["SMAP"] += ["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8", "A-9", "B-1"]
    entities["SMAP"] += ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "D-11"] #, "D-12", "D-13"]
    entities["SMAP"] += ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13"]
    entities["SMAP"] += ["F-1", "F-2", "F-3", "G-1", "G-2", "G-3", "G-4", "G-6", "G-7"]
    entities["SMAP"] += ["P-1", "P-2", "P-3", "P-4", "P-7", "R-1", "S-1", "T-1", "T-2", "T-3"]

    return entities

def get_acf(args, dataloader, datasets, entities):
    out_dir = "output/acf"
    os.makedirs(out_dir, exist_ok=True)
    visualizer = visualization.VisualizerCorrelation()
    inspector = datainspection.DataInspectorACF(int(args.lags), visualizer)
    entity_to_display = []
    #entity_to_display += ["machine-1-1", "machine-2-7", "machine-3-6", "machine-3-10"]
    entity_to_display += ["machine-1-7", "machine-2-1", "machine-2-3", "machine-3-1","machine-3-3", "machine-3-9", "machine-3-10"]
    #entity_to_display += ["machine-1-1", "machine-2-7", "machine-3-10"]
    #entity_to_display += ["psm", "swat", "wadi"]
    #entity_to_display += ["C-1"]               # MSL
    #entity_to_display += ["A-9", "G-7", "R-1"] # SMAP
    for d in datasets:
        dataset_acf = np.zeros([int(args.lags)+1, len(entities[d])])
        for i, entity in enumerate(entities[d]):
            if entity not in entity_to_display:
                continue
            dataloader.load_dataset(d, entity)
            acf_array = inspector.plot_mean_acf(dataloader.data, f"{d}: {entity}")
            dataset_acf[:, i] = acf_array
            if args.save:
                np.save(f"{out_dir}/acf_{d}_{entity}",acf_array)

        mean_acf_dataset = np.mean(dataset_acf, axis=1)
        std_acf_dataset = np.std(dataset_acf, axis=1)
        #min_acf_dataset = np.min(dataset_acf, axis=1)
        #max_acf_dataset = np.max(dataset_acf, axis=1)
        #plt.clf()
        #plt.fill_between(np.arange(0,int(args.lags)+1),
        #                 mean_acf_dataset-std_acf_dataset,
        #                 mean_acf_dataset+std_acf_dataset, alpha=0.2)
        #plt.fill_between(np.arange(0,int(args.lags)+1),
        #                 min_acf_dataset,
        #                 max_acf_dataset, alpha=0.2)
        #plt.plot(mean_acf_dataset)
        #plt.show()
        if args.save:
            np.save(f"{out_dir}/acf_{d}_mean", mean_acf_dataset)
            np.save(f"{out_dir}/acf_{d}_std", std_acf_dataset)

    visualizer.plot_legend()
    inspector.show_correlations()

def detect_stationary(dataloader, datasets, entities):
    #out_dir = "output/stationary"
    #os.makedirs(out_dir, exist_ok=True)
    inspector = datainspection.DataInspectorStationary()
    for d in datasets:
        #dataset_stationary = np.zeros([int(args.lags)+1, len(entities[d])])
        for i, entity in enumerate(entities[d]):
            dataloader.load_dataset(d, entity)
            for channel in range(dataloader.data.shape[1]):
                if min(dataloader.data[:,channel]) == max(dataloader.data[:,channel]):
                    continue
                adf_out = inspector.print_results(dataloader.data[:,channel], "adf")
                kpss_out = inspector.print_results(dataloader.data[:,channel], "kpss")
                print("\n", d, entity, channel, "\n", pd.concat([adf_out, kpss_out], axis=1))

def box_plot(dataloader, datasets, entities):
    bp_dict = dict()
    for d in datasets:
        bp_dict[d] = dict()
        for i, entity in enumerate(entities[d]):
            dataloader.load_dataset(d, entity)
            data = dataloader.data
            if dataloader.label is not None:
                label = dataloader.label
                data = data[label==0]   # just nominal data when having test data.
            bp = plt.boxplot(data)
            bp_data = get_box_plot_data(bp, data.shape[-1])
            bp_dict[d][entity] = bp_data
    return bp_dict




def get_box_plot_data(bp, channels):
    rows_list = []
    for i in range(channels):
        dict1 = {}
        dict1['label'] = i
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)

def main(args):
    datasets = args.datasets.split("_")
    entities = get_dataset_entities()
    #for k in entities:
    #    if k in datasets:
    #        print(k, ":", len(entities[k]))
    dataloader = DataLoader('train')

    #get_acf(args, dataloader, datasets, entities)
    #detect_stationary(dataloader, datasets, entities)
    bp_train = box_plot(dataloader, datasets, entities)
    dataloader = DataLoader('test')
    bp_test = box_plot(dataloader, datasets, entities)
    box_plotter.print_box_plot_tikz(bp_train, bp_test)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--datasets",
                        help="multiple datasets are separated by '_' (SMD_MSL_SMAP_PSM_SWaT_WADI)",
                        default="SMD",
                        )
    parser.add_argument("-l", "--lags",
                        default=1024,
                        )
    parser.add_argument("-s", "--save",
                        default=False,
                        )
    args = parser.parse_args()
    main(args)
