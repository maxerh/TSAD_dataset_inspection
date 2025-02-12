import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import visualization
import matplotlib.pyplot as plt

import datainspection
import tikz_printer.box_plots as box_plotter
import tikz_printer.min_max_plots as min_max_plotter
from data_loader import DataLoader


def get_dataset_entities():
    """
    Generates a dictionary of dataset entities mapped to their respective entity values.

    The function organizes specific labeled entity names under different dataset keys.
    The datasets and their entities represent machine-specific or process-specific
    terminologies and identifiers. This data can be utilized in various contexts,
    such as data processing pipelines, machine learning datasets, or monitoring systems.

    :return: A dictionary where keys are dataset names (e.g., "SMD", "PSM") and values are
        lists of entity identifiers associated with each dataset.
    :rtype: dict
    """
    entities = {"SMD": [], "PSM": ["psm"], "SWaT": ["swat"], "WADI": ["wadi"], "MSL": [], "SMAP": []}

    #entities["SMD"] += ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]
    #entities["SMD"] += ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9"]
    #entities["SMD"] += ["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
    entities["SMD"] += ["3-11"]
    entities["SMD"] = [f"machine-{e}" for e in entities["SMD"]]

    #entities["MSL"] += ["C-1", "C-2", "D-14", "D-15", "D-16", "F-4", "F-5", "F-7", "F-8"]
    #entities["MSL"] += ["M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7", "P-10", "P-11", "P-14", "P-15"]
    #entities["MSL"] += ["S-2", "T-4", "T-5", "T-8", "T-9", "T-12", "T-13"]
    entities["MSL"] += ["C-1"]

    #entities["SMAP"] += ["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8", "A-9", "B-1"]
    #entities["SMAP"] += ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "D-11"] #, "D-12", "D-13"]
    #entities["SMAP"] += ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13"]
    #entities["SMAP"] += ["F-1", "F-2", "F-3", "G-1", "G-2", "G-3", "G-4", "G-6", "G-7"]
    #entities["SMAP"] += ["P-1", "P-2", "P-3", "P-4", "P-7", "R-1", "S-1", "T-1", "T-2", "T-3"]
    entities["SMAP"] += ["A-2"]
    return entities


def get_acf(args, dataloader, datasets, entities):
    """
    Calculates and visualizes the Auto-Correlation Function (ACF) for specified datasets
    and entities. The ACF calculation helps in understanding the pattern and dependency
    within the dataset entities over different lags. This function supports saving the
    calculated ACF values for further analysis.

    :param args: Configuration arguments that include parameters for ACF computation
        such as `lags` and `save`. These control the number of lags and the
        save option for generated plots or data.
    :type args: argparse.Namespace
    :param dataloader: An object responsible for loading datasets and specific entities'
        data for ACF computation. Must implement `load_dataset` and provide data access
        via the `data` attribute.
    :type dataloader: Custom data loader instance
    :param datasets: List of datasets to compute ACF over. Each item represents a dataset
        name that is processed by the function.
    :type datasets: list[str]
    :param entities: Dictionary mapping dataset names to lists of entities, where each
        entity corresponds to a specific time-series entity whose ACF is to be computed.
    :type entities: dict[str, list[str]]
    :return: None
    """
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
    """
    Detects and analyzes stationary data across datasets and entities utilizing specified
    data inspection techniques (ADF and KPSS). The function iterates through the given
    datasets and entities, loads data for each, and performs stationarity checks per channel.
    Results of both tests are printed for each dataset, entity, and channel.

    :param dataloader: A data loader object that manages the loading of datasets and allows
        access to data for each entity in the given datasets.
    :type dataloader: DataLoader
    :param datasets: A list of dataset identifiers specifying which datasets will be processed.
    :type datasets: list
    :param entities: A dictionary where keys are dataset identifiers and values are lists of
        entities present in the corresponding datasets.
    :type entities: dict
    :return: None
    """
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





def main(args):
    datasets = args.datasets.split("_")
    entities = get_dataset_entities()
    #for k in entities:
    #    if k in datasets:
    #        print(k, ":", len(entities[k]))
    dataloader_train = DataLoader('train')
    dataloader_test = DataLoader('test')

    #get_acf(args, dataloader, datasets, entities)
    #detect_stationary(dataloader, datasets, entities)

    box_plotter.box_plot(dataloader_train, dataloader_test, datasets, entities,
                         exclude_anomaly=True)
    min_max_plotter.print_channel_min_max_plot_tikz(dataloader_train, dataloader_test, datasets, entities,
                                                    exclude_anomaly=True)




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
