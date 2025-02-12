import pandas as pd
import matplotlib.pyplot as plt

def box_plot(dl_train, dl_test, datasets, entities, exclude_anomaly=False):
    bp_train = plt_box_plot(dl_train, datasets, entities, exclude_anomaly)
    bp_test = plt_box_plot(dl_test, datasets, entities, exclude_anomaly)
    print_box_plot_tikz(bp_train, bp_test)

def plt_box_plot(dataloader, datasets, entities, exclude_anomaly):
    """
    Generates box plot data for given datasets and entities using data loaded
    from the provided dataloader. The function loops through the specified datasets
    and entities, loads the required data, applies filtering if applicable, and
    calculates the box plot data.

    It uses matplotlib to create box plots for data visualization which helps in
    understanding the distribution of data. The resulting dictionary maps datasets
    and respective entities to their computed box plot data.

    :param dataloader: Object responsible for loading datasets and their corresponding entities.
    :type dataloader: Any

    :param datasets: List of dataset names to be processed.
    :type datasets: List[str]

    :param entities: Dictionary where keys are dataset names and values are
        lists of entities to be processed for the respective datasets.
    :type entities: Dict[str, List[Any]]

    :return: A dictionary where keys are dataset names, and values are
        dictionaries mapping each entity to its corresponding box plot data.
    :return type: Dict[str, Dict[Any, Any]]
    """
    bp_dict = dict()
    for d in datasets:
        bp_dict[d] = dict()
        for i, entity in enumerate(entities[d]):
            dataloader.load_dataset(d, entity)
            data = dataloader.data
            if dataloader.label is not None:
                label = dataloader.label
                if exclude_anomaly:
                    data = data[label==0]   # just nominal data when having test data.
            bp = plt.boxplot(data)
            bp_data = get_box_plot_data(bp, data.shape[-1])
            bp_dict[d][entity] = bp_data
    return bp_dict


def get_box_plot_data(bp, channels):
    """
    Extracts and formats box plot statistical data into a pandas DataFrame.

    This function takes box plot components provided by the *bp* parameter and
    retrieves statistical data such as whiskers, quartiles, and the median for
    each of the specified *channels*. The extracted data is organized into a
    dictionary and subsequently converted into a pandas DataFrame, where each
    row represents the data for an individual channel.

    :param bp: Dictionary-like object containing box plot elements. It is
        assumed to have 'whiskers', 'boxes', and 'medians' attributes, which are
        used to extract statistical data.
    :type bp: dict-like
    :param channels: Number of channels for which box plot data is available.
    :type channels: int
    :return: A pandas DataFrame, where each row contains the statistical data for
        a channel, including 'label', 'lower_whisker', 'lower_quartile', 'median',
        'upper_quartile', and 'upper_whisker'.
    :rtype: pandas.DataFrame
    """
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

def print_box_plot_tikz(bp_train, bp_test):
    for ds in bp_train.keys():
        for entity in bp_train[ds].keys():
            for (channel_train, values_train), (channel_test, values_test) in zip(bp_train[ds][entity].iterrows(), bp_test[ds][entity].iterrows()):
                #if channel_train not in [0,1,3,5,6,20,22,23]: continue # PSM
                #if channel_train not in [0,1,2,4,7,8,11,17]: continue # SWaT
                #if channel_train not in [0,1,5,9,20,29,31,42]: continue # WADI
                #if channel_train not in [0]: continue # MSL/SMAP
                if channel_train not in [0,4,5,6,13,14,15,20]: continue # SMD
                print_addplot_train_test(values_train, values_test)


def print_addplot_train_test(train, test):
    print_addplot_entry(train)
    print_addplot_entry(test)


def print_addplot_entry(data):
    median = data.get('median', None)
    q1 = data.get('lower_quartile', None)
    q3 = data.get('upper_quartile', None)
    whisker_low = data.get('lower_whisker', None)
    whisker_high = data.get('upper_whisker', None)

    tikz_entry = (
        f"\\addplot+[\n"
        f"    boxplot prepared={{\n"
        f"        median={median},\n"
        f"        lower quartile={q1},\n"
        f"        upper quartile={q3},\n"
        f"        lower whisker={whisker_low},\n"
        f"        upper whisker={whisker_high}\n"
        f"    }}\n"
        f"] coordinates {{}};"
    )
    # Print the boxplot entry
    print(tikz_entry)

