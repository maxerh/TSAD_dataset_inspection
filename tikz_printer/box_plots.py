
def print_box_plot_tikz(bp_train, bp_test):
    for ds in bp_train.keys():
        for entity in bp_train[ds].keys():
            for (channel_train, values_train), (channel_test, values_test) in zip(bp_train[ds][entity].iterrows(), bp_test[ds][entity].iterrows()):
                if channel_train not in [0,1,3,5,6,20,22,23]: continue
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
