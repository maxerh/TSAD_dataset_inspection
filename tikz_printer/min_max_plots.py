from utils.data_utils import *


def print_channel_min_max_plot_tikz(dl_train, dl_test, datasets, entities, exclude_anomaly=False):
    train = load_data(dl_train, datasets, entities, exclude_anomaly)
    test = load_data(dl_test, datasets, entities, exclude_anomaly)

    for ds in train.keys():
        color = ds.lower()
        for entity in train[ds].keys():
            tikz_code = [
                "\\begin{figure}",
                "\centering",
                "\\begin{tikzpicture}",
                "    \\begin{axis}[".strip(),
                "        ymin=-1.5, ymax=1.5,",  # Set y-axis limits (adjust as needed)
                "        xmin=-1, xmax=100,",
                "        xlabel={channel}, ylabel={value},",
                "        height=5cm,",
                "        width=\\textwidth,",
                "        legend style={at={(0.5,1.1)}, anchor=south, legend columns=2},",  # Position legend above
                "        legend cell align={left},",
                "        legend entries={ Training data \quad \quad, Testing data},",
                "        ]",
                f"        \\addlegendimage{{color={{{ds.lower()}}}, very thick}};",
                f"        \\addlegendimage{{color={{{ds.lower()}}}, very thick, opacity=0.4}};",
                " "
            ]
            for channel, (train_min, train_max, test_min, test_max) in enumerate(
                    zip(
                        train[ds][entity]["min"],
                        train[ds][entity]["max"],
                        test[ds][entity]["min"],
                        test[ds][entity]["max"]
                    )
            ):
                if train_min == train_max:
                    tikz_code.append(
                        f"        \\addplot[color={color}, mark=diamond*, mark options={{fill={color}}}] coordinates {{"
                        f"({channel-0.15}, {train_min})}};"
                    )
                else:
                    # Add a vertical line for training data
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick] coordinates {{"
                        f"({channel-0.15}, {train_min}) ({channel-0.15}, {train_max})}};"
                    )

                # If test_min == test_max, use a diamond for testing data
                if test_min == test_max:
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick, opacity=0.4, mark=diamond*, mark options={{fill={color}}}] coordinates {{"
                        f"({channel+0.15}, {test_min})}};"
                    )
                else:
                    # Add a vertical line for testing data
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick, opacity=0.4] coordinates {{"
                        f"({channel+0.15}, {test_min}) ({channel+0.15}, {test_max})}};"
                    )

            # Close the axis and tikzpicture environment
            tikz_code.extend(["    \\end{axis}", "\\end{tikzpicture}"])
            tikz_code.extend([
                f"\\caption{{Channel wise data range of nominal data from entity \\texttt{{{entity}}} of the \\acs{{{ds}}} dataset. Diamonds represent constant values.}}",
                f"\\label{{fig:data_range_{ds.lower()}}}",
                "\\end{figure}"
            ])

            # Print or save the TikZ code
            print("\n".join(tikz_code))
            print("\n% ----------------------------------------------------\n")

def print_channel_mean_std_tikz(dl_train, dl_test, datasets, entities, exclude_anomaly=False):
    train = load_data(dl_train, datasets, entities, exclude_anomaly)
    test = load_data(dl_test, datasets, entities, exclude_anomaly)

    for ds in train.keys():
        color = ds.lower()
        for entity in train[ds].keys():
            tikz_code = [
                "\\begin{figure}",
                "\centering",
                "\\begin{tikzpicture}",
                "    \\begin{axis}[".strip(),
                "        ymin=-1.5, ymax=1.5,",  # Set y-axis limits (adjust as needed)
                f"        xmin=-1, xmax=100,",
                "        xlabel={channel}, ylabel={value},",
                "        height=5cm,",
                "        width=\\textwidth,",
                "        legend style={at={(0.5,1.1)}, anchor=south, legend columns=2},",  # Position legend above
                "        legend cell align={left},",
                "        legend entries={ Training data \quad \quad, Testing data},",
                "        ]",
                f"        \\addlegendimage{{color={{{ds.lower()}}}, very thick}};",
                f"        \\addlegendimage{{color={{{ds.lower()}}}, very thick, opacity=0.4}};",
                " "
            ]
            for channel, (train_mean, train_std, test_mean, test_std) in enumerate(
                    zip(
                        train[ds][entity]["mean"],
                        train[ds][entity]["std"],
                        test[ds][entity]["mean"],
                        test[ds][entity]["std"]
                    )
            ):
                if train_std == 0:
                    tikz_code.append(
                        f"        \\addplot[color={color}, mark=diamond*, mark options={{fill={color}}}] coordinates {{"
                        f"({channel-0.15}, {train_mean})}};"
                    )
                else:
                    # Add a vertical line for training data
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick] coordinates {{"
                        f"({channel-0.15}, {train_mean-train_std}) ({channel-0.15}, {train_mean+train_std})}};"
                    )

                if test_std == 0:
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick, opacity=0.4, mark=diamond*, mark options={{fill={color}}}] coordinates {{"
                        f"({channel+0.15}, {test_mean})}};"
                    )
                else:
                    # Add a vertical line for testing data
                    tikz_code.append(
                        f"        \\addplot[color={color}, very thick, opacity=0.4] coordinates {{"
                        f"({channel+0.15}, {test_mean-test_std}) ({channel+0.15}, {test_mean+test_std})}};"
                    )

            # Close the axis and tikzpicture environment
            tikz_code.extend(["    \\end{axis}", "\\end{tikzpicture}"])
            tikz_code.extend([
                f"\\caption{{Channel wise data distribution of nominal data from entity \\texttt{{{entity}}} of the \\acs{{{ds}}} dataset. Diamonds represent constant values.}}",
                f"\\label{{fig:data_dist_{ds.lower()}}}",
                "\\end{figure}"
            ])

            # Print or save the TikZ code
            print("\n".join(tikz_code))
            print("\n% ----------------------------------------------------\n")