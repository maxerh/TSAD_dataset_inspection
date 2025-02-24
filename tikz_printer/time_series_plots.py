from utils.data_utils import *

def ts_plot(dl_train, dl_test, datasets, entities):
    """
    Generate a TikZpicture plot for the selected channels of the dataset stored in dl_train.data.
    The output represents a text-based TikZ plot for command-line usage.
    """

    data_dict = load_data(dl_train, datasets, entities)["data"]
    # Generate addplot for each selected channel
    for ds in datasets:
        for entity in entities[ds]:
            if ds == "PSM": selected_channels = [0, 1, 3, 5, 6, 20, 22, 23]
            if ds == "SWaT": selected_channels = [0, 1, 2, 4, 7, 8, 11, 17]
            if ds == "WADI": selected_channels = [0, 1, 5, 9, 20, 29, 31, 42]
            if ds == "SMAP": selected_channels = [0, 1, 2]
            if ds == "MSL": selected_channels = [0, 1, 2]
            if ds == "SMD": selected_channels = [0, 4, 5, 6, 13, 14, 15, 20]
            min_value = 75

            # Initialize the TikZpicture output format
            tikz_output = []
            tikz_output.append("\\begin{figure}")
            tikz_output.append("\centering")
            tikz_output.append("\\begin{tikzpicture}")
            tikz_output.append("  \\begin{axis}")
            tikz_output.append("    [")
            tikz_output.append("    width=\\textwidth,")
            tikz_output.append("    height=5cm,")
            tikz_output.append(f"    xmin={min_value-5},")
            tikz_output.append(f"    xmax={min_value+105},")
            tikz_output.append("    xlabel={time step},")
            tikz_output.append("    ylabel=\\empty,")
            tikz_output.append("    yticklabel=\\empty,")
            #tikz_output.append("    legend pos=north west,")
            tikz_output.append("    ]")
            data = data_dict[ds][entity]["data"]
            vertical_offset = 2
            for idx, channel in enumerate(selected_channels):
                # Extract values for the channel
                values = data[min_value:min_value+100, channel]

                # Add TikZ formatted data for the channel
                tikz_output.append(f"    \\addplot[{ds.lower()}, thick] coordinates {{")
                for i, value in enumerate(values):
                    tikz_output.append(f"      ({min_value+i}, {value - idx * vertical_offset})")
                tikz_output.append(f"    }};")
                #tikz_output.append(f"    \\addlegendentry{{Channel {channel}}}")

        # Close TikZpicture formatting
        tikz_output.append("  \\end{axis}")
        tikz_output.append("\\end{tikzpicture}")

        tikz_output.append(f"\caption{{Segment of the first {len(selected_channels)} channels (top to bottom) of \\ac{{{ds}}}.}}")
        tikz_output.append(f"\label{{fig:ts_example_{ds.lower()}}}")
        tikz_output.append("\\end{figure}")

    # Print the output to command line
    print("\n".join(tikz_output))