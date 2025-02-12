import numpy as np
import matplotlib.pyplot as plt


def load_data(dl, datasets, entities):
    dict1 = {}
    for d in datasets:
        dict1[d] = {}
        for i, entity in enumerate(entities[d]):
            dl.load_dataset(d, entity)
            data = dl.data
            label = dl.label
            # Handle edge case: If no labels available
            if label is not None and len(label) > 0:
                total_labels = len(label)
                # Compute relative positions
                relative_positions = [idx / total_labels for idx in range(total_labels)]

                # Store relevant info in dictionary
                dict1[d][entity] = {
                    "data": data,
                    "label": label,
                    "relative_positions": relative_positions,
                }
            else:
                # Handle case with missing labels
                dict1[d][entity] = {
                    "data": data,
                    "label": None,
                    "relative_positions": None,
                }

    return dict1

def get_anomaly_distribution(dl, datasets, entities):
    data_dict = load_data(dl, datasets, entities)
    for ds in data_dict.keys():
        rel_positions = []
        for entity in data_dict[ds].keys():
            rel_pos = np.array(data_dict[ds][entity]['relative_positions'])
            label = np.array(data_dict[ds][entity]['label'])
            rel_positions.append(rel_pos[label==1])
        all_rel = np.concatenate(rel_positions)
        hist = np.histogram(all_rel, bins=100)
        d = plot_histogram_tikz(hist, ds)
        print(d)


def plot_histogram_tikz(data, ds):
    """
    Generates TikZ code for a histogram with bars colored according to the input color.

    Args:
        data (list): List of bin heights for the histogram.
        ds (str): The dataset name.

    Returns:
        str: TikZ code as a string.
    """
    color = ds.lower()
    hist, bins = data
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Average of bin edges

    # TikZ header
    tikz_code = [
        "\\begin{figure}",
        "\centering",
        "\\begin{tikzpicture}",
        "    \\begin{axis}[".strip(),
        "        width=\\textwidth,",
        "        height=5cm,",
        "        ybar,",
        "        bar width=0.1cm,",
        "        ymin=0,",
        "        xmin=0,",
        "        xmax=1,",
        "        xlabel={rel. position},",
        "        ylabel={number of anomalies},",
        "        xtick={0,0.25,0.50,0.75,1},",
        "        xticklabel={0,0.25,0.50,0.75,1},",
        "        symbolic x coords={%s}," % ",".join([f"{x:.2f}" for x in bin_centers]),
        "    ]",
    ]

    # Convert data into TikZ coordinates (bin centers and counts)
    coordinates = " ".join(
        f"({bin_center:.2f}, {count})" for bin_center, count in zip(bin_centers, hist)
    )
    # Append the histogram plot with the correct color
    tikz_code.append(f"        \\addplot+[ybar, fill={color}, draw={color}, fill opacity=0.4] coordinates {{{coordinates}}};")

    # Close TikZ and axis environment
    tikz_code.extend(["    \\end{axis}", "\\end{tikzpicture}",])
    tikz_code.extend([
        f"\\caption{{Relative anomaly position in the \\acs{{{ds}}} dataset.}}",
        f"\\label{{fig:anomaly_dist_{ds.lower()}}}",
        "\\end{figure}"
    ])

    return "\n".join(tikz_code)

