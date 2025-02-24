import numpy as np
from utils.data_utils import *

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
    x_coords = np.append(bin_centers, [0,1])
    x_coords.sort()

    # TikZ header
    tikz_code = [
        "\\begin{figure}",
        "\centering",
        "\\begin{tikzpicture}",
        "    \\begin{axis}[".strip(),
        "        width=\\textwidth,",
        "        height=3cm,",
        "        ybar,",
        "        bar width=0.1cm,",
        "        ymin=0,",
        "        xmin=0,",
        "        xmax=1,",
        "        xlabel={rel. position},",
        "        %ylabel={points},",
        "        xtick={0,1},",
        "        xticklabel={0,1},",
        "        symbolic x coords={%s}," % ",".join([f"{x:.3f}" for x in x_coords]),
        "    ]",
    ]

    # Convert data into TikZ coordinates (bin centers and counts)
    coordinates = " ".join(
        f"({bin_center:.3f}, {count})" for bin_center, count in zip(bin_centers, hist) if count > 0
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


def get_anomaly_lengths(dl, datasets, entities):
    data_dict = load_data(dl, datasets, entities)
    for ds in data_dict.keys():
        lengths = []
        for entity in data_dict[ds].keys():
            labels = np.array(data_dict[ds][entity]['label'])
            segments = find_segments(labels)
            for s in segments: lengths.append(s[1]-s[0])
        plot_anomaly_lengths_tikz(ds, lengths)


def plot_anomaly_lengths_tikz(ds, lengths, bin_count=100):
    """
    Generates a TikZ code histogram for the lengths of anomalous segments.

    :param lengths: List of lengths of anomalous segments.
    :type lengths: List[int]
    :param bin_count: Number of bins for the histogram.
    :type bin_count: int
    """
    # Calculate the histogram data
    counts, bins = np.histogram(lengths, bins=bin_count)

    # Determine bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # TikZ header
    tikz_code = [
        "\\begin{figure}",
        "\centering",
        "\\begin{tikzpicture}",
        "    \\begin{axis}[",
        "        width=\\textwidth,",
        "        height=3cm,",
        "        ybar,",
        "        bar width=0.1cm,",
        "        ymin=0,",
        "        xmin=0,",
        f"        xmax={int(max(bins)) + 1},",
        "        xlabel={Anomaly Segment Length},",
        "        %ylabel={Frequency},",
        "        xticklabel style={/pgf/number format/fixed},",
        "    ]"
    ]

    # Convert the histogram data into TikZ coordinates
    coordinates = " ".join(
        f"({int(bin_center)}, {count})"
        for bin_center, count in zip(bin_centers, counts) if count > 0
    )
    color = ds.lower()


    # Append TikZ histogram data
    tikz_code.append(f"        \\addplot+[ybar, fill={color}, draw={color}, fill opacity=0.4] coordinates {{{coordinates}}};")

    # Close TikZ and axis environments
    tikz_code.extend([
        "    \\end{axis}",
        "\\end{tikzpicture}"
    ])
    tikz_code.extend([
        f"\\caption{{Length of anomalous, continuous segments in the \\acs{{{ds}}} dataset.}}",
        f"\\label{{fig:anomaly_len_{ds.lower()}}}",
        "\\end{figure}"
    ])


    # Print the generated TikZ code
    print("\n".join(tikz_code))

