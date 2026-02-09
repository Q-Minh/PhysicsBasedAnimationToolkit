# The idea: tracy provides a way to turn .tracy files into .csv files
#       from tracy/csvexport directory:
#            cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#            cmake --build ./build
#            cd build
#            ./tracy-csvexport -u /path/to/trace.tracy > /path/to/output.csv
# The csv file has (important) columns: name, ns_since_start, exec_time_ns
# We'll sort by ns_since_start, which corresponds exactly to the order of events in the trace.
# If we can identify "Zone Start" events, we can reconstruct the timing of each zone.

import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt 

frame_name = "Physics"

_solver_params = {
    "vbd": {
        "zone_start": "pbat.sim.algorithm.vbd.InitializeSolve",
        "focus": ["pbat.sim.algorithm.vbd.Iterate"]
    },
    "anderson_vbd": {
        "zone_start": "",
        "focus": ""
    },
    "broyden_vbd": {
        "zone_start": "",
        "focus": ""
    },
    "chebyshev_vbd": {
        "zone_start": "",
        "focus": ""
    },
    "newton": {
        "zone_start": "",
        "focus": ""
    },
}

class Node:
    def __init__(self, name, ns_since_start, exec_time_ns):
        self.name = name
        self.ns_since_start = ns_since_start
        self.exec_time_ns = exec_time_ns
        self.ns_end_time = ns_since_start + exec_time_ns
        self.parent = None
        self.children = []

    def hasChildren(self):
        return len(self.children) > 0

def define_args():
    parser = argparse.ArgumentParser(description="Generate plots from a csv extracted from tracy.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output plot file."
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="vbd",
        help=f"Solver type. Available types are {', '.join(_solver_params.keys())}",
        dest="solver",
    )
    return parser.parse_args()


def makeBasicPlot(frame: pd.DataFrame, my_ax, data_label : str, y_label : str, plot_error=False):
    """Create one plot in the figure. Shows the mean, and, if requested, the standard error

    Parameters
    -----------
        result_per_action: np.array
            An array representing all the results for each time step over all the runs
        my_ax: matplotlib.axes.Axes
           The subplot where we'll plot the result
        data_label: str
           The title of the subplot
        y_label: str
           The title of the y-axis
        plot_error: bool
           True if we need to plot the standard error, False otherwise

    """
    # draw the average reward values
    my_ax.plot(frame["mean"], label=data_label)

    if plot_error:
        # standard deviation
        std = frame["std"]
        # standard error
        std_err = std / math.sqrt(len(frame))

        err_plus = frame["mean"] + std_err
        err_minus = frame["mean"] - std_err

        # draw the error bar/area
        my_ax.fill_between(range(0,len(frame)), err_minus, err_plus, alpha=0.4)

    # Add the axes labels
    my_ax.legend()
    my_ax.set_xlabel("frame")
    my_ax.set_ylabel(y_label)


def traceEvents(df):
    events = []
    for index, row in df.iterrows():
        event = Node(name=row["name"], ns_since_start=row["ns_since_start"], exec_time_ns=row["exec_time_ns"])
        for e in reversed(events):
            # Finds earliest parent
            if e.ns_end_time > event.ns_end_time:
                e.children.append(event)
                event.parent = e
                break
        events.append(event)
    return events


def main():
    args = define_args()
    # Read the csv. We only care about the columns name, ns_since_start, exec_time_ns
    df = pd.read_csv(args.input, usecols=["name", "ns_since_start", "exec_time_ns"], dtype={"name": str, "ns_since_start": int, "exec_time_ns": int})

    # Sort the dataframe by ns_since_start
    df = df.sort_values(by="ns_since_start")
    df = df.reset_index(drop=True)

    # Get zone starts
    # zone_starts = df[df["name"] == _solver_params[args.solver]["zone_start"]]
    zone_starts = df[df["name"] == frame_name]
    
    # Stats assembled here
    trace_stat_columns = ["total", "mean", "std"]
    trace_stats = {}
    for focus_property in _solver_params[args.solver]["focus"]:
        trace_stats[focus_property] = []

    for i in range(len(zone_starts)-1):

        df_index = zone_starts.iloc[i].name
        df_index_next = zone_starts.iloc[i+1].name
        df_slice = df[df_index:df_index_next]
        
        for focus_property in _solver_params[args.solver]["focus"]:
            focus_events = df_slice[df_slice["name"] == focus_property]
            focus_total = focus_events["exec_time_ns"].sum()

            # stats include: count, mean, std, min, 25%, 50%, 75%, max.
            focus_stats = focus_events["exec_time_ns"].describe()
            trace_stats[focus_property].append([focus_total, focus_stats["mean"], focus_stats["std"]])

    fig, ax = plt.subplots()

    for focus_property in _solver_params[args.solver]["focus"]:
        trace_stats_df = pd.DataFrame(trace_stats[focus_property], columns=trace_stat_columns)
        # trace_stats_df.to_csv(args.output, index=False)
        
        makeBasicPlot(trace_stats_df, ax, data_label=focus_property, y_label="mean exec time (ns)", plot_error=True)
    fig.savefig(args.output)




if __name__ == "__main__":
    main()
