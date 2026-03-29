# The idea: tracy provides a way to turn .tracy files into .csv files
#       from tracy/csvexport directory:
#            cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#            cmake --build ./build
#            cd build
#            ./tracy-csvexport -u /path/to/trace.tracy > /path/to/output.csv
# The csv file has (important) columns: name, ns_since_start, exec_time_ns
# Note that exec_time_ns also contains exec time of children scopes! (ie NOT self time)
# We'll sort by ns_since_start, which corresponds exactly to the order of events in the trace.
# If we can identify "Zone Start" events, we can reconstruct the timing of each zone.

import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt 

_solver_params = {
    "vbd": {
        "focus": ["pbat.sim.algorithm.vbd.Iterate"]
    },
    "anderson_vbd": {
        "focus": ["pbat.sim.algorithm.vbd.Anderson.Iterate"]
    },
    "broyden_vbd": {
        "focus": ["pbat.sim.algorithm.vbd.Broyden.Iterate"]
    },
    "chebyshev_vbd": {
        "focus": ["pbat.sim.algorithm.vbd.Chebyshev.Iterate"]
    },
    "newton": {
        "focus": ["pbat.sim.algorithm.vbd.Newton.Iterate"]
    },
}

class Node:
    # Simple tree node
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
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file.",
        dest="input",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output plot file.",
        dest="output",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="vbd",
        help=f"Solver type. Available types are {', '.join(_solver_params.keys())}",
        dest="solver",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Simulation frame to analyze."
    )
    parser.add_argument(
        "--frame_set_name",
        type=str,
        default="Physics",
        help="Name of the simulation frames to analyze.",
        dest="frame_name",
    )
    parser.add_argument(
        "--path_conv_values",
        type=str,
        default="",
        help="Path to the convergence values file."
    )
    return parser.parse_args()


def makeBasicPlot(df: pd.DataFrame, my_ax, data_label : str, y_label : str, plot_error=False):
    """Create one plot in the figure. Shows the mean, and, if requested, the standard error

    Parameters
    -----------
        df: pd.DataFrame
           Data for one frame in the simulation
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
    my_ax.plot(df["mean"], label=data_label)

    if plot_error:
        # standard deviation
        std = df["std"]
        # standard error
        std_err = std / math.sqrt(len(df))

        err_plus = df["mean"] + std_err
        err_minus = df["mean"] - std_err

        # draw the error bar/area
        my_ax.fill_between(range(0,len(df)), err_minus, err_plus, alpha=0.4)

    # Add the axes labels
    my_ax.legend()
    my_ax.set_xlabel("frame")
    my_ax.set_ylabel(y_label)


def traceEvents(df: pd.DataFrame) -> list[Node]:
    """
        Create a forest of events from the dataframe, following tracy scopes.
    """
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

    #pie_chart(args, df)
    convergence_graph(args, df)
    


def pie_chart(args, df):
    """
        Create a pie chart from one frame in the data. 
        Returns a chart containing all leaf calls to scopes, 
        as a percentage of the total execution time of that frame.
    """

    zone_starts = df[df["name"] == args.frame_name]

    if args.frame >= len(zone_starts)-1:
        raise ValueError(f"Frame index {args.frame} out of bounds for range {len(zone_starts)-1}")

    df_index = zone_starts.iloc[args.frame].name
    df_index_next = zone_starts.iloc[args.frame+1].name
    df_slice = df[df_index:df_index_next]

    events = traceEvents(df_slice)
    leaves = []
    for e in events:
        if e.hasChildren():
            continue
        leaves.append(e.name)
    
    # remove duplicates
    leaves = sorted(list(set(leaves)))
    short_names = [e.split(".")[-1] for e in leaves]
    sums = []

    for focus_property in leaves:
        focus_events = df_slice[df_slice["name"] == focus_property]
        focus_total = focus_events["exec_time_ns"].sum()
        sums.append(focus_total)

    fig, ax = plt.subplots()

    # Put pie chart on the left
    wedges, _ = ax.pie(sums, radius=1)

    # move/shrink the pie axes to the left (left, bottom, width, height)
    ax.set_position([0.05, 0.1, 0.6, 0.6])

    ax.legend(wedges, short_names,
              title="Profiles",
              loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    fig.savefig(args.output)


def graph(args, df):
    """ Create a graph following the focused scope over all frames. 
        Outputs a plot of execution time with respect to the frame number
    """
    zone_starts = df[df["name"] == args.frame_name]

    # Stats assembled here
    trace_stat_columns = ["total", "mean", "std"]
    trace_stats = {}
    for focus_property in _solver_params[args.solver]["focus"]:
        trace_stats[focus_property] = []

    for i in range(len(zone_starts)-1):

        df_index = zone_starts.iloc[i].name
        df_index_next = zone_starts.iloc[i+1].name
        df_slice = df[df_index:df_index_next]

        events = traceEvents(df_slice)
        leaves = {}
        for e in events:
            if e.hasChildren():
                continue
            leaves[e.name] = []

        for focus_property in leaves:
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


def convergence_graph(args, df):
    """ Create a graph showing the convergence of execution times over frames.
    """
    zone_starts = df[df["name"] == args.frame_name]
    
    conv_values = []
    solver_names = []

    with open(args.path_conv_values, "r") as f:
        for line in f:
            el = line.split(", ")
            solver_names.append(el[0])
            v = [float(e) for e in el[1:]]
            conv_values.append(v)

    if len(zone_starts) != len(solver_names):
        raise ValueError(f"Number of frames in trace ({len(zone_starts)}) does not match number of solvers in convergence plots ({len(solver_names)})")

    fig, ax = plt.subplots()

    for i in range(len(zone_starts)):

        df_index = zone_starts.iloc[i].name
        if i + 1 < len(zone_starts):
            df_index_next = zone_starts.iloc[i+1].name
        else:
            df_index_next = df.index[-1]
        df_slice = df[df_index:df_index_next]

        start_time = df_slice.iloc[0]["ns_since_start"]
        x_axis = [0]

        solver_name = solver_names[i]
        events = traceEvents(df_slice)
        for e in events:
            print("event name:", e.name, type(e.name))
            if type(e.name) == float:
                # The iterate scope holds all that happens when we step the solver
                # We want to see how the convergence value changes with the execution time of each iteration
                x_axis.append(e.ns_since_start - start_time)

        print("solver:", solver_name)
        print("convergence values:", conv_values[i])
        print("x_axis:", x_axis)
        print("~~~~~~~")
        ax.plot(x_axis, conv_values[i], label=solver_name)

    ax.set_xlabel("Execution Time (ns)")
    ax.set_ylabel("Convergence Value")
    ax.legend()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
