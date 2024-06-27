import MPSimulator as MPS
import multiprocessing as mp
import datetime as dt
import WindProfile as WP
import ProfileCollection as PC
from os import listdir
from os.path import isfile, join, dirname, abspath
from json import loads
import inspect
import time
import gc


def run_normal(settings: dict) -> None:
    """
    Run the normal simulation process based on settings.

    Parameters:
    settings (dict): Dictionary of settings to override default options.
    """
    start_time = time.time()

    # Get current script path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = dirname(abspath(filename))
    print(path)

    # Read options from file
    options_dict = read_options_file(join(path, "options.txt"))

    # Set default dates
    start_date = dt.datetime.strptime(options_dict["StartDate"], "%Y-%m-%d")
    end_date = dt.datetime.strptime(options_dict["EndDate"], "%Y-%m-%d")

    # Read additional settings
    base_path = options_dict["Data"]
    output_path = options_dict["TempOut"]
    no_sims = int(settings.get("Sim Count", options_dict["NoSims"]))
    no_decimals = int(options_dict["NoDecimals"])
    auto_blocks = int(options_dict["AutoBlocks"])


    move_at_end = loads(options_dict["MoveAtEnd"].lower())
    final_path = options_dict["MoveTo"]
    start_index = int(settings.get("Start Index", options_dict["StartIndex"]))


    print(base_path)
    only_files = [f for f in listdir(base_path) if isfile(join(base_path, f)) and f.endswith(".tsv")]

    profiles = [
        WP.WindProfile(join(base_path, f), i, 2 if "PVSAT" in f else auto_blocks)
        for i, f in enumerate(only_files, start=1)
    ]

    oPC = PC.ProfileCollection(output_path)
    oPC.add_profiles(profiles)
    oPC.make_analysis_array()
    oPC.make_ranks()
    gc.collect()
    oPC.reorder_columns(False)
    gc.collect()
    oPC.make_correlation_matrix(False)
    gc.collect()
    oPC.calculate_periods_to_simulate()
    oPC.make_simulation_product_matrices()

    oMP = MPS.MPSimulator()

    oMP.simulate(1, no_sims, start_date, end_date, no_decimals, oPC)

    if move_at_end:
        oPC.move_at_end(final_path)

    print("All Done")
    print(f"--- {time.time() - start_time} seconds ---")


def read_options_file(filepath: str) -> dict:
    """
    Read options from a given file and return as a dictionary.

    Parameters:
    filepath (str): The path to the options file.

    Returns:
    dict: Dictionary of options.
    """
    options_dict = {}
    with open(filepath) as options_file:
        options_lines = options_file.readlines()

    for line in options_lines:
        if not line.startswith('end'):
            key, value = line.strip().split('\t')
            options_dict[key] = value

    return options_dict


if __name__ == '__main__':
    run_normal({})
