import datetime as dt
import ProfileCollection as pc

class MPSimulator:
    """
    MPSimulator is a control object designed to manage and run simulations.
    It can be extended to support parallel execution in the future.
    For now it only contains one function that passes variables to the process class
    """

    def __init__(self):
        """
        Initialize the MPSimulator object.
        Currently, it only prints an initialization message.
        """
        print("initialized")

    def simulate(self, start_sim: int, end_sim: int, start_date: dt.datetime, end_date: dt.datetime, no_dec: int, oPC):
        """
        Run the simulation using the provided parameters and ProfileCollection object.

        Parameters:
        start (int): The starting index for the simulation.
        end (int): The ending index for the simulation.
        start_date (dt.datetime): The start date of the simulation period.
        end_date (dt.datetime): The end date of the simulation period.
        no_dec (int): The number of decimal places for the simulation results.
        oPC (ProfileCollection): An instance of ProfileCollection to use for the simulation.
        """
        oPC.simulate(start_sim, end_sim, start_date, end_date, no_dec)
