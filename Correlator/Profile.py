import numpy as np
import pandas as pd
import datetime as dt
import ntpath
from time import sleep


class ProfileClass:
    def __init__(self, file_path: str, profile_id: int):
        """
        Initialize the ProfileClass with a file path and an ID.

        Parameters:
        file_path (str): The path to the input file.
        profile_id (int): The profile ID.
        """
        self.bTSP = False
        self.bTSF = False
        self.path = file_path
        self.file_name = ntpath.basename(file_path)
        self.short_name = str(profile_id).zfill(4)
        self.name = self.file_name[:-4]

        # Read the file to inspect the first column
        temp_df = pd.read_table(file_path, sep='\t', nrows=1)
        first_column_name = temp_df.columns[0] if temp_df.columns[0] else 'Unnamed: 0'

        # Read the file again with the correct column name
        self.oBASEDF = pd.read_table(
            file_path,
            sep='\t',
            parse_dates=[first_column_name],
            index_col=first_column_name
        )

        self.oHistoricTimeSlices = np.zeros(self.oBASEDF.shape, np.int64, order='C')
        self._prepare_columns()
        self.oDF = self.oBASEDF.copy()
        self.oAR = self.oDF.values
        self.oVEC = np.concatenate(self.oAR)
        self.periods_per_block = self.oDF.shape[1]
        self.number_of_blocks = self.oDF.shape[0]

        print(f"Initialized Profile {self.short_name}")

    def _prepare_columns(self):
        """
        Prepare the column names for the data frame.
        """
        l_col_names = [f"{self.short_name}-{str(i).zfill(4)}" for i in range(self.oBASEDF.shape[1])]
        self.oBASEDF.columns = l_col_names

    def set_auto_correlation_blocks(self, acp: int):
        """
        Set the auto-correlation blocks.

        Parameters:
        acp (int): Number of auto-correlation blocks.
        """
        self.acp = acp

    def s_short_name(self) -> str:
        """
        Get the short name of the profile.

        Returns:
        str: The short name of the profile.
        """
        return self.short_name

    def get_name(self) -> str:
        """
        Get the name of the profile.

        Returns:
        str: The name of the profile.
        """
        return self.name

    def make_analysis_matrix(self):
        """
        Create an analysis matrix for the profile.
        """
        self.oAM = np.zeros(
            [self.oDF.shape[0] - self.acp + 1, self.oDF.shape[1] * self.acp], dtype=np.float64
        )

        for i in range(self.acp - 1, self.oDF.shape[0]):
            p = 0
            b = i - self.acp + 1
            for j in range(i - self.acp + 1, i + 1):
                for k in range(self.oDF.shape[1]):
                    self.oAM[b, p] = self.oDF.values[j, k]
                    p += 1

        dates = self.oDF.index.tolist()[self.acp - 1:]
        l_col_names = self._generate_column_names()

        self.oADF = pd.DataFrame(self.oAM, dates, l_col_names, dtype=np.float64)
        print("Initialized Analysis Matrix")

    def _generate_column_names(self) -> list:
        """
        Generate column names for the analysis matrix.

        Returns:
        list: A list of column names.
        """
        i = -self.oDF.shape[1] * (self.acp - 1)
        l_col_names = []
        my_p = 0
        my_t = self.acp - 1

        while i < self.oDF.shape[1]:
            if my_p >= self.oDF.shape[1]:
                my_p = 0
                my_t -= 1

            prefix = "A-" if i < 0 else "S-"
            l_col_names.append(f"{prefix}{self.short_name}-P-{str(my_t).zfill(4)}-{str(my_p).zfill(4)}")

            i += 1
            my_p += 1

        return l_col_names

    def make_ranks(self):
        """
        Create ranks for the analysis matrix.
        """
        arr_list = []

        i = self.acp
        while i < self.oVEC.size:
            tmp_list = [self.oVEC[i - my_counter] for my_counter in range(self.acp)]
            arr_list.append(tmp_list)
            i += 1

        self.rolling_matrix = np.asarray(arr_list, dtype=np.float64)
        print("Made Analysis Matrix")

    def build_simulated_scenario(self, periods: int, tmp_arr: np.ndarray):
        """
        Build a simulated scenario.

        Parameters:
        start_date (dt.datetime): The start date of the simulation.
        periods (int): Number of periods to simulate.
        tmp_arr (np.ndarray): Array of temporary values for the simulation.
        """
        self.simulated_outturn = np.zeros([periods, self.oAR.shape[1]], np.float64)
        self.simulated_outturn = np.array([
            [np.percentile(self.Distributions[self.oFutureTimeSlices[i][j]], float(tmp_arr[i][j] * 100)) for j in
             range(self.simulated_outturn.shape[1])]
            for i in range(self.simulated_outturn.shape[0])
        ])

    def apply_distribution(self, start_date: dt.datetime, periods: int, tmp_arr: np.ndarray):
        """
        Apply distribution to the profile.

        Parameters:
        start_date (dt.datetime): The start date of the distribution application.
        periods (int): Number of periods to apply.
        tmp_arr (np.ndarray): Array of temporary values.
        """
        if self.bTSF:
            self.build_simulated_scenario(start_date, periods, tmp_arr)
            print("hurrah!")
        else:
            self.set_future_time_slice_by_month_and_period(start_date, periods)
            self.build_simulated_scenario( periods, tmp_arr)
            print("Wahey!")

    def write_outputs(self, output_path: str, start_date: dt.datetime, no_dec: int):
        """
        Write the output to a file.

        Parameters:
        output_path (str): The path to the output file.
        start_date (dt.datetime): The start date for the output data.
        no_dec (int): Number of decimal places for the output.
        """
        end_date = start_date + dt.timedelta(days=len(self.simulated_outturn))
        simulated_outturn_dates = np.arange(start_date, end_date, dt.timedelta(days=1)).astype(dt.datetime).flatten()

        o_frame = pd.DataFrame(self.simulated_outturn, index=simulated_outturn_dates, dtype=float)

        result = 0
        while result == 0:
            try:
                o_frame.to_csv(output_path, sep="\t", float_format=f'%.{no_dec}f')
                result = 1
            except:
                sleep(5)

        print("Outputs Written")

    def get_adf(self) -> pd.DataFrame:
        """
        Get the analysis data frame.

        Returns:
        pd.DataFrame: The analysis data frame.
        """
        return self.oADF

    def calculate_distributions(self):
        """
        Calculate the distributions for the profile.
        """
        top_slice = self.oHistoricTimeSlices.max() + 1
        self.Distributions = [[] for _ in range(top_slice)]

        for x, y in zip(self.oHistoricTimeSlices, self.oAR):
            for s, t in zip(x, y):
                self.Distributions[s].append(t)

        for i in range(len(self.Distributions)):
            self.Distributions[i] = np.sort(np.asarray(self.Distributions[i], np.float64))

        print("Calculated Distributions")

    def set_future_time_slice_by_month_and_period(self, start_date: dt.datetime, periods: int):
        """
        Set future time slices by month and period.

        Parameters:
        start_date (dt.datetime): The start date for the time slices.
        periods (int): Number of periods.
        """
        self.oFutureTimeSlices = np.zeros([periods, self.oAR.shape[1]], np.int64)

        for i in range(periods):
            mydt = start_date + dt.timedelta(days=i)
            mymonth = mydt.month
            for j in range(self.oHistoricTimeSlices.shape[1]):
                self.oFutureTimeSlices[i][j] = (mymonth - 1) * self.oFutureTimeSlices.shape[1] + j

        self.bTSF = True
        print("Completed Seasonality by Month")

    def set_time_slice_by_month_and_period(self):
        """
        Set time slices by month and period.
        """
        for i in range(self.oBASEDF.shape[0]):
            mymonth = self.oBASEDF.index[i].month
            for j in range(self.oHistoricTimeSlices.shape[1]):
                self.oHistoricTimeSlices[i][j] = (mymonth - 1) * self.oHistoricTimeSlices.shape[1] + j

        self.bTSP = True
        print("Completed Seasonality by Month")

    def set_time_slice_by_month(self):
        """
        Set
        time
        slices
        by
        month.
        """
        for i in range(self.oBASEDF.shape[0]):
            mymonth = self.oBASEDF.index[i].month
            for j in range(self.oHistoricTimeSlices.shape[1]):
                self.oHistoricTimeSlices[i][j] = mymonth - 1

        self.bTSP = True
        print("Completed Seasonality by Month")
