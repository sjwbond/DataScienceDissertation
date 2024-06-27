import numpy as np
import pandas as pd
from scipy.stats import norm
from os import makedirs
from os.path import exists
import math
import datetime as dt
import gc
from threading import Thread
from numpy import linalg as la


def nearest_pd(A: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive-definite matrix to input.
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B: np.ndarray) -> bool:
    """
    Check if a matrix is positive-definite.
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


class ProfileCollection:
    def __init__(self, output_path: str):
        """
        Initialize the ProfileCollection with a specified output path.
        """
        self.ct = dt.datetime.now()
        self.output_path = f"{output_path}/{self.ct.strftime('%y%m%d_%H%M%S')}/"

        if not exists(self.output_path):
            makedirs(self.output_path)

        self.profile_list = []
        self.oADF = pd.DataFrame()

        print("Collection initialized")

    def add_profiles(self, profiles: list):
        """
        Add multiple profiles to the collection.
        """
        self.profile_list.extend(profiles)

        for profile in profiles:
            tmp_path = f"{self.output_path}/{profile.get_name()}/"
            if not exists(tmp_path):
                makedirs(tmp_path)

    def add_profile(self, profile):
        """
        Add a single profile to the collection.
        """
        self.profile_list.append(profile)

    def make_analysis_array(self):
        """
        Create an analysis array from the profiles.
        """
        self.oADF = self.profile_list[0].get_adf()
        self.oADF.index.name = 'Date'

        for profile in self.profile_list[1:]:
            tmp_arr = profile.get_adf()
            tmp_arr.index.name = 'Date'
            self.oADF = pd.merge(self.oADF, tmp_arr, how='inner', left_index=True, right_index=True)

    def make_ranks(self):
        """
        Rank the values in the analysis array.
        """
        self.oADF = self.oADF.rank(axis=0)
        print("Made Ranks")

    def reorder_columns(self, write_out: bool):
        """
        Reorder the columns in the analysis array.
        """
        cols = self.oADF.columns.tolist()
        S = [col for col in cols if col.startswith('S')]
        A = [col for col in cols if col.startswith('A')]
        cols = A + S

        if write_out:
            self.oADF.to_csv(f"{self.output_path}ADF_Sorted_no.txt", sep="\t")

        self.oADF = self.oADF[cols]

        if write_out:
            self.oADF.to_csv(f"{self.output_path}ADF_Sorted_yes.txt", sep="\t")

        col_arr = np.asarray(cols, np.string_)
        if write_out:
            np.savetxt(f"{self.output_path}Cholesky_Headers.txt", col_arr, delimiter="\t", fmt='%s')

        print("Rearranged Columns")

    def cholesky_decompose(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform Cholesky decomposition on the input matrix.
        """
        try:
            use32bit = True
            N = matrix.shape[0]
            LTM = np.zeros((N, N), np.float32 if use32bit else np.float64)

            for j in range(N):
                S = sum(LTM[j, K] * LTM[j, K] for K in range(j))
                LTM[j, j] = matrix[j, j] - S
                if LTM[j, j] <= 0:
                    break

                LTM[j, j] = math.sqrt(LTM[j, j])
                for i in range(j + 1, N):
                    S = sum(LTM[i, K] * LTM[j, K] for K in range(j))
                    LTM[i, j] = (matrix[i, j] - S) / LTM[j, j]

                gc.collect()

            return LTM

        except Exception as e:
            print(str(e))

    def choleskypy(self, A: np.ndarray) -> np.ndarray:
        """
        Perform Cholesky decomposition using a Python implementation.
        """
        L = [[0.0] * len(A) for _ in range(len(A))]
        for i, (Ai, Li) in enumerate(zip(A, L)):
            for j, Lj in enumerate(L[:i + 1]):
                s = sum(Li[k] * Lj[k] for k in range(j))
                Li[j] = math.sqrt(Ai[i] - s) if i == j else (1.0 / Lj[j] * (Ai[j] - s))
        return np.array(L)

    def make_correlation_matrix(self, write_outputs: bool):
        """
        Create and possibly write the correlation matrix and its Cholesky decomposition.
        """
        self.oCorrelations = self.oADF.corr()

        if write_outputs:
            np.savetxt(f"{self.output_path}Correlation_Matrix.txt", self.oCorrelations, delimiter="\t", fmt='%s')

        try:
            self.oCholeskyMatrix = self.cholesky_decompose(self.oCorrelations.values)
            gc.collect()
        except:
            gc.collect()
            print("Couldn't compute Cholesky")

        if write_outputs:
            np.savetxt(f"{self.output_path}Cholesky_Matrix_New.txt", self.oCholeskyMatrix, delimiter="\t", fmt='%s')

        gc.collect()
        self.oDFCholesky = pd.DataFrame(self.oCholeskyMatrix.T, columns=self.oCorrelations.columns, dtype=np.float64)
        print("Calculated Correlations")

    def calculate_periods_to_simulate(self):
        """
        Calculate the periods to simulate.
        """
        self.SimulationHeader = [col for col in self.oADF.columns if col.startswith("S")]
        print("Calculated Column Headers")

    def make_simulation_product_matrices(self):
        """
        Create matrices for the simulation products.
        """
        self.lSimulationProducts = []
        CholMatHeaders = list(self.oDFCholesky.columns.values)
        self.SimMatProds = [word[2:6] for word in self.SimulationHeader]
        CholMatProdsTimes = [word[2:6] + word[14:19] for word in CholMatHeaders]
        SimMatProdsTimes = [word[2:6] + word[14:19] for word in self.SimulationHeader]
        SimMatProdsTimesS = pd.Series(SimMatProdsTimes)
        CholMatHeadersS = pd.Series(CholMatHeaders)

        chol_matrix_rows = [CholMatHeadersS[CholMatHeadersS == x].index.tolist()[0] for x in self.SimulationHeader]
        prod_cols = [SimMatProdsTimesS[SimMatProdsTimesS == x].index.tolist()[0] for x in CholMatProdsTimes]

        iCholPeriodMax = max(int(word[9:13]) for word in CholMatHeaders)
        tmpArr = np.zeros((iCholPeriodMax + 1, len(self.SimulationHeader)), dtype=np.float64)

        for i in range(len(self.SimulationHeader)):
            CholMatRow = chol_matrix_rows[i]
            for j in range(len(CholMatHeaders)):
                CholMatCol = j
                ProdRow = len(tmpArr) - int(CholMatHeaders[j][9:13]) - 1
                tmpArr[ProdRow][prod_cols[j]] = self.oCholeskyMatrix[CholMatRow][CholMatCol]

            self.lSimulationProducts.append(tmpArr)
        print("Created Product Matrices")

    def norm_s_inv(self, x: float) -> float:
        """
        Apply the normal inverse cumulative distribution function.
        """
        return norm.cdf(x)


    def apply_distributions(self, start_date: dt.datetime, periods: int, sim: int, i_no_dec: int):
        """
        Apply distributions to the profiles.
        """
        for P, profile in enumerate(self.profile_list):
            mylist = [self.SimulationHeader[i] for i in range(len(self.SimMatProds)) if self.SimMatProds[i] == str(P + 1).zfill(4)]
            tmp_arr = self.CUDF[mylist].values
            profile.apply_distribution(start_date, periods, tmp_arr)
            tmp_out_path = f"{self.output_path}{profile.get_name()}/{str(sim + 1).zfill(6)} Profile {profile.s_short_name()} Date {self.ct.strftime('%y%m%d_%H%M%S')}.txt"

            profile.write_outputs(tmp_out_path, start_date, i_no_dec)

    def apply_distributions_non_threaded(self, start_date: dt.datetime, periods: int, sim: int):
        """
        Apply distributions to the profiles without threading.
        """
        for P, profile in enumerate(self.profile_list):
            mylist = [self.SimulationHeader[i] for i in range(len(self.SimMatProds)) if self.SimMatProds[i] == str(P + 1).zfill(4)]
            tmp_arr = self.CUDF[mylist].values
            profile.apply_distribution(start_date, periods, tmp_arr)
            profile.write_outputs(f"{self.output_path}{str(sim).zfill(6)} Profile {str(P + 1).zfill(4)} Date {self.ct.strftime('%y%m%d_%H%M%S')}.txt", start_date)

    def simulate(self, in_start: int, in_end: int, start_date: dt.datetime, end_date: dt.datetime, i_no_dec: int):
        """
        Run simulations for a range of simulation indices.
        """
        periods = (end_date - start_date).days
        for sim in range(in_start, in_end):
            print(f"Simulation {sim}")
            self.simulate_collection(periods)
            self.apply_distributions(start_date, periods, sim, i_no_dec)
            del self.CUDF
            gc.collect()

    def simulate_pool(self, in_args):
        """
        Run simulations using a pool of processes.
        """
        self.low_priority()

        in_start, in_end, start_date, end_date, b_threaded, i_no_dec = in_args
        periods = (end_date - start_date).days

        for sim in range(in_start, in_end):
            print(f"Simulation {sim}")
            self.simulate_collection(periods)
            self.apply_distributions(start_date, periods, sim, b_threaded, i_no_dec)

    def simulate_collection(self, periods: int):
        """
        Simulate the collection of profiles.
        """
        self.blocks_to_simulate = periods
        norm_rands = np.random.normal(0, 1, [self.blocks_to_simulate + self.lSimulationProducts[0].shape[0], len(self.SimulationHeader)])
        correl_norms = np.zeros([self.blocks_to_simulate, len(self.SimulationHeader)])
        correl_unis = np.zeros([self.blocks_to_simulate, len(self.SimulationHeader)])

        for i in range(correl_norms.shape[0]):
            norm_rand_slice = norm_rands[i:i + self.lSimulationProducts[0].shape[0], :]
            for j in range(correl_norms.shape[1]):
                correl_norms[i][j] = (self.lSimulationProducts[j] * norm_rand_slice).sum()

            my_norms = correl_norms[i].flatten()
            correl_unis[i] = norm.cdf(my_norms)

        self.CUDF = pd.DataFrame(correl_unis, columns=np.asarray(self.SimulationHeader), dtype=np.float64)
        print("Finished Simulating")

    def low_priority(self):
        """
        Set the priority of the process to below-normal.
        """
        import sys
        try:
            sys.getwindowsversion()
            is_windows = True
        except AttributeError:
            is_windows = False

        if is_windows:
            import win32api
            import win32process
            import win32con

            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            import os
            os.nice(1)

    def move_at_end(self, s_path: str):
        """
        Move output to a final path at the end of the process.
        """
        from shutil import move
        move(self.output_path, s_path)
        print("Moved")
