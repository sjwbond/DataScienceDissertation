import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import datetime as dt
from numpy import linalg as la


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def load_file_to_array_as_months(inpath,firstyear,lastyear):

    dfs = []
    df = pd.read_csv(inpath,sep="\t")
    df["Date"] = pd.to_datetime(df["Date"])
    hourcols = [str(x) for x in range(1,25)]

    if all(elem in df.columns  for elem in ["Max","Min"]):
        df1 = pd.crosstab(df["Date"].dt.year,
                    df["Date"].dt.month,
                    df["Max"],
                    aggfunc="mean",
                    rownames=["Year"],
                    colnames=["Month"])

        df2 = pd.crosstab(df["Date"].dt.year,
                          df["Date"].dt.month,
                          df["Min"],
                          aggfunc="mean",
                          rownames=["Year"],
                          colnames=["Month"])

        #df1 = df[["Date", "Max"]].resample('M',on='Date').mean()
        #df2 = df[["Date", "Min"]].resample('M',on='Date').mean()
        df1 = df1[[all(x) for x in zip(df1.index >= firstyear,df1.index <= lastyear)]]
        df2 = df2[[all(x) for x in zip(df2.index >= firstyear,df2.index <= lastyear)]]
        years1 = df1.index
        years2 = df2.index
        data1 = df1.to_numpy(np.float32)
        data2 = df2.to_numpy(np.float32)

        dfs.extend([[years1,data1],[years2,data2]])

    elif all(elem in df.columns  for elem in [str(x) for x in range(1,23)]):
        df1 = df[["Date"]]
        df1["Sum"] = df[hourcols].mean(axis=1)
        df2 = pd.crosstab(df1["Date"].dt.year,
                          df1["Date"].dt.month,
                          df1["Sum"],
                          aggfunc="mean",
                          rownames=["Year"],
                          colnames=["Month"])
        #df2 = df1[["Date", "Sum"]].resample('M',on='Date').mean()
        years2 = df2.index
        data2 = df2.to_numpy(np.float32)
        dfs.extend([[years2, data2]])

    return dfs

def load_historics_files_to_numpy(inpath, firstyear = 0, lastyear = 3000):

    filestoload = [inpath + f for f in listdir(inpath) if isfile(join(inpath, f))]
    montharrays = []

    for filepath in filestoload:
        montharrays.extend(load_file_to_array_as_months(filepath,firstyear,lastyear))

    bigarray = np.hstack([x[1] for x in montharrays])

    return bigarray

def get_chol_mat(arr,outfolder):
    corrmat = np.corrcoef(arr.T)
    try:
        cholmat = np.linalg.cholesky(corrmat)
    except np.linalg.LinAlgError:
        corrmat2 = nearestPD(corrmat)
        cholmat = np.linalg.cholesky(corrmat2)

        np.savetxt(outfolder + "corrmat corrected.txt", corrmat2,delimiter='\t')
        np.savetxt(outfolder + "corrmat corrected 2dp.txt", corrmat2,fmt = '%1.2f', delimiter='\t')

    np.savetxt(outfolder + "corrmat.txt", corrmat,delimiter='\t')
    np.savetxt(outfolder + "cholmat.txt", cholmat,delimiter='\t')

    np.savetxt(outfolder + "corrmat 2dp.txt", corrmat,fmt = '%1.2f',delimiter='\t')
    np.savetxt(outfolder + "cholmat 2dp.txt", cholmat,fmt = '%1.2f',delimiter='\t')

    return cholmat

if __name__ == "__main__":


    strnow = dt.datetime.now().strftime("%Y%m%d%H%M")
    filepath = "F:/Weather Simulations/tosim/Final/"
    outfolder = 'F:/temp/'+strnow+"/"

    filepath = "F:/Weather Simulations/tosim/Final"

    os.makedirs(outfolder,exist_ok=True)

    firstyear = 2008
    lastyear = 2017
    bigmat = load_historics_files_to_numpy(filepath,firstyear,lastyear)
    cholmat = get_chol_mat(bigmat,outfolder)



    print("All Done")

