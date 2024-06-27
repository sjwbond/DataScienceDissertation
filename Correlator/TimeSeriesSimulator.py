import datetime as dt
import numpy as np
import pandas as pd
import time
import os
import math
import datetime as dt
import math
import threading
import Profile
from os import listdir
from os.path import isfile, join
from scipy.stats import norm

class TSSimulator():

    def __init__(self,inStart,inEnd, inCap, inWriteText, inWriteXLS, inWriteHours, inWriteMonths):

        self.WriteHours = inWriteHours
        self.WriteMonths = inWriteMonths
        self.WriteTXT = inWriteText
        self.WriteXLS = inWriteXLS
        self.Capacity = inCap
        self.StartDate = inStart
        self.EndDate = inEnd
        self.Diagnostics = True
        self.Simcount = 0
        self.WriteNumber = 0
        self.Sims = []
        self.threads = []
        self.lPortfolioValueMonthly = []
        self.lImbalanceVSBenchmarkMonthly = []
        self.lPortfolioGenerationMonthly = []
        self.lImbalanceVSBenchmarkEURMWhMonthly = []
        self.lPortfolioValueNewMonthly = []
        self.lImbalanceVSBenchmarkNewMonthly = []
        self.lPortfolioGenerationNewMonthly = []
        self.lImbalanceVSBenchmarkEURMWhNewMonthly = []
        self.lMarginalVolume = []
        self.lMarginalCost = []
        self.lMarginalPrice = []


        directory = "C:/temp/TSS/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        mytime = time.strftime("%Y-%m-%d %H%M%S", time.localtime())

        directory = directory + mytime + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.BasePath = directory

        directory = directory + "Simulations/"
        self.SimsDirectory = directory

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.Profiles = []
        self.dfProfiles = []

        self.MonthIndexes = []
        self.Selections = []

        print("TSS Initialised")

    def AddProfile(self, inProfile):
        self.Profiles.append(inProfile)
        self.dfProfiles.append(inProfile.GetDF())
    def IndexFromFile(self,FilePath):

        self.oDF = pd.read_table(FilePath, sep='\t',parse_dates=['Date'])
    def SetAnalysisDates(self,StartDate,EndDate):

        mask = (self.oBaseTable.index >= StartDate) & (self.oBaseTable.index <= EndDate)
        self.oBaseTable = self.oBaseTable.loc[mask]

    def MakeRanks(self):

        iMaxIndex = len(self.oBaseTables)
        self.oRanks = []

        for i in range(0, iMaxIndex):

            self.oRanks.append(pd.DataFrame(index=self.oBaseTables[i].index, columns=self.oBaseTables[i].columns))

            Cols = self.oBaseTables[i].columns.tolist()

            for j in range(0,len(Cols)):#-1):
                self.oRanks[i][Cols[j]] = self.oBaseTables[i][Cols[j]].rank(method='first',ascending=1)

            if(self.Diagnostics):
                self.oRanks[i].to_csv(self.BasePath + "Rank Matrix "+str(i)+ ".txt", header=None, index=None, sep='\t', mode='a',float_format='%.3f')
                self.oBaseTables[i].to_csv(self.BasePath + "Base "+str(i)+ ".txt", header=True, index=True, sep='\t', mode='a',float_format='%.3f')
        print("Finished Ranking Indexes")

    def MakeCorrelationMatrixFromRank(self):

        self.oCorrelMatrix = []
        for i in range(0, len(self.oBaseTables)):
            self.oCorrelMatrix.append(self.oRanks[i].corr())
            if(self.Diagnostics):
                self.oCorrelMatrix[i].to_csv(self.BasePath + "Correlation "+str(i)+ ".txt", header=None, index=None, sep='\t', mode='a',float_format='%.3f')

        print("Finished Making Correlation Matrix")


    def MakeCholeskyFromCorrelation(self):
        self.oCholeskyMatrices = []

        for i in range(0, len(self.oCorrelMatrix)):
            oTMP = self.oCorrelMatrix[i].as_matrix()

            #self.oCholeskyMatrices.append(self.cholesky(oTMP))
            self.oCholeskyMatrices.append(np.linalg.cholesky(oTMP))

        print("Made Cholesky Matrices")


    def SplitBaseByIndex(self):

        self.oBaseTables = []
        iMaxIndex = self.oBaseTable['myIndex'].max()

        for i in range(0, iMaxIndex+1):

            oTMP = self.oBaseTable.loc[self.oBaseTable['myIndex']==i]
            oSUM = oTMP.sum(axis=0)
            if(self.Diagnostics):
                oSUM.to_csv(self.BasePath + "Sum of Base "+str(i)+ ".txt", header=True, index=True, sep='\t', mode='a',float_format='%.3f')


            Cols = oSUM.axes
            ColList = []

            for j in range(0,len(oSUM)):
                if(oSUM[j]>0):
                    ColList.append(Cols[0][j])

            oTMP = oTMP[ColList]

            self.oBaseTables.append(oTMP)

        print("Finished Splitting By Index")

    def RemoveColsByMode(self):

        for i in range(0,len(self.oBaseTables)):
            oTMP = self.oBaseTables[i]
            oArray = oTMP.as_matrix()
            oModes = oTMP.mode().as_matrix()
            oCols = oTMP.columns.tolist()
            oIndex = oTMP.index.tolist()
            oColsToKeep = []

            for j in range(0,len(oCols)):
                oVector = oArray[:,j]
                iTotal = 0
                iCount = 0
                iMode = oModes[0][j]

                if(not math.isnan(iMode)):
                    for k in range(0,len(oVector)):
                        iTotal+=1
                        if(oVector[k]==iMode):
                            iCount+=1

                    myRatio = float(iCount)/float(iTotal)

                    if(myRatio < 0.8):
                        oColsToKeep.append(oCols[j])

                else:
                    oColsToKeep.append(oCols[j])

            oSelectedData = oTMP[oColsToKeep]
            self.oBaseTables[i] = oSelectedData

        print("removed Redundant Columns")
    def IndexBySeason(self):


        tmp = [1,2,3,10,11,12]
        self.MonthIndexes.append(tmp)
        tmp = [4,5,6,7,8,9]
        self.MonthIndexes.append(tmp)

        self.oBaseTable = pd.DataFrame(data=self.Profiles[0].GetArray(), index=self.Profiles[0].GetDates(),columns=self.Profiles[0].GetColNames())  # 1st row as the column names
        #self.oBaseTable = pd.DataFrame(data=self.dfProfiles[0], index=self.dfProfiles[0].index,columns=self.Profiles[0].GetColNames())  # 1st row as the column names


        myTMP = self.oBaseTable['0001-0000']
        self.oBaseTable = self.oBaseTable[self.oBaseTable.sum(axis=1) > 0]

        self.oBaseTable['myIndex'] = 0
        self.oBaseTable['myDate'] = pd.to_datetime(self.oBaseTable.index, errors='coerce')

        cols = self.oBaseTable.columns.tolist()
        cols = cols[-2:]+cols[:-2] # or whatever change you need
        self.oBaseTable.reindex(columns=cols)


        for i in range(0,2):
            for j in range(0,len(self.MonthIndexes[i])):
                self.oBaseTable.loc[(self.oBaseTable['myDate'].dt.month==self.MonthIndexes[i][j]), 'myIndex'] = i

        for p in range(1,len(self.Profiles)):
            oJoinTable = pd.DataFrame(data=self.Profiles[p].GetArray(), index=self.Profiles[p].GetDates(),columns=self.Profiles[p].GetColNames())
            oJoinTable = oJoinTable[oJoinTable.sum(axis=1) > 0]

            self.oBaseTable = pd.concat([self.oBaseTable, oJoinTable], axis=1, join='inner')


        print("Finished Indexation by Season")


    def IndexByMonth(self):

        for i in range(1,12):
            print("blah")

    def cholesky(self,A):
        L = [[0.0] * len(A) for _ in range(len(A))]
        for i, (Ai, Li) in enumerate(zip(A, L)):
            for j, Lj in enumerate(L[:i+1]):
                s = sum(Li[k] * Lj[k] for k in range(j))
                Li[j] = math.sqrt(Ai[i] - s) if (i == j) else \
                          (1.0 / Lj[j] * (Ai[j] - s))
        return L

    def CorrelationMatrixCorrections(self):


        for p in range(0,len(self.oCorrelMatrix)):
            oTMP = self.oCorrelMatrix[p].as_matrix()

            for i in range(0,len(oTMP)):
                for j in range(0,len(oTMP[i])):
                    if (i!=j):
                        if(oTMP[i][j]==1):
                            difference = float(abs(i-j)) / 100
                            oTMP[i][j]-=difference

            self.oCorrelMatrix[p] = pd.DataFrame(data=oTMP, index=self.oCorrelMatrix[p].index,columns=self.oCorrelMatrix[p].columns)


        print("Corrections Made")
    def GetBaseTables(self):
        return self.oBaseTables
    def MakeDistributions(self):

        self.Distributions = []

        for p in range(0,len(self.oBaseTables)):

            oTMP = self.oBaseTables[p].as_matrix()

            oDim = oTMP.shape
            oP = np.zeros([101,oDim[1]],dtype=np.float64,order='C')

            for j in range(0,101):
                percentile = j/100
                pTMP = np.percentile(oTMP, j, axis=0)
                oP[j,:] = pTMP

            self.Distributions.append(oP)

        print("finished Distributions")
    def SetStartDate(self, inDate):
        self.StartDate = inDate
    def SetEndDate(self, inDate):
        self.EndDate = inDate
    def PrepareOutputs(self):

        NoOfDays = (self.EndDate-self.StartDate).days+1

        self.myIndex = pd.date_range(self.StartDate, periods=NoOfDays, freq='D')
        IndexMonths = self.myIndex.month
        myCols = []
        self.SDOY = self.myIndex.dayofyear
        self.DIndex = np.zeros(self.SDOY.shape,dtype=np.float64,order='C')



        for i in range(0,len(self.MonthIndexes)):
            for j in range(0,len(self.MonthIndexes[i])):
                myMonth = self.MonthIndexes[i][j]
                for p in range(0,self.SDOY.shape[0]):
                    if(IndexMonths[p]==myMonth):
                        self.DIndex[p] = i



        for p in range(0,len(self.Profiles)):
           myCols +=self.Profiles[p].ColNames

        self.Outputs = pd.DataFrame(index=self.myIndex, columns=myCols,dtype=np.float64)
        self.Outputs = self.Outputs.fillna(0) # with 0s rather than NaNs
        self.Outputs['PeriodIndex'] = self.DIndex
        self.OutputsA = pd.DataFrame(index=self.myIndex, columns=myCols,dtype=np.float64)
        self.OutputsA = self.OutputsA.fillna(0) # with 0s rather than NaNs
        self.OutputsA['PeriodIndex'] = self.DIndex

        self.PeriodOutputs = []
        self.PeriodOutputsA = []

        for p in range(0,len(self.MonthIndexes)):
            myCols2 = self.oCorrelMatrix[p].columns.tolist()
            oTMP = self.Outputs.loc[self.Outputs['PeriodIndex'] == p]
            oTMP = oTMP[myCols2]
            self.PeriodOutputs.append(oTMP)

            oTMPA = self.OutputsA.loc[self.OutputsA['PeriodIndex'] == p]
            oTMPA = oTMPA[myCols2]
            self.PeriodOutputsA.append(oTMP)


        print("outputs Prepared")

    def SimulateOne(self):

        print("Simulating Number - " + str(self.Simcount))

        SimResult = []
        ProfileResults = []
        CorrelRandsUniforms = []
        NoOfDays = (self.EndDate-self.StartDate).days+1
        MaxProfile1 = pd.DataFrame(data=self.Profiles[0].GetMaximumProfile(),columns=self.Profiles[0].GetColNames())
        MaxProfile2 = pd.DataFrame(data=self.Profiles[0].GetMaximumProfile(),columns=self.Profiles[1].GetColNames())
        MaxProfile = pd.concat([MaxProfile1, MaxProfile2], axis=1)

        CorrelRandsUniform = []

        for p in range(0,len(self.PeriodOutputs)):
            ResultsMat = self.PeriodOutputs[p].as_matrix()
            myDist = self.Distributions[p]
            NormRands = np.random.normal(0, 1, self.PeriodOutputs[p].shape)
            CorrelRands = np.matmul(NormRands, self.oCholeskyMatrices[p].T)
            CorrelRandsUniform = norm.cdf(CorrelRands)
            CorrelRandsUniform *= 100
            CorrelRandsUniform = CorrelRandsUniform.astype(int)
            CorrelRandsUniforms.append(CorrelRandsUniform)

            for i in range(0,CorrelRandsUniform.shape[0]):
                for j in range(0,CorrelRandsUniform.shape[1]):
                    myval = myDist[CorrelRandsUniform[i][j]][j]
                    ResultsMat[i][j] = myval

            self.PeriodOutputs[p].data = ResultsMat
            self.PeriodOutputs[p].sort_index(inplace=True)
            #self.PeriodOutputs[p].to_csv(self.SimsDirectory + " ProfileOutputs " + str(p) + " Thetic.txt", header=True, index=True, sep='\t', mode='a')

            #np.savetxt(self.SimsDirectory + " ResultsMat - Profile " + str(p) + " Thetic.txt",ResultsMat,delimiter='\t')

        self.Simcount +=1

        oTMP = self.PeriodOutputs[0].append(self.PeriodOutputs[1])
        oTMP.fillna(0, inplace=True)
        oTMP.sort_index(inplace=True)

        oTMPA = oTMP.as_matrix()
        #oTMP.to_csv(self.SimsDirectory + " oTMP Thetic.txt", header=True, index=True, sep='\t', mode='a')

        columnlist = self.PeriodOutputs[p].columns.tolist()
        MaxProfile = MaxProfile[columnlist]
        MaxProfileA = MaxProfile.as_matrix()
        Dates = oTMP.index #.tolist()
        DOY = Dates.dayofyear

        for i in range(0,oTMP.shape[0]):
            for j in range(0, oTMP.shape[1]):
                oTMPA[i][j] *= MaxProfileA[DOY[i]-1][j] * self.Capacity

        for p in range(0,len(self.Profiles)):

            ColList = []
            for i in range(0,len(columnlist)):
                profile = int(columnlist[i][:4])
                if(profile == p+1):
                    ColList.append(columnlist[i])

            tmpResult = pd.DataFrame(columns=self.Profiles[p].GetColNames(), index=self.myIndex)
            tmpResult2 = oTMP[ColList]
            tmpResult[ColList] = tmpResult2
            tmpResult.fillna(0, inplace=True)

            ProfileResults.append(tmpResult)

        SimResult.append(ProfileResults)
        ProfileResults = []

        #Anti-thetics for Variance reduction
        for p in range(0,len(self.PeriodOutputs)):
            ResultsMatA = self.PeriodOutputsA[p].as_matrix()
            CorrelRandsUniform = CorrelRandsUniforms[p]
            CorrelRandsUniform = 100-CorrelRandsUniform

            for i in range(0,CorrelRandsUniform.shape[0]):
                for j in range(0,CorrelRandsUniform.shape[1]):
                    myval = myDist[CorrelRandsUniform[i][j]][j]
                    ResultsMatA[i][j] = myval

            self.PeriodOutputsA[p].data = ResultsMatA
            self.PeriodOutputsA[p].sort_index(inplace=True)
            #self.PeriodOutputsA[p].to_csv(self.SimsDirectory + " ProfileOutputs " + str(p) + " AntiThetic.txt", header=True, index=True, sep='\t', mode='a')
            #np.savetxt(self.SimsDirectory + " ResultsMat - Profile " + str(p) + " AntiThetic.txt",ResultsMatA,delimiter='\t')

        self.Simcount +=1

        oTMP = self.PeriodOutputsA[0].append(self.PeriodOutputsA[1])
        oTMP.fillna(0, inplace=True)
        oTMP.sort_index(inplace=True)
        oTMPA = oTMP.as_matrix()
        #oTMP.to_csv(self.SimsDirectory + " oTMP Anti Thetic.txt", header=True, index=True, sep='\t', mode='a')

        columnlist = self.PeriodOutputs[p].columns.tolist()
        MaxProfile = MaxProfile[columnlist]
        MaxProfileA = MaxProfile.as_matrix()
        Dates = oTMP.index #.tolist()
        DOY = Dates.dayofyear

        for i in range(0,oTMP.shape[0]):
            for j in range(0, oTMP.shape[1]):
                res = MaxProfileA[DOY[i]-1][j] * self.Capacity
                if(res == 0 and j ==25):
                    print("WAIT!")
                oTMPA[i][j] *= res

        for p in range(0,len(self.Profiles)):

            ColList = []
            for i in range(0,len(columnlist)):
                profile = int(columnlist[i][:4])
                if(profile == p+1):
                    ColList.append(columnlist[i])

            tmpResult = pd.DataFrame(columns=self.Profiles[p].GetColNames(), index=self.myIndex)
            tmpResult2 = oTMP[ColList]
            tmpResult[ColList] = tmpResult2
            tmpResult.fillna(0, inplace=True)

            ProfileResults.append(tmpResult)

        SimResult.append(ProfileResults)

        self.Sims.append(SimResult)

        print("Simulated ")
    def ClearSims(self):
        self.Sims = []
    def WriteSims(self):

        #For Every simulation
        for sim in range(len(self.Sims)-1,len(self.Sims)):
            oTMP1 = self.Sims[sim]
            #Each simulation is made up of Thetic and Anto-Thetic
            for ath in range(0,len(oTMP1)):
                self.WriteNumber+=1
                #for each profile that is simulated
                oTMP2 = oTMP1[ath]

                lArrays = []
                lArrayNames = []

                for pro in range(0,len(oTMP2)):
                    oTMP3 = oTMP2[pro]
                    lArrays.append(oTMP3)
                    lArrayNames.append("Profile - " + str(pro))

                    if(self.WriteTXT):
                        oTMP3.to_csv(self.SimsDirectory + " Profile - " + str(pro) + " Sim - " +str(self.WriteNumber)+ ".txt", header=True, index=True, sep='\t', mode='a')

                if(self.WriteXLS):
                    myPath = self.SimsDirectory + " Profiles Sim - " +str(self.WriteNumber)+ ".xlsx"
                    #oTMP3.to_excel(myPath,'Sheet1', engine='xlsxwriter')
                    self.threads.append(myExcelWriterHourly(self.WriteNumber,lArrays,lArrayNames, myPath))
                    self.threads[len(self.threads)-1].start()

    def LoadFiles(self, inFolder):
        oFiles = [f for f in listdir(inFolder) if isfile(join(inFolder, f))]
        self.FileCache = []

        for i in range(3,8):
            for f in oFiles:
                tmp = int(f[:4])

                if(i == tmp):
                    self.FileCache.append(f)
                    print("File " + f[:4] + ", File name " + f)

        #something is going wrong here, check it there should only be 96 columns in df not 97
        self.PortfolioForecast = Profile.Profile(inFolder+self.FileCache[0])
        self.DayAheadPrice = Profile.Profile(inFolder+self.FileCache[1])
        self.PortfolioGeneration = Profile.Profile(inFolder+self.FileCache[2])
        self.ImbalanceTakePrice = Profile.Profile(inFolder+self.FileCache[3])
        self.ImbalanceFeedPrice = Profile.Profile(inFolder+self.FileCache[4])

        print("Loaded Files")

    def CalculateValueDF(self):

        self.dfPortfolioForecast = self.PortfolioForecast.GetDF() / 4000
        self.dfDayAheadPrice = self.DayAheadPrice.GetDF()
        self.dfPortfolioGeneration = self.PortfolioGeneration.GetDF() / 4000
        self.dfImbalanceTakePrice = self.ImbalanceTakePrice.GetDF()
        self.dfImbalanceFeedPrice = self.ImbalanceFeedPrice.GetDF()

        arZeros = np.zeros(self.dfPortfolioForecast.shape,dtype=np.float64)
        dfZeros = pd.DataFrame(data=arZeros,index=self.dfPortfolioForecast.index,columns=self.dfPortfolioForecast.columns,dtype=np.float64)

        self.dfPortfolioImbalance = self.dfPortfolioForecast - self.dfPortfolioGeneration
        dfPortfolioImbalanceTake = np.maximum(self.dfPortfolioImbalance,dfZeros)
        dfPortfolioImbalanceFeed = np.maximum(-self.dfPortfolioImbalance,dfZeros)
        self.dfPortfolioImbalanceTakeValue = dfPortfolioImbalanceTake * -self.dfImbalanceTakePrice
        self.dfPortfolioImbalanceFeedValue = dfPortfolioImbalanceFeed * self.dfImbalanceFeedPrice

        self.dfPortfolioBenchmark = self.dfDayAheadPrice * self.dfPortfolioGeneration
        self.dfDayAheadRevenue = self.dfDayAheadPrice * self.dfPortfolioForecast
        self.dfPortfolioValue = self.dfDayAheadRevenue + self.dfPortfolioImbalanceTakeValue + self.dfPortfolioImbalanceFeedValue
        self.dfImbalanceVSBenchmark = self.dfPortfolioValue-self.dfPortfolioBenchmark
        self.dfImbalanceVSBenchmarkEURMWh = self.dfImbalanceVSBenchmark / self.dfPortfolioGeneration

        print("Calculated Value")

    def CorrectColumnHeaders(self,dfFore,dfReal):
        lColNames = dfFore.columns.values.tolist()

        for i in range(0,len(lColNames)):
            lColNames[i] = lColNames[i][-4:]

        lColNames = [str(int(i)+1) for i in lColNames]
        dfFore.columns = lColNames
        dfReal.columns = lColNames

    def CalculateSimulationValueDFMain(self,dfFore,dfReal):

        lArraysToWrite = []
        lArrayNames = []

        lMonthArraysToWrite = []
        lMonthArrayNames = []

        self.dfPortfolioForecastNew = ((self.PortfolioForecast.GetDF()/1000)+dfFore) / 4
        self.dfDayAheadPrice = self.DayAheadPrice.GetDF()
        self.dfPortfolioGenerationNew = ((self.PortfolioGeneration.GetDF()/1000)+dfReal) / 4
        self.dfImbalanceTakePrice = self.ImbalanceTakePrice.GetDF()
        self.dfImbalanceFeedPrice = self.ImbalanceFeedPrice.GetDF()

        arZeros = np.zeros(self.dfPortfolioForecastNew.shape,dtype=np.float64)
        dfZeros = pd.DataFrame(data=arZeros,index=self.dfPortfolioForecastNew.index,columns=self.dfPortfolioForecastNew.columns,dtype=np.float64)

        self.dfPortfolioImbalanceNew = self.dfPortfolioForecastNew - self.dfPortfolioGenerationNew
        dfPortfolioImbalanceTake = np.maximum(self.dfPortfolioImbalanceNew,dfZeros)
        dfPortfolioImbalanceFeed = np.maximum(-self.dfPortfolioImbalanceNew,dfZeros)
        self.dfPortfolioImbalanceTakeValueNew = dfPortfolioImbalanceTake * -self.dfImbalanceTakePrice
        self.dfPortfolioImbalanceFeedValueNew = dfPortfolioImbalanceFeed * self.dfImbalanceFeedPrice

        self.dfPortfolioBenchmarkNew = self.dfDayAheadPrice * self.dfPortfolioGenerationNew
        self.dfDayAheadRevenueNew = self.dfDayAheadPrice * self.dfPortfolioForecastNew
        self.dfPortfolioValueNew = self.dfDayAheadRevenueNew + self.dfPortfolioImbalanceTakeValueNew + self.dfPortfolioImbalanceFeedValueNew
        self.dfImbalanceVSBenchmarkNew = self.dfPortfolioValueNew-self.dfPortfolioBenchmarkNew
        self.dfImbalanceVSBenchmarkEURMWhNew = self.dfImbalanceVSBenchmarkNew / self.dfPortfolioGenerationNew

        self.dfPortfolioValueMonthly = self.dfPortfolioValue.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfImbalanceVSBenchmarkMonthly = self.dfImbalanceVSBenchmark.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfPortfolioGenerationMonthly = self.dfPortfolioGeneration.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfImbalanceVSBenchmarkEURMWhMonthly = (self.dfImbalanceVSBenchmarkMonthly / self.dfPortfolioGenerationMonthly)
        self.dfPortfolioValueNewMonthly = self.dfPortfolioValueNew.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfImbalanceVSBenchmarkNewMonthly = self.dfImbalanceVSBenchmarkNew.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfPortfolioGenerationNewMonthly = self.dfPortfolioGenerationNew.resample("M", how='sum').sum(axis=1).to_frame()
        self.dfImbalanceVSBenchmarkEURMWhNewMonthly = (self.dfImbalanceVSBenchmarkNewMonthly / self.dfPortfolioGenerationNewMonthly)
        self.dfMarginalVolume = self.dfPortfolioGenerationNewMonthly - self.dfPortfolioGenerationMonthly
        self.dfMarginalCost = self.dfImbalanceVSBenchmarkNewMonthly - self.dfImbalanceVSBenchmarkMonthly
        self.dfMarginalPrice = self.dfMarginalCost / self.dfMarginalVolume

        self.lPortfolioValueMonthly.append(self.dfPortfolioValueMonthly )
        self.lImbalanceVSBenchmarkMonthly.append(self.dfImbalanceVSBenchmarkMonthly)
        self.lPortfolioGenerationMonthly.append(self.dfPortfolioGenerationMonthly)
        self.lImbalanceVSBenchmarkEURMWhMonthly.append(self.dfImbalanceVSBenchmarkEURMWhMonthly)
        self.lPortfolioValueNewMonthly.append(self.dfPortfolioValueNewMonthly)
        self.lImbalanceVSBenchmarkNewMonthly.append( self.dfImbalanceVSBenchmarkNewMonthly)
        self.lPortfolioGenerationNewMonthly.append(self.dfPortfolioGenerationNewMonthly)
        self.lImbalanceVSBenchmarkEURMWhNewMonthly.append(self.dfImbalanceVSBenchmarkEURMWhNewMonthly)
        self.lMarginalVolume.append(self.dfMarginalVolume)
        self.lMarginalCost.append(self.dfMarginalCost)
        self.lMarginalPrice.append(self.dfMarginalPrice)


        lMonthArraysToWrite.append(self.dfPortfolioValueMonthly)
        lMonthArrayNames.append('Portfolio Value Monthly')
        lMonthArraysToWrite.append(self.dfImbalanceVSBenchmarkMonthly)
        lMonthArrayNames.append('Imbalance Cost')
        lMonthArraysToWrite.append(self.dfPortfolioGenerationMonthly)
        lMonthArrayNames.append('Portfolio Generation')
        lMonthArraysToWrite.append(self.dfImbalanceVSBenchmarkEURMWhMonthly)
        lMonthArrayNames.append('Imbalance Price')
        lMonthArraysToWrite.append(self.dfPortfolioValueNewMonthly)
        lMonthArrayNames.append('New Portfolio Value Monthly')
        lMonthArraysToWrite.append(self.dfImbalanceVSBenchmarkNewMonthly)
        lMonthArrayNames.append('New Imbalance Cost')
        lMonthArraysToWrite.append(self.dfPortfolioGenerationNewMonthly)
        lMonthArrayNames.append('New Portfolio Generation')
        lMonthArraysToWrite.append(self.dfImbalanceVSBenchmarkEURMWhNewMonthly)
        lMonthArrayNames.append('New Imbalance Price')
        lMonthArraysToWrite.append(self.dfMarginalCost)
        lMonthArrayNames.append('Marginal Imbalance Cost')
        lMonthArraysToWrite.append(self.dfMarginalVolume)
        lMonthArrayNames.append('Marginal Imbalance Volume')
        lMonthArraysToWrite.append(self.dfMarginalPrice)
        lMonthArrayNames.append('Marginal Imbalance Price')


        lArraysToWrite.append(self.dfPortfolioForecast)
        lArrayNames.append('Portfolio Forecast')

        lArraysToWrite.append(dfFore)
        lArrayNames.append('PV Forecast')

        lArraysToWrite.append(self.dfPortfolioForecastNew)
        lArrayNames.append('Portfolio Forecast New')

        lArraysToWrite.append(self.dfPortfolioGeneration)
        lArrayNames.append('Portfolio Generation')

        lArraysToWrite.append(dfReal)
        lArrayNames.append('PV Generation')

        lArraysToWrite.append(self.dfPortfolioGenerationNew)
        lArrayNames.append('Portfolio Generation New')

        lArraysToWrite.append(self.dfPortfolioBenchmark)
        lArrayNames.append('Portfolio Benchmark')

        lArraysToWrite.append(self.dfPortfolioBenchmarkNew)
        lArrayNames.append('Portfolio Benchmark New')

        lArraysToWrite.append(self.dfPortfolioValue)
        lArrayNames.append('Portfolio Value')

        lArraysToWrite.append(self.dfPortfolioValueNew)
        lArrayNames.append('Portfolio Value New')

        lArraysToWrite.append(self.dfImbalanceVSBenchmark)
        lArrayNames.append('Portfolio Imbalance')

        lArraysToWrite.append(self.dfImbalanceVSBenchmarkNew)
        lArrayNames.append('Portfolio Imbalance New')

        lArraysToWrite.append(self.dfImbalanceVSBenchmarkEURMWh)
        lArrayNames.append('Portfolio Imbalance MWh')

        lArraysToWrite.append(self.dfImbalanceVSBenchmarkEURMWhNew)
        lArrayNames.append('Portfolio Imbalance MWh New')

        myPath = self.SimsDirectory + " Sim - " +str(self.WriteNumber)+ ".xlsx"
        myPathMonth = self.SimsDirectory + " Monthly Sim - " +str(self.WriteNumber)+ ".xlsx"
        #writer = pd.ExcelWriter(myPath,engine='xlsxwriter')
        #for i in range(0,len(lArraysToWrite)):
        #    lArraysToWrite[i].to_excel(writer,lArrayNames[i])
        #writer.save()

        if(self.WriteHours):
            self.threads.append(myExcelWriterHourly(self.WriteNumber,lArraysToWrite,lArrayNames, myPath))
            self.threads[len(self.threads)-1].start()
        if(self.WriteMonths):
            self.threads.append(myExcelWriterHourly(self.WriteNumber,lMonthArraysToWrite,lMonthArrayNames, myPathMonth))
            self.threads[len(self.threads)-1].start()
            print("Wrote Months")
        print("Calculated Simulation Value")

    def CalculateSimulationValueDF(self):

        for sim in range(len(self.Sims)-1,len(self.Sims)):
            oTMP1 = self.Sims[sim]
            #Each simulation is made up of Thetic and Anto-Thetic
            for ath in range(0,len(oTMP1)):
                self.WriteNumber+=1
                #for each profile that is simulated
                oTMP2 = oTMP1[ath]

                dfFore = oTMP2[1]
                dfReal = oTMP2[0]
                self.CorrectColumnHeaders(dfFore,dfReal)
                self.CalculateSimulationValueDFMain(dfFore,dfReal)


    def CalculateValue(self):

        self.arPortfolioForecast = self.PortfolioForecast.GetArray()  / 4000
        self.arDayAheadPrice = self.DayAheadPrice.GetArray()
        self.arPortfolioGeneration = self.PortfolioGeneration.GetArray() / 4000
        self.arImbalanceTakePrice = self.ImbalanceTakePrice.GetArray()
        self.arImbalanceFeedPrice = self.ImbalanceFeedPrice.GetArray()

        arZeros = np.zeros(self.arPortfolioForecast.shape,dtype=np.float64)

        self.arPortfolioImbalance = self.arPortfolioForecast - self.arPortfolioGeneration
        arPortfolioImbalanceTake = np.maximum(self.arPortfolioImbalance,arZeros)
        arPortfolioImbalanceFeed = np.maximum(-self.arPortfolioImbalance,arZeros)
        self.arPortfolioImbalanceTakeValue = arPortfolioImbalanceTake * -self.arImbalanceTakePrice
        self.arPortfolioImbalanceFeedValue = arPortfolioImbalanceFeed * self.arImbalanceFeedPrice

        self.arPortfolioBenchmark = self.arDayAheadPrice * self.arPortfolioGeneration
        self.arDayAheadRevenue = self.arDayAheadPrice * self.arPortfolioForecast
        self.arPortfolioValue = self.arDayAheadRevenue + self.arPortfolioImbalanceTakeValue + self.arPortfolioImbalanceFeedValue
        self.arImbalanceVSBenchmark = self.arPortfolioValue-self.arPortfolioBenchmark
        self.arImbalanceVSBenchmarkEURMWh = self.arImbalanceVSBenchmark / self.arPortfolioGeneration

        print("Calculated Value")

    def CalculateSimulationValue(self):

        for sim in range(0,len(self.Sims)):
            oTMP1 = self.Sims[sim]
            #Each simulation is made up of Thetic and Anto-Thetic
            for ath in range(0,len(oTMP1)):
                self.WriteNumber+=1
                #for each profile that is simulated
                oTMP2 = oTMP1[ath]

                arFore = oTMP2[1].as_matrix()
                arReal = oTMP2[0].as_matrix()

                self.arPortfolioForecastNew = (self.PortfolioForecast.GetArray() + arFore)/ 4000
                self.arDayAheadPrice = self.DayAheadPrice.GetArray()
                self.arPortfolioGenerationNew = (self.PortfolioGeneration.GetArray() + arReal)/ 4000
                self.arImbalanceTakePrice = self.ImbalanceTakePrice.GetArray()
                self.arImbalanceFeedPrice = self.ImbalanceFeedPrice.GetArray()

                arZeros = np.zeros(self.arPortfolioForecast.shape,dtype=np.float64)

                self.arPortfolioImbalanceNew = self.arPortfolioForecastNew - self.arPortfolioGenerationNew
                arPortfolioImbalanceTake = np.maximum(self.arPortfolioImbalanceNew,arZeros)
                arPortfolioImbalanceFeed = np.maximum(-self.arPortfolioImbalanceNew,arZeros)
                self.arPortfolioImbalanceTakeValueNew = arPortfolioImbalanceTake * -self.arImbalanceTakePrice
                self.arPortfolioImbalanceFeedValueNew = arPortfolioImbalanceFeed * self.arImbalanceFeedPrice

                self.arPortfolioBenchmarkNew = self.arDayAheadPrice * self.arPortfolioGenerationNew
                self.arDayAheadRevenueNew = self.arDayAheadPrice * self.arPortfolioForecastNew
                self.arPortfolioValueNew = self.arDayAheadRevenueNew + self.arPortfolioImbalanceTakeValueNew + self.arPortfolioImbalanceFeedValueNew
                self.arImbalanceVSBenchmarkNew = self.arPortfolioValueNew-self.arPortfolioBenchmarkNew
                self.arImbalanceVSBenchmarkEURMWhNew = self.arImbalanceVSBenchmarkNew / self.arPortfolioGenerationNew

        print("Calculated Value")


    def WriteConcatMonths(self):
        lArrays = []
        lNames = []
        lArrays.append(pd.concat(self.lPortfolioValueMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lImbalanceVSBenchmarkMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lPortfolioGenerationMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lImbalanceVSBenchmarkEURMWhMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lPortfolioValueNewMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lImbalanceVSBenchmarkNewMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lPortfolioGenerationNewMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lImbalanceVSBenchmarkEURMWhNewMonthly, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lMarginalVolume, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lMarginalCost, axis=1, join='inner'))
        lArrays.append(pd.concat(self.lMarginalPrice, axis=1, join='inner'))
        lNames.append("Portfolio Value")
        lNames.append("Imbalance Cost")
        lNames.append("Portfolio Generation")
        lNames.append("Imbalance Price")
        lNames.append("Portfolio Value New")
        lNames.append("Imbalance Cost New")
        lNames.append("Portfolio Generation New")
        lNames.append("Imbalance Price New")
        lNames.append("Marginal Volume")
        lNames.append("Marginal Cost")
        lNames.append("Marginal Price")
        myPathMonth = self.SimsDirectory + " Monthly Sims.xlsx"
        self.threads.append(myExcelWriterHourly(self.WriteNumber,lArrays,lNames, myPathMonth))
        self.threads[len(self.threads)-1].start()
class myExcelWriter (threading.Thread):
    def __init__(self, threadID,  inTMP,inPath):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.myPath = inPath
        self.oTMP = inTMP
    def run(self):
        self.oTMP.to_excel(self.myPath,'Sheet1', engine='xlsxwriter')

class myExcelWriterHourly (threading.Thread):
    def __init__(self, threadID, inArrays, inNames,inPath):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.myPath = inPath
        self.oArrays = inArrays
        self.oNames = inNames
    def run(self):
        writer = pd.ExcelWriter(self.myPath,engine='xlsxwriter')
        for i in range(0,len(self.oArrays)):
            self.oArrays[i].to_excel(writer,self.oNames[i])
        writer.save()
