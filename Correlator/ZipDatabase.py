
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

class ZipDatabase:

    def __init__(self,FilePath):

        self.archive = zipfile.ZipFile(FilePath, 'r')
        print("initialised Zip Database")

    def xml2df(self, xml_data):

        root = ET.XML(xml_data) # element tree
        all_records = [] #This is our record list which we will convert into a dataframe
        for i, child in enumerate(root): #Begin looping through our root tree
            record = {} #Place holder for our record
            for subchild in child: #iterate through the subchildren to user-agent, Ex: ID, String, Description.
                record[subchild.tag] = subchild.text.replace(',','.') #Extract the text create a new dictionary key, value pair
                all_records.append(record) #Append this record to all_records.

        return pd.DataFrame(all_records) #return records as DataFrame

    def GetDFs(self):
        self.DFs = []

        for i in self.archive.namelist():
            f=self.archive.open(i)

            content=f.read()

            f.close()

            DF = self.xml2df(content)
            #DF = DF[['Data','Ora','CNOR','CSUD','NORD','SARD','SICI','SUD']]
            #DF.dropna(inplace=True)
            #tmp = pd.to_datetime(DF[DF.columns[0]],format='%Y%m%d', errors='coerce')
            #DF[DF.columns[0]] = tmp + pd.to_timedelta(DF[DF.columns[1]].astype(int),unit='h')
            #DF.set_index('Data',inplace=True)
            #DF = DF.drop('Ora', 1)
            #DF = DF[~DF.index.duplicated(keep='first')]

            self.DFs.append(DF)

        return self.DFs
