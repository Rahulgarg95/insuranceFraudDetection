from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger
from AwsS3Storage.awsStorageManagement import AwsStorageManagement

class dataTransformPredict:

     """
          This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.
     """

     def __init__(self):
          self.goodDataPath = "Prediction_Good_Raw_Files_Validated"
          self.logger = App_Logger()
          self.awsObj = AwsStorageManagement()

     def replaceMissingWithNull(self):

          """
              Method Name: replaceMissingWithNull
              Description: This method replaces the missing values in columns with "NULL" to
                           store in the table. We are using substring in the first column to
                           keep only "Integer" data for ease up the loading.
                           This column is anyways going to be removed during prediction.
          """

          try:
               log_file = 'dataTransformLog'
               onlyfiles = self.awsObj.listDirFiles(self.goodDataPath)
               for file in onlyfiles:
                    data = self.awsObj.csvToDataframe(self.goodDataPath, file)
                    # list of columns with string datatype variables
                    columns = ["policy_bind_date","policy_state","policy_csl","insured_sex","insured_education_level","insured_occupation","insured_hobbies","insured_relationship","incident_state","incident_date","incident_type","collision_type","incident_severity","authorities_contacted","incident_city","incident_location","property_damage","police_report_available","auto_make","auto_model"]

                    for col in columns:
                         data[col] = data[col].apply(lambda x: "'" + str(x) + "'")
                    self.awsObj.saveDataframeToCsv(self.goodDataPath, file, data)
                    self.logger.log(log_file," %s: File Transformed successfully!!" % file)
          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               raise e