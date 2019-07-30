import os
import pandas as pd

# this class convert Text to CSV as per the requirement
class TextToCsv(object):

	#this function reads data and make dataframe
	def getdata(self, dir):
		files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
		row = []
		for file in files:
			filepath = os.path.join(dir, file)
			with open(filepath, 'rb') as file:
				txt= file.read()
				txt= txt.decode('utf-8').replace('\n', ' ')
				filename = filepath.split("\\")
				filename = filename[len(filename)-1]
				classname = filename.split(".")[0]
				row.append([txt,classname, filename])
				data = pd.DataFrame(row,columns=['Content','Classname','Filename'])
		return data

	def getdirdata(self, dir):
		subdir = [dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
		row = []
		for dirname in subdir:
			file_path= os.path.join(dir,dirname)			
			files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
			for file in files:
				filepath = os.path.join(file_path, file)
				with open(filepath, 'rb') as file:
					txt= file.read()
					txt= txt.decode('utf-8').replace('\n', ' ')
					filename = filepath.split("\\")
					filename = filename[len(filename)-1]
					# classname = filename.split("_")[0]
					# classname = filename.split("_")[1]
					classname1 = filename.split("_")[0]
					classname2 = filename.split("_")[1]
					classname = classname1 + '_' + classname2
					row.append([txt,classname, filename])
					data = pd.DataFrame(row,columns=['Content','Classname','Filename']).fillna(' ')
		return data

	#this function save the data in CSV
	def getcsv(self, data, save_location):
		csvfile = save_location +r"\rawdata.csv"
		data.to_csv(csvfile, index = False)
		return csvfile