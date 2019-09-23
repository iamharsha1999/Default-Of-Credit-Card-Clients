import csv
import os

cwd = os.getcwd()
#print(cwd)

for file in os.listdir('/home/harsha/Machine_Learning_Project/Dataset'):  # use the directory name here

    file_name, file_ext = os.path.splitext(file)

    if file_ext == '.csv':
        with open('/home/harsha/Machine_Learning_Project/Dataset/' + file,'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            # csv_reader.next()  ## skip one line (the first one)

            newfile = file + '.txt'

            for line in csv_reader:
                with open(newfile, 'a') as new_txt:    #new file has .txt extn
                    txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
                    txt_writer.writerow(line)