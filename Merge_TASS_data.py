# imports
import pandas as pd
import glob
import os
import xml.etree.ElementTree as ET

# Function for xml datasets extraction
def xml_extraction(path):
    pdFrame = pd.DataFrame({'ID':[], 'Text':[],'Tag':[]})
    row=0
    for filepath in glob.glob(os.path.join(path, '*.xml')):
        print(filepath)
        tree = ET.parse(filepath)
        eroot = tree.getroot() # the eroot of the complete tree transformed xml
       # # turn this tree represeentation of the xml into a dataframe
        for tweet in eroot:
            tweet_id = 'ID:'+tweet.find('tweetid').text
            tweetText = tweet.find('content').text
            lang = tweet.find('lang').text
            polarity_value = tweet.find('sentiment').find('polarity').find('value').text
            if lang == 'es':
                pdFrame.loc[row] = [tweet_id,tweetText,polarity_value]
                row+=1
    return pdFrame

# Applying function to TASS files
tass2019 = xml_extraction("/TASS2019")
tass2012 = xml_extraction("/TASS2012")
tass2020 = pd.read_csv("TASS2020/TASS2020.csv", encoding='utf8').reset_index(drop=True)

# Join all TASS datasets
AllTassDf =  pd.concat([tass2012, tass2019, tass2020], ignore_index=True)

# Exporting dataframe as csv file
tassDf.to_csv("ALLdTassDF.csv")
