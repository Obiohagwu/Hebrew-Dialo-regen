//This first block is to download the dataset from a opensubtitles. Given that we are trying to make a conversational bot. We need a dataset of dialogues to train GPT-small.

! wget -q -O he.txt.gz https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.he.gz
! gunzip -k he.txt.gz
! mkdir linesobi123
! split -a 3 -l 100000 he.txt linesobi123/lines-

! git clone https://github.com/PolyAI-LDN/conversational-datasets.git
! pip install -q -r conversational-datasets/requirements.txt


// Mount google drive at contents to save processed dataset.

from google.colab import drive
drive.mount('/content/drive')


//Changed format from txt to json for easier management

! python conversational-datasets/opensubtitles/create_data.py --runner DirectRunner --sentence_files linesobi123/lines-* --output_dir output --dataset_format JSON

from glob import glob
import pandas as pd

def remove_white_space(x):
    return (x.encode('utf-8').replace(' .', '.')
            .replace(' .', '.')
            .replace(' ,', ',')
            .replace(' ?', '?')
            .replace('¿ ', '¿')
            .replace(' !', '!')
            .replace('¡ ', '¡')
            )

// function to convert from jsonlist to dataframe for CSV

def jsonl_list_to_dataframe(file_list, columns=[
        'response', 'context', 'context/0', 'context/1',
        'context/2', 'context/3', 'context/4', 'context/5',
        'context/6', 'context/7', 'context/8', 'context/9'
    ]):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f,
                                   orient='records', encoding='utf-8',
                                   lines=True)[columns]
                      for f in file_list], sort=False)

df = jsonl_list_to_dataframe(glob("output/train*.json"), ).dropna()
df = df.drop_duplicates()
df = df.applymap(remove_white_space)

#To display first few elements of dataframe
df.head()

//Finally we are converting the dataframe to CSV and saving it at location data in drive.

df.to_csv('/content/drive/My Drive/data/final_he_conv.csv', index = False, encoding = 'utf-8')







