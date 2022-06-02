from pathlib import Path
import email
import pandas as pd
import numpy as np


def clean_data(data_location:str):
    data_folder = Path(data_location)
    files = data_folder.glob('./*/*.txt')
    data = [(i.name.split('.')[0], i.read_text()) for i in files]
    data_frame = pd.DataFrame(data, columns = ['file_id','content'])
    class_tags_files = data_folder.glob('./*/*.cats')
    data_cat = [(i.name.split('.')[0], i.read_text().split('\n')[0]) for i in class_tags_files]
    tag_dataframe = pd.DataFrame(data_cat, columns =['file_id','class_tag'])
    email_dataframe = data_frame.set_index('file_id').join(tag_dataframe.set_index('file_id'))
    email_dataframe['class_tag'] = email_dataframe['class_tag'].astype('str')
    email_dataframe['class_tag'] = email_dataframe['class_tag'].apply(lambda x: int(x.split(',')[1].strip()) if x.split(',')[0].strip() == '1' else np.NaN)
    email_dataframe['class_tag'] = email_dataframe['class_tag'].astype('int')
    email_dataframe.reset_index(inplace=True)
    messages = list(map(email.message_from_string, email_dataframe['content']))
    email_dataframe['email_text'] = list(map(get_content_from_email, messages))
    email_dataframe.drop(index = email_dataframe.query("class_tag == 7 or class_tag == 8").index, inplace=True)
    email_dataframe.reset_index(inplace=True)
    # email_dataframe.drop(columns='content', inplace=True)
    email_dataframe.to_csv('final_data.csv', index= False)
    return None

def get_content_from_email(email_content):
    '''get the content from email objects'''
    parts = []
    for sub_part in email_content.walk():
        if sub_part.get_content_type() == 'text/plain':
            payload = sub_part.get_payload()
            payload = ' '.join(payload.split())
            parts.append( payload )
    return ''.join(parts)

if __name__ == '__main__':
    clean_data('./enron_with_categories')
