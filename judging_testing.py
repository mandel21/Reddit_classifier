import fasttext
import polars as pl
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, judging_file_path, extro_intro_file_path):
        self.df_judgement = pl.read_csv(judging_file_path, has_header=True)
        self.df_extro_intro = pl.read_csv(extro_intro_file_path, has_header=True)

    def filter_common_ids(self):
        common_ids = set(self.df_judgement['auhtor_ID'].to_list()) & set(self.df_extro_intro['auhtor_ID'].to_list())

        self.df_extro_intro_no_commonids = self.df_extro_intro.filter(~pl.col('auhtor_ID').is_in(common_ids))
        self.df_extro_intro_only_commonids = self.df_extro_intro.filter(pl.col('auhtor_ID').is_in(common_ids))

    def write_to_csv(self, dataframe, csv_file_path):
        dataframe.write_csv(csv_file_path)

class TextDataProcessor:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def write_to_txt(self, dataframe, txt_file_path, target_column="extrovert"):
        with open(txt_file_path, "w", encoding="utf-8") as f:
            for i in range(len(dataframe)):
                label = "__label__" + str(dataframe.item(row=i, column=target_column))
                post = dataframe.item(row=i, column="post")
                f.write(f"{label} {post}\n")

class Hypo2ModelTrainer:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model = None

    def train_model(self, epochs=50, word_ngrams=2, learning_rate=0.5, dimensions=100):
        self.model = fasttext.train_supervised(
            input=self.train_file_path, epoch=epochs, wordNgrams=word_ngrams, lr=learning_rate, dim=dimensions
        )

    def evaluate_model(self):
        if self.model is not None:
            accuracy = self.model.test(self.test_file_path)
            print("Test accuracy of hypo2 model:", accuracy[1])
            self.model.test_label(self.test_file_path)

# Example usage:
judging_file_path = "judging_perceiving.csv"
extro_intro_file_path = "extrovert_introvert.csv"

data_processor = DataProcessor(judging_file_path, extro_intro_file_path)
data_processor.filter_common_ids()

# Save filtered dataframes to CSV files
data_processor.write_to_csv(data_processor.df_extro_intro_no_commonids, 'df_extro_intro_no_commonids.csv')
data_processor.write_to_csv(data_processor.df_extro_intro_only_commonids, 'df_extro_intro_only_commonids.csv')

# Split dataframes into train and test sets
train_data, test_data = train_test_split(data_processor.df_extro_intro_no_commonids, test_size=0.2, random_state=123)

text_processor_train = TextDataProcessor(train_data, None)
text_processor_train.write_to_txt(train_data, "train_hypo2_data.txt")

text_processor_test = TextDataProcessor(None, data_processor.df_extro_intro_only_commonids)
text_processor_test.write_to_txt(data_processor.df_extro_intro_only_commonids, "test_hypo2_data.txt")

hypo2_trainer = Hypo2ModelTrainer("train_hypo2_data.txt", "test_hypo2_data.txt")
hypo2_trainer.train_model()
hypo2_trainer.evaluate_model()

    


