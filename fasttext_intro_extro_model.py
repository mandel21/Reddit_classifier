import fasttext
import polars as pl
import spacy
from imblearn.over_sampling import RandomOverSampler

class TextProcessor:
    def __init__(self, csv_file_path):
        self.df_extro_intro = pl.read_csv(csv_file_path, has_header=True)
        self.nlp = spacy.load('en_core_web_sm')

    def lemmatize_posts(self):
        for i in range(len(self.df_extro_intro.head())):
            self.df_extro_intro['post'][i] = " ".join([token.lemma_ for token in self.nlp(self.df_extro_intro.item(row=i, column="post"))])

    def drop_id_column(self):
        self.df_extro_intro_dropID = self.df_extro_intro.drop(["auhtor_ID"])

    def preprocess_data(self):
        self.lemmatize_posts()
        self.drop_id_column()
        print(self.df_extro_intro.describe())

    



    def oversample_data(self, target_column="extrovert"):
        X = self.df_extro_intro_dropID.drop(target_column).to_numpy()
        y = self.df_extro_intro_dropID[target_column].to_numpy()

        oversampler = RandomOverSampler(random_state=123)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)

        self.df_extro_intro_oversampled = pl.DataFrame({"post": X_resampled[:, 0], target_column: y_resampled})


    def split_train_test(self, train_size=30000):
        self.df_extro_intro_dropID = self.df_extro_intro_dropID.sample(fraction=1, shuffle=True, seed=123)
        self.train_data, self.test_data = self.df_extro_intro_dropID.head(train_size), self.df_extro_intro_dropID.tail(-train_size)
        print(len(self.train_data))

    def write_to_txt(self, dataframe, txt_file_path):
        with open(txt_file_path, "w", encoding="utf-8") as f:
            for i in range(len(dataframe)):
                label = "__label__" + str(dataframe.item(row=i, column="extrovert"))
                post = dataframe.item(row=i, column="post")
                f.write(f"{label} {post}\n")

class TextModelTrainer:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model = None

    def train_model(self, epochs=50, word_ngrams=2, learning_rate=1.0):
        self.model = fasttext.train_supervised(input=self.train_file_path, epoch=epochs, wordNgrams=word_ngrams, lr=learning_rate)

    def save_model(self, model_file_path):
        if self.model is not None:
            self.model.save_model(model_file_path)

    def test_model(self, test_file_path):
        if self.model is not None:
            result = self.model.test(test_file_path)
            print("Test Accuracy:", result[1])
            print(*result)

class TextModelTrainer:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model = None

    def train_model(self, epochs=50, word_ngrams=2, learning_rate=1.0):
        self.model = fasttext.train_supervised(input=self.train_file_path, epoch=epochs, wordNgrams=word_ngrams, lr=learning_rate)

    def save_model(self, model_file_path):
        if self.model is not None:
            self.model.save_model(model_file_path)

    def test_model(self, test_file_path):
        if self.model is not None:
            result = self.model.test(test_file_path)
            print("Test Accuracy:", result[1])
            print(*result)



# Example usage:
text_processor = TextProcessor("extrovert_introvert.csv")
text_processor.preprocess_data()
text_processor.split_train_test()

text_processor.write_to_txt(text_processor.train_data, "train_data.txt")
text_processor.write_to_txt(text_processor.test_data, "test_data.txt")


model_trainer = TextModelTrainer("train_data.txt", "test_data.txt")
model_trainer.train_model()
model_trainer.save_model('text_classification_model.bin')
model_trainer.test_model("test_data.txt")
