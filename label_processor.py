import os
import re
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Optional, Tuple, Union, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection, metrics
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   FunctionTransformer,
                                   MultiLabelBinarizer,
                                   LabelBinarizer,
                                   StandardScaler)
from torch import nn
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import CountVectorizer

questions_by_id = {
    '64': 'Did you find this conversation helpful?',
    '65': 'How helpful was it?',
    '66': 'Would you like to leave a note for your Crisis Volunteer',
    '69': 'How old are you?',
    '73': 'Gender. Do you consider yourself to be:',
    '74': 'Sexual orientation. Do you consider yourself to be:',
    '75': 'What is your race or origin?',
    '151': 'Have you ever served in the military?',
    '152': 'In which branches have you served?',
    '205': 'Do you have a condition or disability that may require accommodations?',
    '77': 'Feeling nervous or anxious over the last 2 weeks?',
    '78': 'Cannot stop worrying over the last 2 weeks?',
    '79': 'Little interest in doing things over the last 2 weeks?',
    '80': 'Feeling depressed or hopeless over the last 2 weeks?',
    '81': 'Feeling left out over the last 2 weeks?',
    '82': 'Feeling isolated from others over the last 2 weeks?',
    '84': 'Why did you text us today?',
    '144': 'How did you learn about us?',
    '145': 'Where in the media did you learn about us?',
    '85': 'Beside texting us, how else do you get help when in crisis?',
    '86': 'Who do you talk to?',
    '88': 'Consider your feelings below, and let us know if they changed after you texted with a Crisis Volunteer today.',
    '96': 'In your conversation, did you mention an experience or feelings that you have not shared with anyone else?',
    '97': 'In the conversation, I believe my Crisis Volunteer was genuinely concerned for my well-being.',
    '98': 'Did you and your crisis volunteer agree on a plan to follow at the end of your conversation?',
    '99': 'How likely are you to follow these plans?',
    '312': 'Are you self-isolating at home because you have, or someone in your household has, symptoms of coronavirus?'
}


def list_series_values(ser):
    a = set()
    for i in ser.dropna():
        for j in i:
            a.add(j)
    return a


def list_series_value_counts(ser):
    a = defaultdict(lambda: 0)
    for i in ser.dropna():
        for j in i:
            a[j] += 1
    return dict(a)


class LabelProcessorSimplified:
    ordinals = [
        ['Not at all', 'More than half the time', 'Several days', 'Nearly every day'],
        ['1 (slightly helpful)', '2', '3', '4', '5 (very helpful)'],
        ['No', '1 (slightly helpful)', '2', '3', '4', '5 (very helpful)'],
        ['13 or younger', '14-24', '25-44', '45-64', '65+'],
        ['14-17', '18-21', '22-24', '25-34', '35-44', '45-54', '55-64'],
        ['More', 'Same', 'Less'],
        ['No', 'Not sure', 'Yes'],
        ['I will', 'I might', 'Not likely'],
        ['Strongly agree', 'Somewhat agree', 'Neither agree nor disagree', 'Somewhat disagree', 'Strongly disagree'],
        ["good", "ehhh", "bad", "upset"],
    ]
    ordinals_frozen = [frozenset(i) for i in ordinals]

    def __init__(self,
                 label_df: pd.DataFrame,
                 column_threshold: int,
                 answer_threshold: int,
                 merge_ages=True,
                 remove_prefer_not_to_answer=True,
                 all_questions=True):
        assert label_df.index.name == "conversation_id"
        self.raw_df = label_df.copy()
        self.raw_df.columns = [str(i) for i in self.raw_df.columns]
        self.column_threshold = column_threshold
        self.answer_threshold = answer_threshold

        if merge_ages:
            self.merge_ages()
        if remove_prefer_not_to_answer:
            self.remove_prefer_not_to_answer()
        self.all_questions = self.get_questions(all_questions)

        self.threshold_columns(self.column_threshold)

        self.questions_to_do = set(self.all_questions.copy())
        self.label_questions = []
        self.df = self.raw_df[[]].copy()  # Keep only index

        self.subcols = defaultdict(list)
        self.col_types = {}

        self.vectorizers = {}
        self.label_weights = None

        self.mse_questions = []
        self.binary_questions: list
        self.softmax_questions: list

    def get_binary_softmax_indxs(self):
        binary_questions = [i for i, j in self.col_types.items() if j != "categorical"]
        binary_questions = [i for j in binary_questions for i in self.subcols[j]]
        binary_questions = [self.label_questions.index(i) for i in binary_questions]
        self.binary_questions = binary_questions

        softmax_questions = [i for i, j in self.col_types.items() if j == "categorical"]
        softmax_questions = [self.subcols[i] for i in softmax_questions]
        softmax_questions = [[self.label_questions.index(i) for i in s] for s in softmax_questions]
        self.softmax_questions = softmax_questions

        # Temporarily, bc I'm being lazy
        self.binary_questions += [i for j in self.softmax_questions for i in j]

        self.binary_questions.sort()

        # [x[:,i] for i in softmax_questions]
        # x[:,binary_questions]

    def get_label_weights(self):
        self.label_weights = {}

        for _, l in self.subcols.items():
            w = 1 / len(l)
            for i in l:
                self.label_weights[i] = w

    def remove_analyzers(self):
        # For pickling
        for vec in self.vectorizers.values():
            vec.analyzer = None

    def drop_empty_rows(self):
        all_labels_are_na = self.df[self.label_questions].isna().all(1)
        self.df.drop(all_labels_are_na[all_labels_are_na].index, inplace=True)

    def add_union(self, col, vals: Union[list, str], name=None, substring=False):
        if type(vals) == str and name is None:
            name = f"{col}_{vals}"
            vals = [vals]
        col = self.raw_df[col].dropna()
        checkbox = type(col.values[0]) == list
        vals = set(vals)
        if checkbox and substring:
            col = col.apply(" ".join).str.lower()
        if checkbox and not substring:
            new_col = col.apply(lambda a: len(vals.intersection(a)) > 0)
        elif substring:
            vals = [val.lower() for val in vals]
            new_col = col.apply(lambda a: sum(val in a for val in vals) > 0)
        else:
            new_col = col.apply(lambda a: a in vals)
        self.df.loc[new_col.index, name] = new_col
        self.subcols[col.name].append(name)
        self.label_questions.append(name)

    def convert_mse(self, question: Union[list, str], super_col=None):
        if type(question) != list:
            question = [question]
        for q in question:
            self.questions_to_do.remove(q)
            self.df[q] = self.raw_df[q]
            self.label_questions.append(q)
            self.mse_questions.append(self.label_questions.index(q))
        if super_col is not None:
            self.subcols[super_col] += question
        else:
            self.subcols[question] += question

    def convert_free_response(self, question, max_features):
        self.questions_to_do.remove(question)
        vectorizer = CountVectorizer(analyzer=get_stemming_analyzer(), binary=True, max_features=max_features)
        self.vectorizers[question] = vectorizer

        col = self.raw_df[question].dropna()
        if type(col.values[0]) == list:
            col = col.apply(" ".join)
        a = vectorizer.fit_transform(col)
        vectorizer.get_feature_names()
        new_col_names = [f"{question}_{i}" for i in vectorizer.get_feature_names()]
        new_columns = pd.DataFrame(a.toarray(), columns=new_col_names, index=col.index)
        self.df = self.df.join(new_columns, how="outer")
        self.label_questions += new_col_names
        self.subcols[question] += new_col_names
        self.col_types[question] = "free response"

    def convert_all_checkboxs_to_binary(self, threshold):
        cols = [c for c in self.questions_to_do if type(self.raw_df[c].dropna().values[0]) == list]
        for col in tqdm(cols, desc="convert_checkbox_to_binary", leave=False):
            self.questions_to_do.remove(col)
            col = self.raw_df[col].dropna().copy()
            vc = list_series_value_counts(col)
            values = {col if i >= threshold else "other" for col, i in
                      vc.items()}  # set, so that "other" only appears once
            if len(values) < 2:
                continue

            self.col_types[col.name] = "checkbox"
            col = col.apply(lambda a: {i if i in values else "other" for i in a})
            for v in values:
                new_col = f"{col.name}_{v}"
                self.df.loc[col.index, new_col] = col.apply(lambda a: v in a).astype(float)
                self.label_questions.append(new_col)
                self.subcols[col.name].append(new_col)

            # df.loc[col.index, f"{col.name}_size"] = col.apply(len)
            # self.questions.append(f"{col.name}_size")

    def convert_all_ordinals_to_binary(self):
        cols = [c for c in self.questions_to_do if frozenset(self.raw_df[c].dropna().unique()) in self.ordinals_frozen]
        for col in tqdm(cols, desc="convert_ordinal_to_binary", leave=False):
            u_frozen = frozenset(self.raw_df[col].dropna().unique())
            order = self.ordinals[self.ordinals_frozen.index(u_frozen)]
            self.convert_ordinal_to_binary(col, order)

    def convert_ordinal_to_binary(self, col, order):
        self.questions_to_do.remove(col)
        col = self.raw_df[col].dropna()
        self.col_types[col.name] = "ordinal"

        for i, v in enumerate(order[:-1]):
            new_col = f"{col.name}>{v}"
            self.df.loc[col.index, new_col] = col.apply(lambda a: order.index(a) > i).astype(float)
            self.label_questions.append(new_col)
            self.subcols[col.name].append(new_col)

    def convert_remaining_to_categorical_binary(self, threshold):
        # RUN AFTER CHECKBOX and ORDINAL
        cols = list(self.questions_to_do)
        for col in tqdm(cols, desc="convert_categorical_to_binary", leave=False):
            self.questions_to_do.remove(col)
            col = self.raw_df[col].dropna()
            vc = col.value_counts()
            values = {col if i >= threshold else "other" for col, i in
                      vc.items()}  # set, so that "other" only appears once
            if len(values) < 2:
                continue

            self.col_types[col.name] = "categorical"
            col = col.apply(lambda a: a if a in values else "other")
            if len(values) == 2:
                a = [i for i in values if (type(i) == str and i.lower() == "yes") or (type(i) != str and i)]
                values = a if len(a) == 1 else list(values)[:1]
                self.col_types[col.name] = "binary"
            for v in values:
                new_col = f"{col.name}_{v}"
                self.df.loc[col.index, new_col] = col.apply(lambda a: a == v).astype(float)
                self.label_questions.append(new_col)
                self.subcols[col.name].append(new_col)

    def merge_ages(self):
        df = self.raw_df
        i70 = df[~df['70'].isna()]
        i71 = df[~df['71'].isna()]
        i72 = df[~df['72'].isna()]

        i70 = i70[i70[['71', '72']].isna().all(1)]
        i71 = i71[i71[['70', '72']].isna().all(1)]
        i72 = i72[i72[['70', '71']].isna().all(1)]

        df.loc[i70.index, 'age2'] = i70['70']
        df.loc[i71.index, 'age2'] = i71['71']
        df.loc[i72.index, 'age2'] = i72['72']

        df.drop(['70', '71', '72'], axis="columns", inplace=True)

    def remove_prefer_not_to_answer(self):
        self.raw_df.replace('Prefer not to answer', np.nan, inplace=True)

    def get_questions(self, all_questions):
        df = self.raw_df
        if all_questions:
            return df.columns.tolist()
        all_questions = df.columns[
            [re.fullmatch("\\d+", i) is not None for i in df]]
        all_questions = list(all_questions) + ['age2', 'age_re']
        all_questions = [i for i in all_questions if i in df.columns]
        return all_questions

    def threshold_columns(self, num_answers_threshold: int = 500):
        # Remove columns where there aren't sufficiently many values
        for c in self.all_questions:
            if self.raw_df[c].count() < num_answers_threshold:
                self.all_questions.remove(c)

    def take_intersection_df(self, df_other: pd.DataFrame):

        features_self = list(self.df.columns)
        features_other = list(df_other.columns)

        assert df_other.index.name == "conversation_id"
        assert self.df.index.name == "conversation_id"

        df = self.df.copy().join(df_other, on="conversation_id", how="inner")

        all_idxs = np.arange(df.shape[0])
        df['id'] = all_idxs
        df = df.set_index('id')
        self.df = df[features_self]
        X = df[features_other]
        return X


def get_stemming_analyzer():
    import nltk.stem
    from sklearn.feature_extraction.text import CountVectorizer
    stemmer = nltk.stem.PorterStemmer()
    temp = CountVectorizer(stop_words='english', strip_accents='ascii')
    tokenizer = temp.build_analyzer()
    stopwords = temp.get_stop_words()  # SKLearn List
    stopwords = stopwords.union(
        nltk.corpus.stopwords.words('english')  # NLTK list
    )
    stopwords = stopwords.union(
        ['go', 'texter', 'bit', 'www', 'uk', "counsellor", "gave", "give", "given", "got", "let", "said", "say", "way"]
        # Manual list
    )

    def stemmed_words(doc):
        l = [stemmer.stem(w) for w in tokenizer(doc) if w.lower() not in stopwords]
        return [i for i in l if i not in stopwords]

    return stemmed_words


class WeightedCrossEntropy(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropy, self).__init__()
        self.cross_entropy_layer = nn.CrossEntropyLoss(reduction='none')
        self.weights = weights if type(weights) == torch.Tensor else torch.tensor(weights)

    def forward(self, x, target):
        loss = self.cross_entropy_layer(x, target)
        weights = self.weights[target]
        return (loss * weights).mean(0)
