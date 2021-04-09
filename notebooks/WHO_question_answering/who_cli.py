import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import tensorflow_hub as hub
import tensorflow_text
import re

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from timeit import default_timer as timer

urls = [
    "https://www.who.int/news-room/q-a-detail/q-a-coronaviruses",
    "https://www.who.int/news-room/q-a-detail/q-a-on-covid-19-pregnancy-childbirth-and-breastfeeding",
    "https://www.who.int/news-room/q-a-detail/q-a-on-covid-19-hiv-and-antiretrovirals",
    "https://www.who.int/news-room/q-a-detail/q-a-similarities-and-differences-covid-19-and-influenza",
    "https://www.who.int/news-room/q-a-detail/q-a-on-mass-gatherings-and-covid-19",
    "https://www.who.int/news-room/q-a-detail/q-a-on-smoking-and-covid-19",
    "https://www.who.int/news-room/q-a-detail/be-active-during-covid-19",
    "https://www.who.int/news-room/q-a-detail/malaria-and-the-covid-19-pandemic",
    "https://www.who.int/news-room/q-a-detail/q-a-on-infection-prevention-and-control-for-health-care-workers-caring-for-patients-with-suspected-or-confirmed-2019-ncov",
    "https://www.who.int/news-room/q-a-detail/middle-east-respiratory-syndrome-coronavirus-(mers-cov)",
]


def iter_who_faq(url="https://www.who.int/news-room/q-a-detail/q-a-coronaviruses"):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "lxml")
    for panel in soup.find_all(attrs={"class": "sf-accordion__panel"}):
        question = panel.find_all("a")[0].text.strip()
        answers = [p.text.strip() for p in panel.find_all("p") if p.text.strip()] + [
            list_element.text.strip() for list_element in panel.find_all("li")
        ]
        if not answers:
            print(url, question)
        for answer in answers:
            yield question, answer, url


def iter_who_faqs(urls=urls):
    for url in urls:
        for out in iter_who_faq(url=url):
            yield out


def preprocess_sentences(input_sentences):
    return [
        re.sub(r"(covid-19|covid)", "coronavirus", input_sentence, flags=re.I)
        for input_sentence in input_sentences
    ]


def main():
    print("Reading WHO FAQ websites")
    data = pd.DataFrame(
        data=list(iter_who_faqs()), columns=["Context", "Answer", "Source"]
    )

    print("Loading TensorFlow Multilingual Universal Sentence Encoder model")
    module = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
    )

    print("Encoding FAQs")
    response_encodings = module.signatures["response_encoder"](
        input=tf.constant(preprocess_sentences(data.Answer)),
        context=tf.constant(preprocess_sentences(data.Context)),
    )["outputs"]

    question_str = "Input your question (leave blank to exit)"
    print(question_str)
    question = input()
    while question:
        start = timer()
        question_encoding = module.signatures["question_encoder"](
            tf.constant(preprocess_sentences([question]))
        )["outputs"]
        test_response = data.loc[
            np.argmax(np.inner(question_encoding, response_encodings), axis=1),
            ["Answer", "Source"],
        ]
        end = timer()
        print(test_response["Answer"].values[0])
        print("This response was gathered from:")
        print(test_response["Source"].values[0])
        print("This query took {} seconds".format(end - start))
        question = input()


if __name__ == "__main__":
    main()
