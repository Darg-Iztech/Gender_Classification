import gensim, itertools
import numpy as np
from params import FLAGS
import zipfile
import xml.etree.ElementTree as xmlParser
from nltk.tokenize import TweetTokenizer
import os
import sys

# takes char-embedding file path and returns character list and vectors, relatively. the order of character is used, with unichar() func. char value can be taken.
def readCharEmbeddings(path, embedding_size):
    in_file = gensim.models.word2vec.LineSentence(path)

    lines = lambda: itertools.islice(in_file, None)

    model_tuple = lambda: ((line[0], [float(value) for value in line[1:]]) for line in lines())
    model_dict = dict(model_tuple())

    char_list = [int(char_number) for char_number in model_dict.keys() if
                 len(model_dict[char_number]) == embedding_size]
    vectors = [embed for embed in model_dict.values() if len(embed) == embedding_size]

    char_list.append(ord('U'))
    vectors.append(np.random.randn(embedding_size))
    char_list.append(ord('P'))
    vectors.append(np.zeros(embedding_size))

    return char_list, np.array(vectors)


##READS GLOVE EMBEDDINGS
def readGloveEmbeddings(path, embedding_size):
    DOC_LIMIT = None
    in_file = gensim.models.word2vec.LineSentence(path)

    lines = lambda: itertools.islice(in_file, DOC_LIMIT)
    model_tuple = lambda: ((line[0], [float(value) for value in line[1:]]) for line in lines())

    model_dict = dict(model_tuple())
    temp_vocab = list(model_dict.keys())
    temp_vectors = list(model_dict.values())

    vocab = list()
    vectors = list()
    # remove erratic embeddings
    count = 0

    for line in temp_vectors:
        if len(line) == embedding_size:  # problem here FLAGS instead of emb_size param!!!
            vocab.append(temp_vocab[count])
            vectors.append(temp_vectors[count])
        count += 1
    del temp_vectors, temp_vocab, model_dict

    # add special tokens
    vocab.append("UNK")
    vectors.append(np.random.randn(embedding_size))
    vocab.append("PAD")
    vectors.append(np.zeros(embedding_size))

    embeddings = np.array(vectors)
    return vocab, embeddings


# reads PAN2017 - author profiling training dataset
# one-hot vectors: female = [0,1]
#		   male   = [1,0]
# input:  string = path to the zip-file corresponding to the training data
# output: list ("training_set")  = author - tweet pairs
#	 dict ("target_values") = author(key) - ground-truth(value) pairs
#	 list ("seq-lengths")   = lenght of each tweet in the list "training_set"
def readData(path, mode):

    path = os.path.join(os.path.join(path, FLAGS.lang), "text")
    tokenizer = TweetTokenizer()
    training_set = []
    target_values = {}
    seq_lengths = []

    # for each author
    for name in os.listdir(path):

        if mode != "Test":
            # ground truth values are here
            if name.endswith(".txt"):
                file_name = os.path.join(path,name)

                if sys.version_info[0] < 3:
                    text = open(file_name, 'r')

                    # each line = each author
                    for line in text:
                        words = line.strip().split(b':::')
                        if words[1].decode() == "female":
                            target_values[words[0].decode()] = [0, 1]
                        elif words[1].decode() == "male":
                            target_values[words[0].decode()] = [1, 0]

                else:
                    text = open(file_name, 'r', encoding="utf8")

                    # each line = each author
                    for line in text:
                        words = line.strip().split(':::')
                        if words[1] == "female":
                            target_values[words[0]] = [0, 1]
                        elif words[1] == "male":
                            target_values[words[0]] = [1, 0]

        # tweets are here
        if name.endswith(".xml"):

            # get author name from file name
            base = os.path.basename(name)
            author_id = os.path.splitext(base)[0]

            # parse tweets
            xml_file_name = os.path.join(path,name)
            if sys.version_info[0] < 3:
                xmlFile = open(xml_file_name, "r")
            else:
                xmlFile = open(xml_file_name, "r", encoding="utf8") 
            rootTag = xmlParser.parse(xmlFile).getroot()

            # for each tweet
            for documents in rootTag:
                for document in documents.findall("document"):
                    words = tokenizer.tokenize(document.text)
                    training_set.append([author_id, words])  # author-tweet pairs
                    seq_lengths.append(len(words))  # length of tweets will be fed to rnn as timestep size

    if mode != "Test":
        return training_set, target_values, seq_lengths
    else:
        return training_set, None, seq_lengths


# takes list of tweet and list of characters
# returns character order of each character in each tweet
def char2id(tweets, char_list):
    batch_tweet_ids = []
    for tweet in tweets:
        tweet_ids = []
        for word in tweet:
            for char in word:
                if char != 'P':
                    char = char.lower()
                try:
                    tweet_ids.append(char_list[ord(char)])
                except:
                    tweet_ids.append(char_list[ord('U')])

        batch_tweet_ids.append(tweet_ids)
    return batch_tweet_ids


#### PREPARES TEST DATA ####
def prepTestData(tweets, user, target):
    # prepare output
    test_output = user2target(user, target)

    # prepare input by adding padding
    tweet_lengths = [len(tweet) for tweet in tweets]
    max_tweet_length = max(tweet_lengths)

    test_input = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        padded_tweet = []
        for j in range(max_tweet_length):
            if len(tweet) > j:
                padded_tweet.append(tweet[j])
            else:
                padded_tweet.append("PAD")
        test_input.append(padded_tweet)

    return test_input, test_output


def user2target(users, targets):
    target_values = []
    for user in users:
        target_values.append(targets[user])
    return target_values


# takes all tweets, users and targets
# creates a batch as a list of characters according to given iteration no, batch size and
# size of maximum tweet
def prepCharBatchData(mode, tweets, users, targets, iter_no, batch_size, max_tweet_size):
    start = iter_no * batch_size
    end = iter_no * batch_size + batch_size

    if end > len(tweets):
        end = len(tweets)

    batch_tweets = tweets[start:end]
    batch_users = users[start:end]

    # prepare output
    if mode != "Test":
        batch_output = user2target(batch_users, targets)
    else:
        batch_output = None

    batch_input = list()

    for tweet in batch_tweets:
        tweet_char_list = list()
        for word in tweet:
            tweet_char_list.extend([char for char in word.lower()])

        size = len(tweet_char_list)
        if size < max_tweet_size:
            for i in range(max_tweet_size - size):
                tweet_char_list.append('P')

        batch_input.append(tweet_char_list)

    return batch_input, batch_output


# changes tokenized words to their corresponding ids in vocabulary
def word2id(tweets, vocab):
    batch_tweet_ids = []
    for tweet in tweets:
        tweet_ids = []
        for word in tweet:
            if word != "PAD":
                word = word.lower()

            try:
                tweet_ids.append(vocab[word])
            except:
                tweet_ids.append(vocab["UNK"])

        batch_tweet_ids.append(tweet_ids)

    return batch_tweet_ids


# prepares batch data - also adds padding to tweets
# input: tweets(list) - list of tweets corresponding to the authors in:
#	users(list) - owner of the abve tweets
#	targets(dict) - ground-truth gender vector of each owner
#	seq_len(list) - sequence length for tweets
#	iter_no(int) - current # of iteration we are on
# returns: batch_input - ids of each words to be used in tf_embedding_lookup
# 	  batch_output - target values to be fed to the rnn
#	  batch_sequencelen - number of words in each tweet(gives us the # of time unrolls)
def prepWordBatchData(mode, tweets, users, targets, seq_len, iter_no):
    start = iter_no * FLAGS.batch_size
    end = iter_no * FLAGS.batch_size + FLAGS.batch_size
    if end > len(tweets):
        end = len(tweets)

    batch_tweets = tweets[start:end]
    batch_users = users[start:end]
    batch_sequencelen = seq_len[start:end]

    # prepare output
    if mode != "Test":
        batch_output_temp = user2target(batch_users, targets)
        batch_output = batch_output_temp[0]

    # prepare input by adding padding
    tweet_lengths = [len(tweet) for tweet in batch_tweets]
    max_tweet_length = max(tweet_lengths)

    batch_input = []
    for i in range(FLAGS.batch_size):
        tweet = batch_tweets[i]
        padded_tweet = []
        for j in range(max_tweet_length):
            if len(tweet) > j:
                padded_tweet.append(tweet[j])
            else:
                padded_tweet.append("PAD")
        batch_input.append(padded_tweet)

    if mode != "Test":
        return batch_input, np.asarray(batch_output).reshape(1, 2), batch_sequencelen
    else:
        return batch_input, None, batch_sequencelen


#############################################################################################################
def prepTestData(tweets, user, target, max_tweet_size):
    test_output = user2target(user, target)

    test_input = list()

    for tweet in tweets:
        tweet_char_list = list()
        for word in tweet:
            tweet_char_list.extend([char for char in word.lower()])

        size = len(tweet_char_list)
        if size < max_tweet_size:
            for i in range(max_tweet_size - size):
                tweet_char_list.append('P')

        test_input.append(tweet_char_list)

    return test_input, test_output
