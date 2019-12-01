import torch
import torchtext.data as data
from torchtext import datasets
import torchtext.vocab as vocab
import train
import CNN
import RCNN
import random
import argparse
import spacy
import time
from nltk.tokenize import TweetTokenizer


parser = argparse.ArgumentParser(description='CNN for sentiment analysis with rnn filter')
parser.add_argument('-load-model', type=bool, default=False, help='whether load the pretrained model')
args = parser.parse_args()
TYPE = 1  # the dataset type {0:IMDB, 1:SST}
ADD_SENTI = False  # whether to add sentiment embeddings
WVINIT = True   # whether to use local word vector or pretrained word vector
FINETURN = False  # whether to adjust word vector and sentiment vector
MODEL = True   # load the model or train from scratch
# spacy_en = spacy.load("en")
#
#
# def tokenizer(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]
token = TweetTokenizer(strip_handles=True, reduce_len=True)
tokenizer = token.tokenize
SEED = 1234
torch.manual_seed(SEED)
TEXT = data.Field(tokenize=tokenizer, lower=True, include_lengths=True)
STEXT = data.Field(tokenize=tokenizer, lower=True)  #build senti_vector
LABEL = data.LabelField(dtype=torch.float)
if TYPE == 0:
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    s, _ = datasets.IMDB.splits(STEXT, LABEL)
    s, _ = s.split(random_state=random.seed(SEED))
else:
    paths = '.data/sst'
    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=paths,
        train='train.json',
        validation='validation.json',
        test='test.json',
        format='json',
        fields=fields
    )
    s, _ = data.TabularDataset.splits(
        path=paths,
        train='train.json',
        test='test.json',
        format='json',
        fields={'text': ('text', STEXT), 'label': ('label', LABEL)}
    )
MAX_VOCAB_SIZE = 25000
senti_embeddings = vocab.Vectors(name='newSentiVectors.txt',
                                  unk_init=torch.Tensor.zero_)

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.840B.300d",
                 unk_init=torch.Tensor.normal_)

STEXT.build_vocab(s,
                 max_size=MAX_VOCAB_SIZE,
                 vectors=senti_embeddings,
                 unk_init=torch.Tensor.zero_)

LABEL.build_vocab(train_data)
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    batch_size=BATCH_SIZE,
    )
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
if ADD_SENTI:
    SENTIMENT_SIZE = len(STEXT.vocab)
    # print(len(STEXT.vocab))
    # print(len(TEXT.vocab))
    # print(STEXT.vocab.stoi)
    # print("-----------------\n")
    # print(TEXT.vocab.stoi)
    SPAD_IDX = STEXT.vocab.stoi[STEXT.pad_token]
    SENTI_DIM = 26
    # model = CNN.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX,
    #                 senti_size=SENTIMENT_SIZE, senti_dim=SENTI_DIM, passes=2, add_senti=True, spad_idx=SPAD_IDX)
    model = RCNN.CNN(INPUT_DIM, EMBEDDING_DIM, 8, 300, OUTPUT_DIM, DROPOUT, PAD_IDX,
                     senti_size=SENTIMENT_SIZE, senti_dim=SENTI_DIM, passes=2, add_senti=True, spad_idx=SPAD_IDX)
    SUNK_IDX = STEXT.vocab.stoi[STEXT.unk_token]
    model.sentiembedding.weight.data.copy_(STEXT.vocab.vectors)
    model.sentiembedding.weight.data[SUNK_IDX] = torch.zeros(SENTI_DIM)
    model.sentiembedding.weight.data[SPAD_IDX] = torch.zeros(SENTI_DIM)
else:
    # model = CNN.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model = RCNN.CNN(INPUT_DIM, EMBEDDING_DIM, 5, 300, OUTPUT_DIM, DROPOUT, PAD_IDX)
if WVINIT:
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
if MODEL:
    model.load_state_dict(torch.load('sentiment-model.pt'))
# test_loss, test_acc, test_p, test_r, test_f1 = train.train(model, test_iterator)
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
N_EPOCHS = 30
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    if not FINETURN:
        model.embedding.weight.requires_grad_(False)
        if ADD_SENTI:
            model.sentiembedding.weight.requires_grad_(False)

    start_time = time.time()

    train_loss, train_acc, train_p, train_r, train_f1 = train.train(model, train_iterator)
    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = train.train(model, valid_iterator)
    test_loss, test_acc, test_p, test_r, test_f1 = train.predict(model, test_iterator)
    end_time = time.time()

    epoch_mins, epoch_secs = train.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'sentiment-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f'\t Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Precision: {test_p:.3f} | '
          f'Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f}')

