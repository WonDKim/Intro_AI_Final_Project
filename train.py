import torch.optim as optim
import torch.nn as nn
import torch
import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def get_four_params(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    tp0 = ((rounded_preds == 1) & (y == 1)).float().cpu().sum()
    tn0 = ((rounded_preds == 0) & (y == 0)).float().cpu().sum()
    fn0 = ((rounded_preds == 0) & (y == 1)).float().cpu().sum()
    fp0 = ((rounded_preds == 1) & (y == 0)).float().cpu().sum()
    return tp0, tn0, fn0, fp0


def train(model, iterator):
    epoch_loss = 0
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    # optimizer = adabound.AdaBound(model.parameters(), lr=0.0006)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    tp, tn, fn, fp = 0.0, 0.0, 0.0, 0.0
    for batch in iterator:
        # start = time.time()
        optimizer.zero_grad()
        text, text_lengths = batch.text
        # print(text.shape)
        # print(text.permute(1, 0))
        # print(text_lengths)
        # print(text.shape[1])
        # print(type(text_lengths))
        text = text.cuda()
        text_lengths = text_lengths.cuda()
        batch.label = batch.label.cuda()
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        # acc = binary_accuracy(predictions, batch.label)
        tp0, tn0, fn0, fp0 = get_four_params(predictions, batch.label)
        tp = tp + tp0
        tn = tn + tn0
        fn = fn + fn0
        fp = fp + fp0
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        # epoch_acc += acc.item()
    p = tp/(tp + fp)
    r = tp/(tp + fn)
    f1 = 2*r*p/(r+p)
    epoch_acc = (tp + tn)/(tp+tn+fp+fn)
    return epoch_loss / len(iterator), epoch_acc.item(), p.item(), r.item(), f1.item()


# def evaluate(model, iterator):
#     epoch_loss = 0
#     epoch_acc = 0
#     optimizer = optim.Adam(model.parameters(), lr=0.0004)
#     # optimizer = adabound.AdaBound(model.parameters(), lr=0.0006)
#     criterion = nn.BCEWithLogitsLoss()
#
#     model.train()
#
#     for batch in iterator:
#         optimizer.zero_grad()
#         text, text_lengths = batch.text
#         predictions = model(text, text_lengths.squeeze(1)
#         loss = criterion(predictions, batch.label)
#         acc = binary_accuracy(predictions, batch.label)
#
#         loss.backward()
#
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, iterator):
    epoch_loss = 0
    epoch_acc = 0
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    tp, tn, fn, fp = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            text = text.cuda()
            text_lengths = text_lengths.cuda()
            batch.label = batch.label.cuda()
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            tp0, tn0, fn0, fp0 = get_four_params(predictions, batch.label)
            tp = tp + tp0
            tn = tn + tn0
            fn = fn + fn0
            fp = fp + fp0

            epoch_loss += loss.item()
            # epoch_acc += acc.item()
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * r * p / (r + p)
    epoch_acc = (tp + tn) / (tp + tn + fp + fn)
    return epoch_loss / len(iterator), epoch_acc.item(), p.item(), r.item(), f1.item()
