import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fliter_size, hidden_dim, output_dim,
                 dropout, pad_idx, senti_size=0, senti_dim=26, passes=2, add_senti=False,spad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fliter_size = fliter_size
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True
                           )
        self.add_senti = add_senti
        if self.add_senti:
            self.passes = passes
            self.sentiembedding = nn.Embedding(senti_size, senti_dim, padding_idx=spad_idx)
            self.senti_feature_embedding = nn.Embedding(senti_dim, 100)
            # W makes the output of the model to a fixed length tensor
            self.fc = nn.Linear(hidden_dim*2+100, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.embedding_dropout = nn.Dropout(0.4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]
        bs = text.size(0)  # batch size

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = self.embedding_dropout(embedded)

        chunks, length = split_sentence(embedded, self.fliter_size, text_lengths)
        # print(len(chunks))
        # print(chunks)
        # print(length.shape)
        conved = []
        # print("length:",length)
        for i in range(len(chunks)):
            sub_embedded = chunks[i]
            # print("subembedded",sub_embedded.shape)
            # print(length[i])
            # print(length[i].shape)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(sub_embedded, length[i], batch_first=True)
            output, (hidden, cell) = self.rnn(packed_embedded)
            # output, (hidden, cell) = self.rnn(sub_embedded)
            # print("before_hidden",hidden.shape)
            # hidden = hidden[-1, :, :]
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            # hidden = torch.sum(hidden, dim=0)
            # # print(hidden.shape)
            hidden = hidden.unsqueeze(2)
            conved.append(hidden)
        conved = torch.cat(conved, dim=2)
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        # pooled = max_pool(conved)
        if self.add_senti:
            vs = self.sentiembedding(text)  # vs = [batch size,sentence length,embd_size]
            vp = torch.mean(vs, dim=1)  # vp = [batch size,embd_size] initialize vp
            for k in range(self.passes + 1):
                s = torch.bmm(vs, vp.unsqueeze(2)).squeeze()  # s = [batch size, sentence length]
                s = s / s.norm(dim=-1).view(bs, 1)  # s = [batch size, sentence length]
                s = F.softmax(s, -1).unsqueeze(1)  # s = [batch size,1,sentence length]
                vo = torch.bmm(s, vs).squeeze(1)  # s = [batch size, embd_size]
                vp = vo + vp  # vp = [batch size, embd_size] = [batch size, 26]
            W = self.senti_feature_embedding.weight  # W = [26, 100]
            out = torch.bmm(vp.unsqueeze(1),
                            W.unsqueeze(0).repeat(bs, 1, 1)).squeeze()  # out=[batch size, feature_size]
            # out = self.fc0(out)   # out = [batchsize,1]

        cat = pooled
        if self.add_senti:
            cat = torch.cat([cat, out], dim=1)

        # cat = [batch size, hidden_dim+?100]
        # print(cat.shape)
        return self.fc(cat)


def split_sentence(sentence, fliter_size, lengths):
    # sentence = [bs, sentence_length, embedded_dim)  length = [bs] list
    chunks = []
    length = []
    # sentence.requires_grad_(False)
    sentence_length = sentence.shape[1]
    bs = sentence.shape[0]
    ed = sentence.shape[2]
    # print(sentence.shape)
    # print(lengths.shape)
    if sentence_length < fliter_size:
        sentence = torch.cat((sentence, torch.zeros(bs, fliter_size-sentence_length, ed).cuda()), dim=1)
        chunks.append(sentence)
        length = lengths.unsqueeze(0)
        # length.requires_grad_(False)
        # print("min",length)
        # print("min",length.shape)
    else:
        for i in range(sentence_length-fliter_size+1):
            # chunk = [bs, fliter_size, embedded_dim]
            chunk = sentence[:, i:i+fliter_size, :]
            # chunk.requires_grad_(False)
            chunks.append(chunk)
        for i in range(lengths.shape[0]):
            value = lengths[i]
            tensor = torch.Tensor(sentence_length-fliter_size+1)
            for j in range(sentence_length-fliter_size+1):
                if j < value-fliter_size+1:
                    tensor[j] = fliter_size
                else:
                    tensor[j] = value - j
                if tensor[j] < 1:
                    tensor[j] = 1
            tensor = tensor.unsqueeze(0)
            # tensor.requires_grad_(False)
            length.append(tensor)
        length = torch.cat(length, dim=0)
        length = length.permute(1, 0)
        # print("sentence_split",length)
        # print("chunk",len(chunks))
    return chunks, length


def max_pool(vector_list):
    # vector_list = list(elements) element = [bs, hidden_dim]
    # print(vector_list)
    length = len(vector_list)
    bs = vector_list[0].shape[0]
    # print(bs)
    normal_list = []
    for i in range(length):
        # normal_list = list(elements) element=[bs]
        normal_list.append(torch.norm(vector_list[i], dim=1))
        indexs = []
    for i in range(bs):
        index = -1
        value = float("-inf")
        for j in range(length):
            if float(normal_list[j][i])>value:
                index = j
                value = float(normal_list[j][i])
        indexs.append(index)

    pooled_list = []
    for i in range(bs):
        index = indexs[i]
        tensor = vector_list[index][i]
        pooled_list.append(tensor.unsqueeze(0))
    pooled_out = torch.cat(pooled_list, dim=0)
    return pooled_out

