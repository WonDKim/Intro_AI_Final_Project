import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx, senti_size=0, senti_dim=26, passes=2, add_senti=False,spad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim),
                      padding=(fs-1, 0)
                      )
            for fs in filter_sizes
        ])
        self.add_senti = add_senti
        if self.add_senti:
            self.passes = passes
            self.sentiembedding = nn.Embedding(senti_size, senti_dim, padding_idx=spad_idx)
            self.senti_feature_embedding = nn.Embedding(senti_dim, 100)
            # W makes the output of the model to a fixed length tensor
            self.fc0 = nn.Linear(100, 1)  # sentiment fully connectec layer
            self.fc = nn.Linear(len(filter_sizes) * n_filters+100, output_dim)
        else:
            self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]  text_lengths = [batchsize] list
        text = text.permute(1, 0)

        # text = [batch size, sent len]
        bs = text.size(0)  # batch size

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        # print("embedded",embedded,embedded.size())
        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]
        if self.add_senti:
            # sentiembedded = self.sentiembedding(text)
            # vp = torch.mean(sentiembedded, dim=1)  # initialize vector vp vp=[batch size,emb dim]
            vs = self.sentiembedding(text)  # vs = [batch size,sentence length,embd_size]
            # print("vs", vs, vs.size())
            # print("text", text)
            # print("vs", vs)
            vp = torch.sum(vs, dim=1)  # vp = [batch size,embd_size] initialize vp
            vp = vp/text_lengths.unsqueeze(1).float()
            # vp = torch.mean(vs, dim=1)
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

        cat = self.dropout(torch.cat(pooled, dim=1))
        if self.add_senti:
            cat = self.dropout(torch.cat([cat, out], dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


