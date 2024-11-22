import torch
from torch import nn

import torch.nn.functional as F
import math
import copy

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# 对每个签到进行嵌入
# Transformer 直接用的 nn.Embedding，此处作者自己写了针对 POI 的 embedding
class TransRecEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class POI_UserEmbedding(nn.Module):  # p u t poi感知
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        poi_num = vocab_size["POI"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        return torch.cat((poi_emb, user_emb, hour_emb), 1)


class CAT_UserEmbedding(nn.Module):  # c u t 类别感知
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]

        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)

    def forward(self, x):
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        return torch.cat((cat_emb, user_emb, hour_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split the embedding into self.heads different pieces
        # Multi head
        # [len, embed_size] --> [len, heads, head_dim]
        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        # 爱因斯坦求和约定，矩阵计算的简单表示方式
        energy = torch.einsum("qhd,khd->hqk", [queries, keys])  # [heads, query_len, key_len]

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # [heads, query_len, key_len]

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )  # [query_len, key_len]

        out = self.fc_out(out)  # [query_len, key_len]

        return out


# 对应 TransformerBlock
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# 对应 Encoder
class TransRecEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,  # TransRecEmbedding
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransRecEncoder, self).__init__()

        # Transformer 直接用的 nn.Embedding，此处用的是作者自己写的针对 POI 的 TransRecEmbedding
        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        # num_encoder_layers 个 EncoderBlock
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        # 因为是在 Encoder 中，所以 value, key, query 都一样
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# 对应 UserEncoder
class UserEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,  # UserEmbedding
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(UserEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        # num_encoder_layers 个 EncoderBlock
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        # 因为是在 Encoder 中，所以 value, key, query 都一样
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# 因为 query 和 key 的维度不一样大，所以专门写了这个 Attention
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # 将 q 的维度调整为和 k 一样大
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        weight = torch.softmax(torch.inner(q, key), dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        out = torch.sum(torch.mul(value, weight), 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


class TransRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=2,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.1,
            random_mask=True,
            mask_prop=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.random_mask = random_mask
        self.mask_prop = mask_prop

        # LAYERS
        self.embedding = TransRecEmbedding(
            f_embed_size,
            vocab_size
        )

        # user_embedding 单独计算user_embedding
        if settings.enable_user_embedding:
            self.user_embed_size = f_embed_size * 3

            # poi感知
            self.POI_userEmbedding = POI_UserEmbedding(
                f_embed_size,
                vocab_size
            )
            self.POI_user_Encoder = UserEncoder(
                self.POI_userEmbedding,
                self.user_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )

            # 类别感知
            self.CAT_userEmbedding = CAT_UserEmbedding(
                f_embed_size,
                vocab_size
            )
            self.CAT_user_Encoder = UserEncoder(
                self.CAT_userEmbedding,
                self.user_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )

            self.user_embed_dim_squeeze = nn.Linear(self.user_embed_size, f_embed_size)

            self.user_embed_size = f_embed_size * 3;

        if settings.LS_strategy == 'TransLSTM':
            self.long_lstm = nn.LSTM(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
                num_layers=num_lstm_layers,
                dropout=0
            )
            self.short_lstm = nn.LSTM(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
                num_layers=num_lstm_layers,
                dropout=0
            )

        elif settings.LS_strategy == 'DoubleTrans':
            self.short_encoder = TransRecEncoder(
                self.embedding,
                self.total_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )
            self.long_encoder = TransRecEncoder(
                self.embedding,
                self.total_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )
        else:
            pass

        if settings.enable_user_embedding:
            self.final_attention = Attention(
                qdim=f_embed_size * 2,
                kdim=self.total_embed_size
            )
        else:
            self.final_attention = Attention(
                qdim=f_embed_size,
                kdim=self.total_embed_size
            )

        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()

        if settings.enable_alpha:
            self.alpha_gru = nn.GRU(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
            )

            if settings.enable_user_embedding:
                self.alpha_input_embed_size = 3 * self.total_embed_size + 2 * f_embed_size
            else:
                self.alpha_input_embed_size = 3 * self.total_embed_size + 1 * f_embed_size
            self.alpha_linear = nn.Sequential(
                nn.Linear(self.alpha_input_embed_size, self.total_embed_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(self.total_embed_size, f_embed_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(f_embed_size, 1),
                nn.Sigmoid())
            self.alpha = 0.5

        # 使用一个 dict 存储需要追踪的参数
        self.track_parameters = {}

    def cal_CL_loss(self, short, long, short_proxy, long_proxy):
        def euclidean_distance(x, y):
            return torch.square(x - y)

        if settings.CL_strategy == 'BPR':
            long_mean_recent_loss = torch.sum(F.softplus(torch.sum(long * (-long_proxy + short_proxy), dim=-1)))
            short_recent_mean_loss = torch.sum(F.softplus(torch.sum(short * (-short_proxy + long_proxy), dim=-1)))
            mean_long_short_loss = torch.sum(F.softplus(torch.sum(long_proxy * (-long + short), dim=-1)))
            recent_short_long_loss = torch.sum(F.softplus(torch.sum(short_proxy * (-short + long), dim=-1)))
            return long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        elif settings.CL_strategy == 'Triplet':
            triplet_loss = (
                nn.TripletMarginWithDistanceLoss(distance_function=euclidean_distance, margin=1.0, reduction='sum'))
            long_loss_1 = triplet_loss(long, long_proxy, short_proxy)
            long_loss_2 = triplet_loss(long_proxy, long, short)
            short_loss_1 = triplet_loss(short, short_proxy, long_proxy)
            short_loss_2 = triplet_loss(short_proxy, short, long)
            return (long_loss_1 + long_loss_2 + short_loss_1 + short_loss_2) * 0.01
        elif settings.CL_strategy == 'NativeNCE':
            pos1 = torch.mean(torch.mul(long, long_proxy))
            pos2 = torch.mean(torch.mul(short, short_proxy))
            neg1 = torch.mean(torch.mul(long, short_proxy))
            neg2 = torch.mean(torch.mul(short, long_proxy))
            pos = (pos1 + pos2) / 2
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss
        elif settings.CL_strategy == 'CosineNCE':
            pos1 = F.cosine_similarity(long, long_proxy, dim=0)
            pos2 = F.cosine_similarity(short, short_proxy, dim=0)
            neg1 = F.cosine_similarity(long, short_proxy, dim=0)
            neg2 = F.cosine_similarity(short, long_proxy, dim=0)
            pos = (pos1 + pos2) / 2
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss
        else:
            raise NotImplementedError

    def forward(self, sample):
        # process input sample
        # [(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq])],[(seq2)],...]
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]

        if settings.enable_drop:
            short_term_features = short_term_sequence[:, :- 1 - settings.drop_steps]
            target = short_term_sequence[0, -1 - settings.drop_steps]
        else:
            short_term_features = short_term_sequence[:, :- 1]
            target = short_term_sequence[0, -1]

        user_id = short_term_sequence[2, 0]

        # region long-term
        if settings.LS_strategy == 'TransLSTM':
            long_term_out = []  # [6*10, 8*10, ...]
            for seq in long_term_sequences:
                long_embedding = self.embedding(seq)  # 嵌入层
                output, _ = self.long_lstm(torch.unsqueeze(long_embedding, 0))
                long_output = torch.squeeze(output)
                long_term_out.append(long_output)  # [seq_num, len, emb_size]
            long_term_catted = torch.cat(long_term_out, dim=0)
        else:
            long_term_out = []  # [6*10, 8*10, ...]
            for seq in long_term_sequences:
                output = self.long_encoder(feature_seq=seq)
                long_term_out.append(output)  # [seq_num, len, emb_size]
            long_term_catted = torch.cat(long_term_out, dim=0)
        # endregion

        # region short-term
        short_embedding = self.embedding(short_term_features)
        if settings.LS_strategy == 'TransLSTM':
            output, _ = self.short_lstm(torch.unsqueeze(short_embedding, 0))
            short_term_state = torch.squeeze(output)
        elif settings.LS_strategy == 'DoubleTrans':
            short_term_state = self.short_encoder(feature_seq=short_term_features)
        else:
            raise NotImplementedError
        # endregion

        # region user_embedding
        if settings.enable_user_embedding:
            # POI感知
            user_seqs_long = long_term_sequences
            his_out_POI = []
            for seq in user_seqs_long:
                his_output_long = self.POI_user_Encoder(feature_seq=seq)
                his_out_POI.append(his_output_long)
            his_output_short = self.POI_user_Encoder(feature_seq=short_term_features)
            his_out_POI.append(his_output_short)
            his_out_cat = torch.cat(his_out_POI, dim=0)
            user_embed_POI = torch.mean(his_out_cat, dim=0)  # 获取到poi感知
            user_embed_POI = self.user_embed_dim_squeeze(user_embed_POI)

            # 类别感知  改成LSTM方式
            his_out_CAT = []
            for seq in user_seqs_long:
                cat_his_output_long = self.CAT_user_Encoder(feature_seq=seq)
                his_out_CAT.append(cat_his_output_long)
            cat_his_output_short = self.CAT_user_Encoder(feature_seq=short_term_features)
            his_out_CAT.append(cat_his_output_short)
            his_out_cat = torch.cat(his_out_CAT, dim=0)
            user_embed_CAT = torch.mean(his_out_cat, dim=0)  # 获取到cat感知
            user_embed_CAT = self.user_embed_dim_squeeze(user_embed_CAT)
            user_embed = torch.cat([user_embed_POI, user_embed_CAT], dim=-1)
        # else:
        # user_embed = self.embedding.user_embed(user_id)  # 用户嵌入 （60大小）一维[60]
        # endregion
        if settings.enable_user_embedding:
            long_term_prefer = self.final_attention(user_embed, long_term_catted,
                                                    long_term_catted)  # 一维tensor[300] 偏好序列进行注意力机制融合
            short_term_prefer = self.final_attention(user_embed, short_term_state, short_term_state)  # 一维tensor[300]
        else:
            long_term_prefer = long_term_catted  # 理论上应该保持一维[300]
            short_term_prefer = short_term_state  # 理论上应该保持一维[300]

        if settings.enable_CL:
            # proxy for long-term
            long_term_embeddings = []
            for seq in long_term_sequences:
                seq_embedding = self.embedding(seq)
                long_term_embeddings.append(seq_embedding)
            long_term_embeddings = torch.cat(long_term_embeddings, dim=0)
            long_term_proxy = torch.mean(long_term_embeddings, dim=0)
            # proxy for short-term
            short_term_proxy = torch.mean(short_embedding, dim=0)

            CL_loss = self.cal_CL_loss(short_term_prefer, long_term_prefer,
                                       short_term_proxy, long_term_proxy)

        # final output
        if settings.enable_alpha:
            # region fusion long and short like CLSR
            all_term_features = torch.cat([torch.cat(long_term_sequences, dim=-1), short_term_features],
                                          dim=-1)  # 二维tensor[5=特征个数,len=序列长度]
            all_term_embedding = self.embedding(all_term_features)  # 二维[len=序列长度，300=总嵌入大小]
            _, h_n = self.alpha_gru(all_term_embedding)  # 二维tensor[1,300]
            h_n = torch.squeeze(h_n)  # 维度压缩，一维tensor[300]
            last_time = short_term_features[3, -1]  # 当前时间
            last_time_embedding = self.embedding.hour_embed(last_time)  # 一维tensor[60]
            if settings.enable_user_embedding:
                concat_all = torch.cat([h_n, long_term_prefer, short_term_prefer, user_embed, last_time_embedding],
                                       dim=-1)  # 一维tensor[1020=3*300+2*60]
            else:
                concat_all = torch.cat([h_n, long_term_prefer, short_term_prefer, last_time_embedding],
                                       dim=-1)  # 一维tensor[1020=3*300+1*60]
            self.alpha = self.alpha_linear(concat_all)  # 一维tensor[长期偏好融合值]
            self.track_parameters['alpha'] = self.alpha.item()
            # alpha 放到 long
            final_att = long_term_prefer * self.alpha + short_term_prefer * (1 - self.alpha)  # 一维tensor[300]
            output = self.out_linear(final_att)  # 一维tensor[POI个数]
            # endregion
        else:
            if settings.enable_fix_alpha:  # 以固定权重值的方式融合长短期
                final_att = long_term_prefer * settings.fix_alpha + short_term_prefer * (1 - settings.fix_alpha)
                output = self.out_linear(final_att)  # 一维tensor[POI个数]

            elif settings.enable_filatt:  # 以注意力的形式融合长短期偏好
                h = torch.cat((long_term_catted, short_term_state))  # concat long and short
                f_user_embed = self.embedding.user_embed(user_id)
                final_att = self.final_attention(f_user_embed, h, h)
                output = self.out_linear(final_att)

            else:  # 不区分偏好重要度，直接拼接
                final_att = long_term_prefer + short_term_prefer
                output = self.out_linear(final_att)  # 一维tensor[POI个数]

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)
        poi_loss = self.loss_func(pred, label)
        if settings.enable_CL:
            loss = poi_loss + settings.CL_weight * CL_loss
            self.track_parameters['poi_loss'] = poi_loss.item()
            self.track_parameters['CL_weight'] = settings.CL_weight
            self.track_parameters['CL_loss'] = CL_loss.item()
        else:
            loss = poi_loss

        return loss, output

    def predict(self, sample):
        test_loss, pred_raw = self.forward(sample)
        ranking = torch.sort(pred_raw, descending=True)[1]

        if settings.enable_drop:
            target = sample[-1][0, -1 - settings.drop_steps]
        else:
            target = sample[-1][0, -1]

        return ranking, target, test_loss

    def print_parameters(self, epoch):
        print(f'\n{settings.output_file_name} parameters epoch {epoch}:')
        # 将 self.track_parameters 中的所有参数打印到一行
        for key, value in self.track_parameters.items():
            print(f'{key}: {value}', end='\t')
        print('')


if __name__ == "__main__":
    pass
