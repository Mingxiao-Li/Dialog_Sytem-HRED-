import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ContextRNN,HREDDecoderRNN,EncoderRNN
from modules import l2_pooling


class hred(nn.Module):

    def __init__(self,hparams, n_words, itfloss_weights, fix_embedding = False):
        super(hred,self).__init__()
        self.training = True
        self.hparams = hparams
        self.n_words = n_words
        self.embedding = nn.Embedding(
            num_embeddings = n_words,
            embedding_dim = hparams['hidden_size'],
            padding_idx = hparams['PAD_id']
        )
        self.embedding.weight.requires_grad = not fix_embedding

        self.encoder = EncoderRNN(
            hidden_size = hparams['hidden_size'],
            embedding = self.embedding,
            num_layers = hparams['num_layers'],
            droput = hparams['dropout'],
            bidirectional= True,
        )

        self.context = ContextRNN(
            hidden_size = hparams['hidden_size'] * 2,
            num_layers = hparams['num_layers'],
            dropout = hparams['dropout']
        )

        self.decoder = HREDDecoderRNN(
            embedding = self.embedding,
            hidden_size = hparams['hidden_size'],
            context_hidden_size = hparams['hidden_size']*2,
            output_size = n_words,
            num_layers = hparams['num_layers'],
            dropout = hparams['dropout']
        )

        self.criterion = nn.NLLLoss(
            weight = torch.tensor(itfloss_weights).cuda() if itfloss_weights else None,
            ignore_index = hparams['PAD_id']
        )

    def forward(self, data, train=True):

        src = data['src']
        src_len = data['src_len']
        #src = [[[dials],[[uttr],[uttr]],...]
        #src_len = [[lengths]]
        batch_size = src.size(0)
        MAX_DIAL_LEN = src.size(1)
        MAX_UTTR_LEN = src.size(2)
        hidden_size = self.hparams['hidden_size']
        num_layers = self.hparams['num_layers']

        src = src.view(batch_size * MAX_DIAL_LEN, -1)
        #src = [[uttr],[uttr],[uttr],...]
        src_len = src_len.flatten()
        #src_len = [lengths]

        if len(src_len) > 1:
            src_len, perm_index = src_len.sort(0, descending = True)
            #perm_index permutation of index

            #back_index = get original idex
            back_index = [(perm_index == i).nonzero().flatten().item() for i in range(perm_index.size(0))]
            src = src[perm_index]

        src = src.cuda()
        src_len = src_len.cuda()

        encoder_output, encoder_hidden = self.encoder(src,src_len)
        #shape encoder_output =[batch*MAX_DIA_LEN,seq_len,num_dir * hidden_size]
        #shape encoder_hidden = [num_layer*num_dir,batch*MAX_DIA_LEN,hidden_size]
        encoder_hidden = encoder_hidden.transpose(1,0).contiguous().view(batch_size * MAX_DIAL_LEN, num_layers, -1)


        if len(src_len) > 1:
            encoder_output = encoder_output[back_index]
            encoder_hidden = encoder_hidden[back_index]
            src_len = src_len[back_index]

        if self.hparams["l2_pooling"]:
            # Separate forward and backward hiddens
            encoder_output = encoder_output.view(batch_size * MAX_DIAL_LEN, MAX_UTTR_LEN,2,-1)
            forward = l2_pooling(encoder_output[:,:,0],src_len)
            backward = l2_pooling(encoder_output[:,:,1],src_len)
            encoder_hidden = torch.cat((forward,backward),dim=1).view(batch_size,MAX_DIAL_LEN,-1)
            #encoder_hidden shape =[batch_size, MAX_DIAL_LEN, hiddensize*2]

        else:
            encoder_hidden = encoder_hidden[:,-1].view(batch_size,MAX_DIAL_LEN,-1)

        context_output, _ = self.context(encoder_hidden)
        # batch, MAX_DIAL_LEN, hidden_size -> batch * uttr, hidden_size


        context_output = context_output[:,-1]
        #take the output of the last GRU unit shape= (batch, hidden_size (512))


        decoder_hidden = (context_output[:,:hidden_size]+context_output[:,hidden_size:])
        decoder_hidden = decoder_hidden.expand(num_layers, batch_size, hidden_size).contiguous()

        if train:
            return self.compute_loss(decoder_hidden, context_output, data["tgt"])
        else:
            return self.beam_search(decoder_hidden[:,-1].unsqueeze(1).contiguous(),context_output[-1].unsqueeze(0))

    def compute_loss(self, initial_hidden, context_hidden, tgt):
        PAD_id = self.hparams["PAD_id"]
        tgt = tgt.cuda()
        batch_size = tgt.size(0)
        MAX_TGT_LEN = tgt.size(1)
        teacher_forcing_ratio = self.hparams["teacher_forcing_ratio"]
        loss_name = self.hparams["loss"]
        mmi_lambda = self.hparams["mmi_lambda"]
        mmi_gamma = self.hparams["mmi_gamma"]

        loss = 0
        print_losses = []
        n_totals = 0

        decoder_input = torch.ones(batch_size, 1).type(torch.cuda.LongTensor)
        decoder_hidden = initial_hidden
        ut_decoder_hidden = torch.zeros(decoder_hidden.size()).cuda()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for t in range(MAX_TGT_LEN):
            #shape decoder_input = (batch_size,1)
            #decoder_hidden = (num_layers, batch_size, hidden_size)
            #context_hidden = (batch, hidden_size (512))
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, context_hidden
            )
            if loss_name == "mmi":
                ut_decoder_output, ut_decoder_hidden = self.decoder(
                    decoder_input, ut_decoder_hidden, context_hidden
                )
            mask_loss = self.criterion(F.log_softmax(decoder_output,dim=1),tgt[:,t])

            if loss_name == "mmi" and t+1 <= mmi_gamma:
                mask_loss -= mmi_lambda *  self.criterion(F.log_softmax(ut_decoder_output,dim=1),tgt[:,t])
            n_total = (tgt[:,t] != PAD_id).sum().item()
            loss += mask_loss
            print_losses.append(mask_loss.item()*n_total)
            n_totals += n_total

            if use_teacher_forcing:
                decoder_input = tgt[:,t].view(-1,1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)]).cuda()

        return loss, (sum(print_losses)/n_totals)

    def beam_search(self, initial_hidden, context_hidden):
        n_words = self.n_words
        EOS_id = self.hparams["EOS_id"]
        batch_size = context_hidden.size(0)
        hidden_szie = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]
        beam_width = self.hparams["beam_width"]
        len_alpha = self.hparams["len_alpha"]
        suppress_lmd = self.hparams["suppress_lambda"]
        MAX_UTTR_LEN = self.hparams["MAX_UTTR_LEN"]

        decoder_hidden = initial_hidden

        decoder_input = torch.ones(batch_size,1).type(torch.cuda.LongTensor)

        decoder_output, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, context_hidden
        )
        topv, topi = F.log_softmax(decoder_output, dim=1).topk(beam_width)

        topv = topv.flatten()
        decoder_input = topi.t()
        decoder_hidden = decoder_hidden.expand(num_layers, beam_width, hidden_szie).contiguous()
        inf_uttrs = [[id.item()] for id in decoder_input]
        #inf_uttrs top #beam_width words  [[1],...[10]]
        repet_counts = torch.ones(beam_width, decoder_output.size(1)).type(torch.cuda.FloatTensor)

        decoder_output = decoder_output.expand(beam_width, n_words)
        #  each -beam --> n_words probability
        for _ in range(MAX_UTTR_LEN-1):
            for b in range(beam_width):
                repet_counts[b, inf_uttrs[b][-1]] += 1
            eos_idx = [idx for idx,words in enumerate(inf_uttrs) if words[-1] == EOS_id]

            prev_output, prev_hidden = decoder_output, decoder_hidden

            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,context_hidden)
            #decoder_input: top #beam_width words obtained in previous time step

            suppressor = torch.ones(repet_counts.size()).cuda() / repet_counts.pow(suppress_lmd)

            decoder_output = topv.unsqueeze(1) + F.log_softmax(decoder_output *  suppressor, dim = 1)

            if len(eos_idx) > 0:
                decoder_output[eos_idx] = float("-inf")
                decoder_output[eos_idx, EOS_id] = prev_output[eos_idx, EOS_id]
                decoder_hidden[:,eos_idx] = prev_hidden[:, eos_idx]

            lp = torch.tensor([(5+len(inf_uttr)+1)**len_alpha / (5+1)**len_alpha for inf_uttr in inf_uttrs]).cuda()
            normalized_output = decoder_output/lp.unsqueeze(1)
            topv, topi = normalized_output.topk(beam_width)
            topv, topi = topv.flatten(), topi.flatten()
            topv, perm_index = topv.sort(0, descending = True)

            topv = topv[:beam_width]
            decoder_input = topi[perm_index[:beam_width]].view(-1,1)
            former_index = perm_index[:beam_width] // beam_width
            #flatten * beam_width  former_index id of previous beam
            decoder_output = decoder_output[former_index]
            decoder_hidden = decoder_hidden[:, former_index]
            repet_counts = repet_counts[former_index]

            inf_uttrs = [
                inf_uttrs[former] + [decoder_input[i].item()]
                if inf_uttrs[former][-1] != EOS_id
                else inf_uttrs[former]
                for i, former in enumerate(former_index)
            ]

            if sum([words[-1] == EOS_id for words in inf_uttrs]) == beam_width:
                break

        return inf_uttrs, topv










