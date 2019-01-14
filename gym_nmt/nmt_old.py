import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models import EncoderRNN, AttnDecoderRNN

def generate_mask(batch_size,sequence_length,num_blanks):
	"""Generate the mask to be fed into the model."""

	p = np.ones((batch_size,sequence_length))

	blanks = np.random.randint(0,sequence_length-1,size = num_blanks)
	for i in range(batch_size):
		p[i,blanks[i]] = 0

	return p

def transform_input_with_is_missing_token(inputs, targets_present):
  """Transforms the inputs to have missing tokens when it's masked out.  The
  mask is for the targets, so therefore, to determine if an input at time t is
  masked, we have to check if the target at time t - 1 is masked out.
  e.g.
	inputs = [a, b, c, d]
	targets = [b, c, d, e]
	targets_present = [1, 0, 1, 0]
  then,
	transformed_input = [a, b, <missing>, d]
  Args:
	inputs:  torch.int32 Tensor of shape [batch_size, sequence_length] with tokens
	  up to, but not including, vocab_size.
	targets_present:  torch.int Tensor of shape [batch_size, sequence_length] with
	  True representing the presence of the word.
  Returns:
	transformed_input:  torch.int32 Tensor of shape [batch_size, sequence_length]
	  which takes on value of inputs when the input is present and takes on
	  value=vocab_size to indicate a missing token.
  """


	input_missing = torch.tensor(np.full(batch_sizesequence_length),fill_value = vocab_size,dtype = torch.int32)

	# The 0th input will always be present .

	zeroth_input_present = torch.ones([batch_size,1],dtype=torch.uint8 )	

	# Input present mask.
	inputs_present = torch.concat([zeroth_input_present, targets_present[:, :-1]], axis=1)

	transformed_input = torch.where(inputs_present, inputs, input_missing)
	return transformed_input

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device) # Doubt

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing



    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
