import fairseq
import numpy as np
import torch

from arguments import get_args
args = get_args()


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def load_cpickle_gc(fname):
	output = open(fname, 'rb')

	gc.disable()
	mydict = pickle.load(output)
	gc.enable()
	output.close()
	return mydict


def load_train_data():

	print('Loading data')

	train_data = load_cpickle_gc('data/data.pkl')
	task = load_cpickle_gc('data/task.pkl')

	if (args.reduced):
		src = fairseq.data.Dictionary()
		tgt = fairseq.data.Dictionary()

		data = []

		for i in range(args.num_sentences):
			ssen = train_data[i]['net_input']['src_tokens'][0].numpy().tolist()
			tsen = train_data[i]['target'][0].numpy().tolist()
			
			for s in ssen:
				src.add_symbol(task.src_dict[s])
			for t in tsen:
				tgt.add_symbol(task.tgt_dict[t])

			
		for i in range(args.num_sentences):
			d = {}
			d['id']=i
			ssen = train_data[i]['net_input']['src_tokens'][0].numpy().tolist()
			tsen = train_data[i]['target'][0].numpy().tolist()
			
			new_ssen = []
			for s in ssen:
				word = task.src_dict[s]
				ind = src.index(word)
				new_ssen.append(ind)
			
			new_tsen = []
			for t in tsen:
				word = task.tgt_dict[t]
				ind = tgt.index(word)
				new_tsen.append(ind)
			
			prev = [2] + new_tsen[:-1]
			
			prev = np.array(prev).reshape(1,len(prev))
			new_ssen = np.array(new_ssen).reshape(1,len(new_ssen))
			new_tsen = np.array(new_tsen).reshape(1,len(new_tsen))
			

			d['net_input'] = {'src_tokens':torch.tensor(new_ssen),'prev_output_tokens':torch.tensor(prev)}
			d['target'] = torch.tensor(new_tsen)
			data.append(d)

		train_data = data

		task = AttrDict()

		task.update({'source_dictionary':src})
		task.update({'src_dict':src})
		task.update({'target_dictionary':tgt})
		task.update({'tgt_dict':tgt})


	print('Data loaded')

	return train_data,task