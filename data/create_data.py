import fairseq
from fairseq import tasks
from copy import deepcopy
import _pickle as pickle

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

arg  = AttrDict()
arg.update({'task':'translation'})
arg.update({'data':['iwslt14.tokenized.de-en']})
arg.update({'lazy_load':False})
arg.update({'left_pad_source':False})
arg.update({'left_pad_target':False})
arg.update({'source_lang':None})
arg.update({'target_lang':None})
arg.update({'raw_text':False})
arg.update({'train_subset':'train'})
arg.update({'valid_subset':'valid'})
arg.update({'max_source_positions':1024})
arg.update({'max_target_positions':1024})
task = fairseq.tasks.setup_task(arg)
task.load_dataset('train')

epoch_itr = task.get_batch_iterator(
	dataset=task.dataset(arg.train_subset),
	max_tokens=4000,
	max_sentences=1,
	max_positions=(100,100),
	ignore_invalid_inputs=True,
	required_batch_size_multiple=1,

)
train_data = list(epoch_itr.next_epoch_itr())

with open('parrot.pkl', 'wb') as f:
	pickle.dump(train_data, f)
