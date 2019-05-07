class args:
    beam=5
    cpu=False
    data=['data-bin/iwslt14.tokenized.de-en']
    diverse_beam_groups=-1
    diverse_beam_strength=0.5
    fp16=False
    fp16_init_scale=128
    fp16_scale_tolerance=0.0
    fp16_scale_window=None
    gen_subset='test'
    lazy_load=False
    left_pad_source='False'
    left_pad_target='False'
    lenpen=1
    log_format=None
    log_interval=1000
    match_source_len=False
    max_len_a=0
    max_len_b=200
    max_sentences=128
    max_source_positions=1024
    max_target_positions=1024
    max_tokens=None
    memory_efficient_fp16=False
    min_len=1
    min_loss_scale=0.0001
    model_overrides='{}'
    nbest=1
    no_beamable_mm=False
    no_early_stop=False
    no_progress_bar=False
    no_repeat_ngram_size=0
    num_shards=1
    num_workers=0
    path=''
    prefix_size=0
    print_alignment=False
    quiet=True
    raw_text=False
    remove_bpe='@@ '
    replace_unk=None
    required_batch_size_multiple=8
    results_path=None
    sacrebleu=False
    sampling=False
    sampling_temperature=1
    sampling_topk=-1
    score_reference=False
    seed=1
    shard_id=0
    skip_invalid_size_inputs_valid_test=False
    source_lang=None
    target_lang=None
    task='translation'
    tensorboard_logdir=''
    threshold_loss_scale=None
    unkpen=0
    unnormalized=False
    upsample_primary=1
    user_dir=None

class SacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def corpus_bleu(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])
    
    def sentence_bleu(self):
        senbleu = 0
        for hyp,ref in zip(self.sys,self.ref):
            senbleu += self.sacrebleu.sentence_bleu(hyp,ref)
        return senbleu/len(self.sys)
