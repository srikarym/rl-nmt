"""Run on the cluster
NOTE: See local_config.template.py for a local_config template.
"""
import os
import sys

email = "msy290@nyu.edu"
directory="/misc/kcgscratch1/ChoGroup/srikar/rl-nmt"
run = "week11_nobp"
slurm_logs = os.path.join(directory, "slurm_logs",run)
slurm_scripts = os.path.join(directory, "slurm_scripts",run)

logdir = os.path.join(directory, "logs",run)
savedir = os.path.join(directory, "models",run)

if not os.path.exists(slurm_logs):
	os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
	os.makedirs(slurm_scripts)
if not os.path.exists(logdir):
	os.makedirs(logdir)
if not os.path.exists(savedir):
	os.makedirs(savedir)


def train(flags, jobname=None, time=24):
	num_processes = flags["num-processes"]

	jobcommand = "srun python3 -B train.py "
	args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
	jobcommand += " ".join(args)

	outname = run_name+".txt"
	outname = os.path.join(logdir,outname)

	jobcommand += " |& tee " +outname
	print(jobcommand)

	jobnameattrs=run_name+ j["run-name"]
	slurmfile = os.path.join(slurm_scripts, jobnameattrs + '.slurm')
	with open(slurmfile, 'w') as f:
		f.write("#!/bin/bash\n")
		f.write("#SBATCH --job-name" + "=" + jobname + "\n")
		f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".out"))
		f.write("#SBATCH --qos=batch\n")
		f.write("#SBATCH --mail-type=END,FAIL\n")
		f.write("#SBATCH --mail-user=%s\n" % email)
		f.write("module load anaconda3\n")
		f.write("source activate /misc/kcgscratch1/ChoGroup/srikar/new-env\n")
		f.write("module load cuda-9.0\n")
		f.write("module load gcc-8.2\n")

		f.write(jobcommand + "\n")

	s = "sbatch --qos batch --gres=gpu:1 --constraint=gpu_12gb --nodes=1 "
	s += "--mem=200GB --time=%d:00:00 %s &" % (
		time, os.path.join(slurm_scripts, jobnameattrs + ".slurm"))
	os.system(s)


job = {
		"env-name":"nmt_train-v0","n-epochs-per-word": 50, "n-epochs": 2000,
		"num-processes": 100, "ppo-batch-size" :450, "log-dir": logdir, "save-dir": savedir,
		"save-interval":500,"num-steps": 100,"sen_per_epoch": 1,"use-wandb":"",
		"eval-interval":1,"entropy-coef":0.04,"use-gae":"","checkpoint":"",
		"file-path":"checkpoint_best.pt","threshold":"1.1"
		}

for n_words in [5,10]:
	for nsen in [10,100]:
		j = {k:v for k,v in job.items()}
		time = 48
		j["n-words"] = n_words
		old_save_dir = j["save-dir"]
		j["num-sentences"]=nsen
		run_name=run+"_nwords_{}_sen_{}".format(n_words,nsen)
		j["save-dir"]=os.path.join(old_save_dir,run_name)
		j["wandb-name"]=run
		j["run-name"]= "nwords_" + str(n_words) +"_sen_"+str(nsen)
		jobname = 'n'+str(n_words)+"-"+str(nsen)
		train(j, jobname=jobname, time=time)
