"""Run on the cluster
NOTE: See local_config.template.py for a local_config template.
"""
import os
import sys

email = "msy290@nyu.edu"
directory="./"
slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")

logdir = os.path.join(directory, "logs")
savedir = os.path.join(directory, "models")

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

    jobcommand = "srun python3 -B train_10_sen.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)

    outname = run_name+".txt"
    outname = os.path.join(logdir,outname)

    jobcommand += " |& tee -a " +outname
    print(jobcommand)

    jobnameattrs=run_name+"_nsteps_"+str(j['num-steps'])

    slurmfile = os.path.join(slurm_scripts, jobnameattrs + '.slurm')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name" + "=" + jobname + "\n")
        f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".out"))
        f.write("#SBATCH --error=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".err"))
        f.write("#SBATCH --qos=batch\n")
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("module load anaconda3\n")
        f.write("source activate /misc/kcgscratch1/ChoGroup/srikar/new-env\n")
        f.write("module load cuda-9.0\n")
        f.write("module load gcc-8.2\n")

        f.write(jobcommand + "\n")

    s = "sbatch --qos batch --gres=gpu:1 --constraint=gpu_12gb --nodes=1 "
    s += "--mem=100GB --time=%d:00:00 %s &" % (
        time, os.path.join(slurm_scripts, jobnameattrs + ".slurm"))
    os.system(s)


job = {
        "env-name":"nmt_easy_2-v0","n-epochs-per-word": 10000, "n-epochs": 10000,
        "num-processes": 50, "ppo-batch-size" :800, "log-dir": logdir, "save-dir": savedir,
        "save-interval":1000,"num-steps": 16,"sen_per_epoch": 1,"use-wandb":"","use-gae":"",
        }

for seed in range(1, 4):
    j = {k:v for k,v in job.items()}
    time = 48
    j["seed"] = seed
    for nsen in range(10,31,10):
        j["num-sentences"]=nsen
        j["save-dir"]=os.path.join(j["save-dir"],str(nsen))
        j["wandb-name"]=str(nsen)
        run_name="nsen_{}_seed_{}".format(nsen,seed)
        train(j, jobname=run_name, time=time)
