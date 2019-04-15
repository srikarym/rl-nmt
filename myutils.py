import torch
import numpy as np
import wandb

class logger():

	def __init__(self,num_sentences,task):
		self.num_sentences = num_sentences
		self.task = task
		self.success = {i:0 for i in range(num_sentences)}
		self.rows = [[] for i in range(num_sentences)]
		self.eval_rewards = [0]*num_sentences

	def print_stuff(self,i,obs,action,reward,info):

		for j in range(self.num_sentences):
			print('Sentence {}, step {}'.format(j,i))
			print('Source sentence: ',self.task.src_dict.string(obs[0][j].long(), bpe_symbol='@@ ').replace('@@ ','').replace('<pad>',''))
			print('Target sentence: ',self.task.tgt_dict.string(obs[1][j].long(), bpe_symbol='@@ ').replace('@@ ','').replace('<pad>',''))
			print('True action:',self.task.tgt_dict[int(info[:,0][j])])
			print('action predicted by the model: ',self.task.tgt_dict[int(action[j].cpu().numpy()[0].tolist())])
			print('Reward: ',reward[j])
			print()

			if int(info[:,0][j]) == int(action[j].cpu().numpy()[0].tolist()):
				self.success[j] += 1
				self.rows[j].append("T")
			else:
				self.rows[j].append("F")

	def append_rewards(self,done,reward):

		for j in range(len(reward)):
			self.eval_rewards[j] = max(self.eval_rewards[j],reward.squeeze(1).cpu().numpy().tolist()[j])


	def get_reward(self):
		return np.mean(self.eval_rewards)

	def to_wandb(self,epoch,n_words,value_loss_epoch,action_loss_epoch,dist_entropy_epoch,mean_reward_epoch,\
			ranks_epoch,total_loss_epoch,rewards,eval_reward_best,speed,bleuscore):


		columns = ["word_"+str(i) for i in range(n_words+1)]
		wandb.log({"Truth_table": wandb.Table(rows=self.rows, columns=columns)},step = epoch)

		print(" Epoch, {} Evaluation using {} episodes: mean reward {:.5f}\n".\
			format(epoch,len(self.eval_rewards),rewards))

		wandb.log({"Loss/Value_loss ": value_loss_epoch ,
		   "Loss/Action_loss": action_loss_epoch ,
		   "Loss/Dist_entropy": dist_entropy_epoch ,
		   "Rewards/Mean_reward": mean_reward_epoch ,
		   "Misc/Mean_rank": ranks_epoch,
		   "Loss/Total_loss": total_loss_epoch ,
		   "Rewards/Mean_evaluation_reward": rewards,
		   "Rewards/Best_eval_reward":eval_reward_best,
		   "Misc/Missing_words":n_words,
		   "Misc/Steps_per_sec":speed,
		   "Bleuscore":bleuscore},step = epoch)

	@staticmethod
	def log(epoch,n_words,value_loss_epoch,action_loss_epoch,dist_entropy_epoch, nll_loss_epoch, \
		mean_reward_epoch,ranks_epoch,total_loss_epoch,speed,bleuscore):
		
		wandb.log({"Loss/Value_loss ": value_loss_epoch ,
		   "Loss/Action_loss": action_loss_epoch ,
		   "Loss/Dist_entropy": dist_entropy_epoch ,
		   "Rewards/Mean_reward": mean_reward_epoch ,
		   "Misc/Mean_rank": ranks_epoch,
		   "Loss/Total_loss": total_loss_epoch ,
		   "Loss/NLL_loss": nll_loss_epoch ,
		   "Misc/Missing_words":n_words,
		   "Misc/Steps_per_sec":speed,
		   "Bleuscore":bleuscore},step = epoch)