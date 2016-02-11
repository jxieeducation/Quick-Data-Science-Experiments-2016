import random

POP_SIZE = 25
GENE_LENGTH = 9

div = 1e-7
para_ps = range(0, 400, 1)
para_is = range(0, 250, 1)
para_ds = range(0, 250, 1)

class PID:
	def __init__(self, kp=None, ki=None, kd=None, bits=None):
		if not bits:
			self.kp = kp
			self.ki = ki
			self.kd = kd
			self.bits = self.encode()
		else:
			# decode bits into normal stuff
			self.bits = bits
			self.kp, self.ki, self.kd = self.decode(bits)

	def __repr__(self):
		return "{kp: %d, ki: %d, kd: %d}" % (self.kp, self.ki, self.kd)

	def get_kp(self):
		return self.kp * div

	def get_ki(self):
		return self.ki * div

	def get_kd(self):
		return self.kd * div

	# 1, 2, 3 --> 000000001000000010000000011 
	def encode(self):
		bits = ""
		for n in [self.kp, self.ki, self.kd]:
			val = bin(n)[2:]
			val = ('0' * (GENE_LENGTH - len(val))) + val
			bits += val
		return bits

	def decode(self, bin_repr):
		kp = bin_repr[:GENE_LENGTH]
		ki = bin_repr[GENE_LENGTH:GENE_LENGTH * 2]
		kd = bin_repr[GENE_LENGTH * 2:]
		kp = int(kp, 2)
		ki = int(ki, 2)
		kd = int(kd, 2)
		return kp, ki, kd


def get_initial_population():
	pid_list = []
	for i in range(POP_SIZE):
		kp = random.choice(para_ps)
		ki = random.choice(para_is)
		kd = random.choice(para_ds)
		pid_list += [PID(kp, ki, kd)]
	return pid_list

#https://github.com/yati-sagade/Genetic-algorithm-demo/blob/master/ga.py










