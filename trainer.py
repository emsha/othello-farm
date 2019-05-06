import torch.optim as optim

class Trainer():
	def __init__(epochs, learningrate):
		self.epochs = epochs
		self.learningrate = learningrate

	def train_on_data(self, net, inputs, labels, batch_size):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr= self.learningrate, momentum=0.5)

		for _ in epochs:
			for in_batch, labels_batch in self.make_batches(batch_size, inputs, labels):
				optimizer.zero_grad()
				outputs = net(in_batch)
				loss = criterion(outputs, labels_batch)
				loss.backward()
				optimizer.step()

	def make_batches(self, batch_size, inputs, labels):
		n = batch_size
		batch_in = [inputs[i * n:(i + 1) * n] for i in range((len(inputs) + n - 1) // n )]
		batch_labels = [labels[i * n:(i + 1) * n] for i in range((len(labels) + n - 1) // n )]
		return batch_in, batch_labels
	
	def train_on_batch(self, in_batch, label_batch):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr= self.learningrate, momentum=0.5)

		for _ in epochs:
			optimizer.zero_grad()
			outputs = net(in_batch)
			loss = criterion(outputs, label_batch.long())
			loss.backward()
			optimizer.step()