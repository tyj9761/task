import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data ,datasets


#使用torchtext对数据进行#

#设置随机种子数，该数可以保证随机数是可重复的
SEED = 1234

#设置种子
torch.manual_seed(SEED)
torch.backends.cudnn.determinstic = True 

#读取数据和标签
TEXT = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)

#下载数据集并划分训练集和测试集
trai_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

#划分验证集
train_data, valid_data = train_data.split(splits_ratio=0.8 , random_state = random.seed(SEED))

#构建查找表 bulid_vocab ？为什么没有进行向量表示 因为后面有embedding层

MAX_VOCVB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCVB_SIZE)
LABEL.build_vocab(train_data)

#构建迭代器
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucktItrator.splits(
	(train_data, valid_data, test_data),
	batch_size = BATCH_SIZE,
	device = device)

#开始构造模型
class RNN(nn.Module):
	def __init__(self, input_dim, embedding_dim ,hidden_dim, output_dim):

		super().__init__()

		self.embedding = nn.Embedding(input_dim, embedding_dim)

		self.rnn = nn.RNN(embedding_dim, hidden_dim)

		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, text):

		#text = [sentence len, batch size]

		embedded = self.embedding(text)

		#embedded = [sentence len, batch size, embed dim]

		output, hidden = self.rnn(embedded)

		#output = [sentence len, batch size, hid dim]
		#hidden = [1, batcha size, hid dim]

		assert torch.equal(output[-1,:,:], hidden.squeeze(0))

		return self.fc(hidden.squeeze(0))


INPUT_DIM = len(TEXT.vocab)
EMBEDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 100

model = RNN(INPUT_DIM, EMBEDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

#训练模型#

#设置优化器 ，使用随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#定义损失函数
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

#计算准确率的函数
def binary_accracy(preds, y):

	rounded_pred = torch.round(torch.sigmod(preds))
	correct = (rounded_pred == y).float()

	acc = correct.sum() / len(correct)
	return acc

#训练测试集
def train(model, iterator, optimizer, criterion):

	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in interator:

		optimizer.zero_grad()#手动将grad清零

		predictions = model(batch.text).squeeze(1) #model里

		loss = criterion(predictions, batch.label)

		acc = binary_accracy(predictions, batch.label)

		loss.backward()

		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

#验证集
def evaluate(model, iterator, criterion):

	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torc.n_grad():

		for batch in iterator:

			predictions = model(batch.text).squeeze(1)

			loss = criterion(predictions, batch.label)

			acc = binary_accracy(predictions, batch.label)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

#计算epoch时间
def epoch_time(star_time, end_time):
	elapsed_time = end_time - star_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_seccs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_seccs


N_EPOCH = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCH):

	start_time = time.time()

	train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	valid_loss, valid_acc = evaluate(model,valid_iterator, criterion)

	end_time = time.time()

	epoch_mins, epoch_sec = epoch_time(start_time, end_time)

	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), 'tut1-model.pt')

	print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


#将训练好的参数放到验证集上
model.load_state_dict(torch.load('tut1-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')