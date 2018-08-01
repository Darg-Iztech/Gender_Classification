
class flags(object):

	def __init__(self):
		self.dev_sample_percentage = 0.1
		self.word_embed_path = "./glove.twitter.27B.50d.txt"
		self.char_embed_path = "./char_embeddings.27B.25d.txt"
		self.model_path = "./models/cnnonly"
		self.model_name = "model-14.ckpt"
		self.log_path = "./runlogs_cnn.txt"
		self.optimize = False
		self.mode = "Valid"
		self.lang = "en"


		# Model Hyperparameters
		self.embedding_dim = 25
		self.filter_sizes_cnn1 = "3,6"
		self.num_filters = 75
		self.dropout_keep_prob = 1.0
		self.l2_reg_lambda = 0.0005
		self.word_embedding_size = 50
		self.learning_rate = 0.0001
		self.num_classes = 2
		self.sequence_length = 190


		# Training parameters
		self.batch_size = 100
		self.num_epochs = 20
		self.evaluate_every = 25
		self.checkpoint_every = 1000000
		self.num_checkpoints = 1



FLAGS = flags()
		
