from params import FLAGS
import tensorflow as tf
import preprocessing
from text_cnn import TextCNN
import random
import argparse
import xml.etree.ElementTree as ET
import os
import sys
import numpy as np


class Network(object):


	def __init__(self, use_trained_model, mode, input_file_path):

		self.use_trained_model = use_trained_model
		self.mode = mode

		print("----READING DATA----")
		#inputs
		self.X = tf.placeholder(tf.int32, [None, None])

		if self.mode != "Test":
			self.Y = tf.placeholder(tf.float32, [1, FLAGS.num_classes])

		# tf variables
		#self.tf_ideal_learning_rate = tf.placeholder(tf.float32, shape=[])
		self.tf_ideal_l2_reg_parameter = tf.placeholder(tf.float32, shape=[])
		self.sequence_length = tf.placeholder(tf.int32, [None])


		print("reading embeddings...")
		# read word embeddings
		self.vocabList, self.embeddings = preprocessing.readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)
		self.char_list, self.char_embeddings = preprocessing.readCharEmbeddings(path=FLAGS.char_embed_path, embedding_size=FLAGS.embedding_dim)


		#create word embeddings
		self.tf_embeddings = tf.Variable(tf.constant(0.0, shape=[self.embeddings.shape[0], self.embeddings.shape[1]]), trainable=False, name="tf_embeddings")
		self.embedding_placeholder = tf.placeholder(tf.float32, [self.embeddings.shape[0], self.embeddings.shape[1]])
		self.embedding_init = self.tf_embeddings.assign(self.embedding_placeholder)


		print("transforming dictionaries...")
		# turn list to a dict for increase in performance
		self.vocabulary = {}
		self.char_vocabulary = {}

		for i in range(len(self.vocabList)):
			self.vocabulary[self.vocabList[i]] = i

		for i in range(len(self.char_list)):
			self.char_vocabulary[self.char_list[i]] = i

		del self.char_list, self.vocabList 
		


		print("reading the text data...")
		#read tweets
		self.tr_set, self.target_val, self.seq_len = preprocessing.readData(input_file_path, self.mode)

		self.tweets = [row[1] for row in self.tr_set]
		self.users = [row[0] for row in self.tr_set]


		self.valid_set_size = int(len(self.tweets) * FLAGS.dev_sample_percentage)


		#split dataset into parts according to mode
		if mode == "Train":
			self.train_tweets = self.tweets
			self.train_users = self.users
			self.train_seqlen = self.seq_len
			print("Training set size of tweets: " + str(len(self.train_tweets))) 

		elif mode == "Valid":
			self.valid_tweets = self.tweets[:self.valid_set_size]
			self.train_tweets = self.tweets[self.valid_set_size:]
			self.valid_users = self.users[:self.valid_set_size]
			self.train_users = self.users[self.valid_set_size:]
			self.valid_seqlen = self.seq_len[:self.valid_set_size]
			self.train_seqlen = self.seq_len[self.valid_set_size:]
			print("Training set size of tweets: " + str(len(self.train_tweets)) + " Validation set size of tweets: " + str(len(self.valid_tweets))) 


		elif mode == "Test":
			self.test_tweets = self.tweets
			self.test_users = self.users
			self.test_seqlen = self.seq_len

			print("Test set size of tweets:" + str(len(self.test_tweets))) 




	######################################################################################################################
	#
	# Train function has 2 ability depending on mode
	# 	mode = "Valid" splits data into Train and Validation sets
	# 	and prints out validation results after each epoch
	#
	#	mode = "Train" gets all data, displays only training accuracy
	#
	######################################################################################################################
	def train(self, run=0):

		#get the architecture
		accuracy, train_op, loss_op, prediction, cnn = self.architecture()
		saver = tf.train.Saver()

		with tf.Session() as sess:

			# init variables
			init = tf.global_variables_initializer()

			sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.embeddings})
			del self.embeddings

			sess.run(init)
			
			
			if self.use_trained_model == True:
				load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
				saver.restore(sess, load_as)
				print("Loading the pretrained model from " + str(load_as))
			

			print("----TRAINING STARTED----")
			# for each epoch
			for epoch in range(FLAGS.num_epochs):
				
				train_true_pred = 0.0
				train_count = 0.0
				epoch_loss = 0.0
				display_loss = 0.0
				epoch_accuracy = 0.0

				batch_count = int(len(self.train_tweets) / FLAGS.batch_size)
				
				# for each batch
				for i in range(batch_count):
					

					char_batch_x, _ = preprocessing.prepCharBatchData(self.mode, self.train_tweets, self.train_users, self.target_val, i, batch_size=FLAGS.batch_size,
												    max_tweet_size=FLAGS.sequence_length)
					char_batch_x = preprocessing.char2id(char_batch_x, self.char_vocabulary)

					
					batch_x, batch_y, batch_seqlen = preprocessing.prepWordBatchData(self.mode, self.train_tweets, self.train_users, self.target_val, self.train_seqlen, i)
					batch_x = preprocessing.word2id(batch_x, self.vocabulary)



					
					#shuffle the tweets
					c = list(zip(batch_x, batch_seqlen, char_batch_x))
					random.shuffle(c)
					batch_x, batch_seqlen, char_batch_x = zip(*c)

					del c
					
						
					_, accu, predi, user_loss = sess.run([train_op,accuracy, prediction, loss_op],  feed_dict={
																self.X: batch_x,
																self.Y: batch_y,
																self.sequence_length: batch_seqlen,
																self.tf_ideal_l2_reg_parameter: FLAGS.l2_reg_lambda,
																cnn.input_x: char_batch_x, cnn.input_y: batch_y,
																cnn.dropout_keep_prob: FLAGS.dropout_keep_prob})

					train_true_pred += accu
					train_count += 1
					display_loss += user_loss
					if i% FLAGS.evaluate_every == 0:
						print("Epoch " + str(epoch) + " Batch " + str(i) + " , Minibatch Loss= " + str(display_loss) + " , Training Accuracy= " + str(train_true_pred/train_count) + 								", progress= %" + "{0:.2f}".format((float(i) / batch_count) * 100))
						if i% (FLAGS.evaluate_every*20) == 0:
							with open(FLAGS.log_path, 'a') as f:
								line = "Epoch " + str(epoch) + " Batch " + str(i) + " , Minibatch Loss= " + str(display_loss)
								line = line + " , Training Accuracy= " + str(train_true_pred/train_count)
								f.write(line + "\n")

						train_true_pred = 0
						train_count = 0
						epoch_loss += display_loss
						display_loss = 0


				if self.mode == "Valid":

					batch_count = int(len(self.valid_tweets) / FLAGS.batch_size)
					true_pred = 0.0
					false_pred = 0.0
					att_word = 0.0
					att_char = 0.0
					att_capt = 0.0

					for i in range(batch_count):
						char_batch_x, _ = preprocessing.prepCharBatchData(self.mode, self.valid_tweets, self.valid_users,self.target_val, i,batch_size=FLAGS.batch_size, 														max_tweet_size=FLAGS.sequence_length)

						char_batch_x = preprocessing.char2id(char_batch_x, self.char_vocabulary)

						batch_x, batch_y, batch_seqlen = preprocessing.prepWordBatchData(self.mode, self.valid_tweets, self.valid_users, self.target_val, self.valid_seqlen, i)
						batch_x = preprocessing.word2id(batch_x, self.vocabulary)


						# shuffle the tweets
						c = list(zip(batch_x, batch_seqlen, char_batch_x))
						random.shuffle(c)
						batch_x, batch_seqlen, char_batch_x = zip(*c)

						del c

						loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={
																self.X: batch_x,
																self.Y: batch_y, 
																self.sequence_length: batch_seqlen,
																self.tf_ideal_l2_reg_parameter: FLAGS.l2_reg_lambda,
																cnn.input_x: char_batch_x, cnn.input_y: batch_y,
																cnn.dropout_keep_prob: FLAGS.dropout_keep_prob})
						
				
						pred = pred.reshape(2).tolist()
						batch_y = batch_y.reshape(2).tolist()
						if pred.index(max(pred)) == batch_y.index(max(batch_y)):
							true_pred += 1
						else:
							false_pred += 1


					epoch_accuracy = true_pred / (true_pred + false_pred)
					print("Validation accuracy of users: " + str(true_pred / (true_pred + false_pred)))
					print("Epoch loss: " + str(epoch_loss))
					with open(FLAGS.log_path, 'a') as f:
						line = "Validation accuracy of users: " + str(true_pred / (true_pred + false_pred)) + "\n" + "Epoch loss: " + str(epoch_loss)
						f.write(line + "\n")


				if not FLAGS.optimize or epoch_accuracy>=0.75:
					model_name = "model-" + str(run) + "-" + str(epoch) + ".ckpt"
					save_as = os.path.join(FLAGS.model_path, model_name)
					save_path = saver.save(sess, save_as)
					print("Training model saved in path: %s" % save_path)

			'''
			if FLAGS.optimize:
				model_name = "model-" + str(run) + ".ckpt"
				save_as = os.path.join(FLAGS.model_path, model_name)
				save_path = saver.save(sess, save_as)
				print("Validated model saved in path: %s" % save_path)
			'''







	######################################################################################################################
	#
	# Test function only work when it is called.
	# If it is called without setting mode to "Test"
	# There will be data splitting error.
	#  !!! SET MODE TO TEST TO USE THIS FUNCTION !!!
	#
	######################################################################################################################
	def test(self):
		accuracy, train_op, loss_op, prediction, cnn = self.architecture()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.embeddings})
			
			if self.use_trained_model == True:
				print("loading pre-trained model")
				load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
				saver.restore(sess, load_as)

			batch_count = int(len(self.test_tweets) / FLAGS.batch_size)
			true_pred = 0.0
			false_pred = 0.0
			predictions = []


			print("----TESTING STARTED----")
			for i in range(batch_count):
				char_batch_x, _ = preprocessing.prepCharBatchData(self.mode, self.test_tweets, self.test_users, self.target_val, i,batch_size=FLAGS.batch_size,max_tweet_size=FLAGS.sequence_length)
				char_batch_x = preprocessing.char2id(char_batch_x, self.char_vocabulary)

				batch_x, batch_y, batch_seqlen = preprocessing.prepWordBatchData(self.mode, self.test_tweets, self.test_users, self.target_val, self.test_seqlen, i)
				batch_x = preprocessing.word2id(batch_x, self.vocabulary)

				# shuffle the tweets
				c = list(zip(batch_x, batch_seqlen, char_batch_x))
				random.shuffle(c)
				batch_x, batch_seqlen, char_batch_x = zip(*c)
				

				pred = sess.run([prediction], feed_dict={self.X: batch_x, 
														self.sequence_length: batch_seqlen,
														self.tf_ideal_l2_reg_parameter: FLAGS.l2_reg_lambda,
														cnn.input_x: char_batch_x,
														cnn.dropout_keep_prob: FLAGS.dropout_keep_prob})

				
				predictions.append(pred[0][0])


			print("----TESTING IS FINISHED-------") 

			return self.test_users, predictions








	######################################################################################################################
	#Neural network specifications
	def architecture(self):

		#######  DEFINITIONS #######

		multiplier = len(FLAGS.filter_sizes_cnn1.split(","))

		#create system parameters
		weights = {'fc1' : tf.Variable(tf.random_normal([(multiplier*FLAGS.num_filters), FLAGS.num_classes]), name="fc1-weights"),
			   'att2-W-char' : tf.Variable(tf.random_normal([multiplier*FLAGS.num_filters, multiplier*FLAGS.num_filters]), name='att2-weights-W-char'),
			   'att2-v-char' : tf.Variable(tf.random_normal([multiplier*FLAGS.num_filters]), name='att2-weigths-v-char'),}

		bias = {'fc1' : tf.Variable(tf.random_normal([FLAGS.num_classes]), name="fc1-bias-noreg"),
			'att2-W-char' : tf.Variable(tf.random_normal([multiplier*FLAGS.num_filters]), name="att2-char-bias-noreg")}

		# cnn initialization
		cnn = TextCNN(
			sequence_length=FLAGS.sequence_length,
			num_classes=FLAGS.num_classes,
			embedding_size=FLAGS.embedding_dim,
			filter_sizes=list(map(int, FLAGS.filter_sizes_cnn1.split(","))),
			num_filters=FLAGS.num_filters,
			vocab_size=self.char_embeddings.shape[0],
			l2_reg_lambda=FLAGS.l2_reg_lambda)
		del self.char_embeddings	
		


		#ARCHITECTURE 

		# forward pass
		
		cnn_output = cnn.h_pool_flat

		# attention on char - user level
		att_context_vector_char = tf.tanh(tf.matmul(cnn_output, weights["att2-W-char"]) + bias["att2-W-char"])
		attentions_char = tf.nn.softmax(tf.matmul(att_context_vector_char, tf.expand_dims(weights["att2-v-char"], -1)), axis=0)
		attention_output_char = tf.reduce_sum(cnn_output * attentions_char, 0)
		attention_output_char = tf.reshape(attention_output_char, [1, multiplier*FLAGS.num_filters])
			
		# BPTT
		logits = tf.matmul(attention_output_char, weights['fc1']) + bias['fc1']
		prediction = tf.nn.softmax(logits)

		if self.mode != "Test":
			loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Y))

			# add L2 regularization
			l2 = self.tf_ideal_l2_reg_parameter * sum(
				tf.nn.l2_loss(tf_var)
				for tf_var in tf.trainable_variables()
				if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
			)
			loss_op += l2

			# optimizer
			optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			train_op = optimizer.minimize(loss_op)

			# calculate training accuracy for checking correctness
			correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


			return accuracy, train_op, loss_op, prediction, cnn
		else:
			return None, None, None, prediction, cnn
			




	######################################################################################################################
	#deletion of the object
	def __del__(self):
		print("Deleted the network object..preparing for the another run")



###############################################################################################################################################
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help='absolute path to input file')
	parser.add_argument('-o', help='absolute path of output file')

	args = parser.parse_args()
	input_file_path = args.i
	output_file_path = args.o


	if FLAGS.mode == "Test":
	####################################   ENGLISH   ##########################################################
		tf.reset_default_graph()
		FLAGS.lang = "en"
		FLAGS.learning_rate = 0.001
		FLAGS.l2_reg_lambda = 0.00001
		FLAGS.num_filters = 75
		FLAGS.filter_sizes_cnn1 = "3,6"
		FLAGS.model_path = "./models/cnnonly"
		FLAGS.model_name = "model-14.ckpt"
		
		network = Network(use_trained_model=True, mode=FLAGS.mode, input_file_path=input_file_path)
		users, pred = network.test()
		
		for i in range(len(pred)):
			author_id = users[i*100]

			if pred[i][0] > pred[i][1]:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "male", "gender_img":"","gender_comb":""})
			else:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "female", "gender_img":"","gender_comb":""})

			tree = ET.ElementTree(root)
			write_output = output_file_path + "/" +FLAGS.lang+ "/" + str(author_id) + ".xml"

			if not os.path.exists(output_file_path + "/" +FLAGS.lang+ "/"):
				os.makedirs(output_file_path + "/" +FLAGS.lang+ "/")

			tree.write(write_output)
			
	####################################   SPANISH   ##########################################################
		tf.reset_default_graph()
		FLAGS.lang = "es"
		FLAGS.learning_rate = 0.001
		FLAGS.l2_reg_lambda = 0.00005
		FLAGS.num_filters = 60
		FLAGS.filter_sizes_cnn1 = "3,6"
		FLAGS.model_path = "./models/cnnonly/es"
		FLAGS.model_name = "model-es_em-0-16-size60.ckpt"
	
		network = Network(use_trained_model=True, mode=FLAGS.mode, input_file_path=input_file_path)
		users, pred = network.test()
		for i in range(len(pred)):
			author_id = users[i*100]

			if pred[i][0] > pred[i][1]:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "male", "gender_img":"","gender_comb":""})
			else:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "female", "gender_img":"","gender_comb":""})

			tree = ET.ElementTree(root)
			write_output = output_file_path + "/" +FLAGS.lang+ "/" + str(author_id) + ".xml"

			if not os.path.exists(output_file_path + "/" +FLAGS.lang+ "/"):
				os.makedirs(output_file_path + "/" +FLAGS.lang+ "/")

			tree.write(write_output)	
	
	####################################   ARABIC   ##########################################################
		tf.reset_default_graph()
		FLAGS.lang = "ar"
		FLAGS.learning_rate = 0.001
		FLAGS.l2_reg_lambda = 0.00001
		FLAGS.num_filters =  50
		FLAGS.filter_sizes_cnn1 = "3,6,9"
		FLAGS.model_path = "./models/cnnonly/ar"
		FLAGS.model_name = "model-2-11-size50.ckpt"
	
		network = Network(use_trained_model=True, mode=FLAGS.mode, input_file_path=input_file_path)
		users, pred = network.test()
		for i in range(len(pred)):
			author_id = users[i*100]

			if pred[i][0] > pred[i][1]:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "male", "gender_img":"","gender_comb":""})
			else:
				root = ET.Element("author", attrib={"id": str(author_id), "lang": "/" +FLAGS.lang+ "/", "gender_txt": "female", "gender_img":"","gender_comb":""})

			tree = ET.ElementTree(root)
			write_output = output_file_path + "/" +FLAGS.lang+ "/" + str(author_id) + ".xml"

			if not os.path.exists(output_file_path + "/" +FLAGS.lang+ "/"):
				os.makedirs(output_file_path + "/" +FLAGS.lang+ "/")

			tree.write(write_output)
			
			
			
	elif FLAGS.mode == "Train":
		FLAGS.learning_rate = 0.001
		FLAGS.l2_reg_lambda = 0.00001
		network = Network(use_trained_model=False, mode=FLAGS.mode, input_file_path=input_file_path)

		network.train()

	elif FLAGS.mode == "Valid":
		if FLAGS.optimize:
			l_rate = [0.001]     # learning rate and reg_params should be given as list
			reg_param = [0.0001] # This way, you can use grid search for hyperparameter optimization
			i = 0
			for alpha in l_rate:
				for lambda_reg in reg_param: 
					FLAGS.learning_rate = alpha
					FLAGS.l2_reg_lambda = lambda_reg
					FLAGS.num_filters = 100
					FLAGS.lang = "ar"
					FLAGS.log_path = "./runlogs_cnn_ar.txt"
					tf.reset_default_graph()
					network = Network(use_trained_model=False, mode=FLAGS.mode, input_file_path=input_file_path)
					with open(FLAGS.log_path, 'a') as f:
						line = "Started training with-->  alpha=" + str(FLAGS.learning_rate) + " lambda=" + str(FLAGS.l2_reg_lambda)
						f.write(line + "\n")
					print("Started training with-->  alpha=" + str(FLAGS.learning_rate) + " lambda=" + str(FLAGS.l2_reg_lambda))
					network.train(i)
					i+=1
					del network 
		else:
			FLAGS.learning_rate = 0.001
			FLAGS.l2_reg_lambda = 0.00001
			network = Network(use_trained_model=False, mode=FLAGS.mode, input_file_path=input_file_path)
			network.train()


