# coding: utf-8

import numpy as np
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
import logging


def chunks(*args, **kwargs):
	return list(chunksYielder(*args, **kwargs))
def chunksYielder(l, n):
	"""Yield successive n-sized chunks from l."""
	if l is None:
		return []
	for i in range(0, len(l), n):
		yield l[i:i + n]

def getDistilBertRepresentations\
(
	model,
	inputs,
	layer='distilbert', # distilbert, pre_classifier, dropout, classifier
	
):
	"""
		Get only one input sample
		model is a TFDistilBertForSequenceClassification
		See https://huggingface.co/transformers/_modules/transformers/modeling_tf_distilbert.html#TFDistilBertModel
	"""
	distilbert_output = model.distilbert(inputs)
	hidden_state = distilbert_output[0]
	pooled_output = hidden_state[:, 0]
	if layer == 'distilbert':
		return pooled_output
	pooled_output = model.pre_classifier(pooled_output)
	if layer == 'pre_classifier':
		return pooled_output
	pooled_output = model.dropout(pooled_output, training=False)
	if layer == 'dropout':
		return pooled_output
	logits = model.classifier(pooled_output)
	if layer == 'classifier':
		return logits
	else:
		raise Exception("Please choose a layer in ['distilbert', 'pre_classifier', 'dropout', 'classifier']")


distilBertTokenizerSingleton = None
def distilBertEncode\
(
	doc,
	maxLength=512,
	multiSamplage=False,
	multiSamplageMinMaxLengthRatio=0.3,
	bertStartIndex=101,
	bertEndIndex=102,
	preventTokenizerWarnings=False,
	loggerName="transformers.tokenization_utils",
	proxies=None,
	logger=None,
	verbose=True,
):
	"""
		Return an encoded doc for DistilBert.
		This function return a list of document parts if you set multiSamplage as True.
	"""
	# We set the logger level:
	if preventTokenizerWarnings:
		previousLoggerLevel = logging.getLogger(loggerName).level
		logging.getLogger(loggerName).setLevel(logging.ERROR)
	# We init the tokenizer:
	global distilBertTokenizerSingleton
	if distilBertTokenizerSingleton is None:
		distilBertTokenizerSingleton = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', proxies=proxies)
	tokenizer = distilBertTokenizerSingleton
	# We tokenize the doc:
	if isinstance(doc, list):
		doc = " ".join(doc)
	doc = tokenizer.encode(doc, add_special_tokens=False)
	# In case we want multiple parts:
	if multiSamplage:
		# We chunk the doc:
		parts = chunks(doc, (maxLength - 2))
		# We add special tokens (only one [CLS] at the begining and one [SEP] at the end,
		# even for entire documents):
		parts = [[bertStartIndex] + part + [bertEndIndex] for part in parts]
		# We remove the last part:
		if len(parts) > 1 and len(parts[-1]) < int(maxLength * multiSamplageMinMaxLengthRatio):
			parts = parts[:-1]
		# We pad the last part:
		parts[-1] = parts[-1] + [0] * (maxLength - len(parts[-1]))
		# We check the length of each part:
		for part in parts:
			assert len(part) == maxLength
		# We reset the logger:
		if preventTokenizerWarnings:
			logging.getLogger(loggerName).setLevel(previousLoggerLevel)
		return parts
	# In case we have only one part:
	else:
		# We truncate the doc:
		doc = doc[:(maxLength - 2)]
		# We add special tokens:
		doc = [bertStartIndex] + doc + [bertEndIndex]
		# We pad the doc
		doc = doc + [0] * (maxLength - len(doc))
		# We check the length:
		assert len(doc) == maxLength
		# We reset the logger:
		if preventTokenizerWarnings:
			logging.getLogger(loggerName).setLevel(previousLoggerLevel)
		return doc


class DeepStyle:
	"""
		DeepStyle provides an interface of the DBert-ft model. It allows to embed document in a stylometric space.
	"""
	def __init__(self, path, batchSize=16, layer='distilbert'):
		"""
			path is the directory of the DBert-ft model
		"""
		self.batchSize = batchSize
		self.layer = layer
		self.path = path
		if not (os.path.isfile(self.path + "/config.json") and os.path.isfile(self.path + "/tf_model.h5")):
			raise Exception("You need to provide the pretrained model directory that contains tf_model.h5 and config.json")
		self.model = None
		self.__load()
	
	def __load(self):
		dbertConf = DistilBertConfig.from_pretrained(self.path + '/config.json')
		self.model = TFDistilBertForSequenceClassification.from_pretrained\
		(
			self.path + '/tf_model.h5',
			config=dbertConf,
		)

	def embed(self, text):
		"""
			Give raw text and get the style vector. If the text is longer than 512 wordpieces, it will be split and the style vector will be the mean of all embeddings.
		"""
		encodedText = distilBertEncode\
		(
			text,
			maxLength=512,
			multiSamplage=True,
			preventTokenizerWarnings=True,
		)
		encodedBatches = chunks(encodedText, self.batchSize)
		embeddings = []
		for encodedBatch in encodedBatches:
			outputs = getDistilBertRepresentations(self.model, np.array(encodedBatch), layer=self.layer)
			for output in outputs:
				embeddings.append(np.array(output))
		return np.mean(embeddings, axis=0)