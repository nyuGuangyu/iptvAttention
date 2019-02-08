import os
import numpy as np
import cPickle as cp
from tqdm import tqdm
import tensorflow as tf
import datetime
import operator


infile_name = "data_attention.cp"
outfile_name = "preprocessed_data.cp"
begin_date = datetime.datetime(2014, 8, 1)
end_date = datetime.datetime(2014, 9, 1)
cut_off_date = (end_date - begin_date) // 4 *3 + begin_date

def unpickle(file):
	with open(file, 'rb') as fo:
		dic = cp.load(fo)
	return dic

def pk_to_disk(path, obj):
	with open(path,'wb') as f:
	    cp.dump(obj,f)


def main():
	raw_data = unpickle(infile_name)
	# store processed data to feed to data loader, see data loader interface
	processed_data = {}
	total_count = 0 
	zero_count = 0
	nonzero_count = 0

	train_samples = []
	test_samples = []

	for (user,dt),l in raw_data.items():
		total_count += 1
		print total_count, len(raw_data)
		if sum(l[1]) == 0:
			zero_count += 1
		else:
			nonzero_count += 1
			if dt <= cut_off_date:
				# collect user,dt,hour,6scores,label in tuple, later can sort by dt or by user,dt
				train_samples.append((user,dt,dt.hour,l[0],l[2],l[3],l[4],l[5],l[6],l[1])) 
			else:
				test_samples.append((user,dt,dt.hour,l[0],l[2],l[3],l[4],l[5],l[6],l[1])) 

	# sort by dt, ascending order.
	train_samples.sort(key=operator.itemgetter(1)) 
	test_samples.sort(key=operator.itemgetter(1))

	# extract np.array: 
	# x_train_base(172,6), x_train_hour(24,1), y_train(172,1), x_test_base(172,6), x_test_hour(24,1), y_test(172,1)
	train_bases = []
	test_bases = []
	train_hours = []
	test_hours = []
	train_labels = []
	test_labels = []

	for sample in train_samples:
		train_bases.append([sample[3],sample[4],sample[5],sample[6],sample[7],sample[8]])
		onehot_hour = [0]*24
		onehot_hour[sample[2]-1] = 1
		train_hours.append(onehot_hour)
		train_labels.append(sample[-1])
	for sample in test_samples:
		test_bases.append([sample[3],sample[4],sample[5],sample[6],sample[7],sample[8]])
		onehot_hour = [0]*24
		onehot_hour[sample[2]-1] = 1
		test_hours.append(onehot_hour)
		test_labels.append(sample[-1])


	x_train_base = np.array(train_bases)
	x_test_base = np.array(test_bases)
	x_train_hour = np.array(train_hours)
	x_test_hour = np.array(test_hours)
	y_train = np.array(train_labels)
	y_test = np.array(test_labels)
	print x_train_base.shape
	print x_test_base.shape
	print x_train_hour.shape
	print x_test_hour.shape
	print y_train.shape
	print y_test.shape

	processed_data['x_train_base'] = x_train_base
	processed_data['x_test_base'] = x_test_base
	processed_data['x_train_hour'] = x_train_hour
	processed_data['x_test_hour'] = x_test_hour
	processed_data['y_train'] = y_train
	processed_data['y_test'] = y_test

	pk_to_disk(outfile_name,processed_data)

if __name__ == '__main__':
    main()
