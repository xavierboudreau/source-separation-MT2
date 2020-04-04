import pickle

def get_from_pickle(filename):
	try:
		with open(filename, 'rb') as pickle_file:
			return pickle.load(pickle_file)
	except FileNotFoundError:
		return None

def save_to_pickle(events, filename):
	with open(filename, 'wb+') as pickle_file:
		pickle.dump(events, pickle_file)