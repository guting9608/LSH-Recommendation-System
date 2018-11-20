import pyspark
import itertools
import sys


if __name__ == "__main__":
	sc = pyspark.SparkContext()
	data = sc.textFile(sys.argv[1])

	def mapper(data):
		data = data.split(",")
		user = data[0]
		movies = data[1:]
		val = []
		for x in range(0, 100):
			val.append(0)
		for v in movies:
			val[int(v)] = 1
		return (user,val) 


	def get_signature(data):
		signature = [100000]*20
		for x in range(0, 100):
			if data[x] == 1:
				for i in range(0, 20):
					h = (3*x + 13*i)%100
					if signature[i] > h:
						signature[i] = h
		return signature

	def get_bands(data):
		user = data[0]
		sign = data[1]
		n= 4
		band =[((i,tuple(sign[i:i + n])),user) for i in range(0, len(sign), n)]
		return band

	def get_dictionary_input(data):
		data = data.split(",")
		return (data[0], data[1:])

	def jaccard_similarity(data):
		data = list(data)
		user1 = set(dict_inp[data[0]])
		user2 = set(dict_inp[data[1]])
		jaccard_similarity = float(len(user1.intersection(user2))) /float(len(user1.union(user2)))    
		return (data[0], (data[1], jaccard_similarity))

	def get_similar_users(data):
		user1 = data[0]
		user2 = data[1][0]
		return ((user1, (user2, data[1][1])), (user2, (user1, data[1][1])))

	def get_topFive_list(data):
		rec = []
		data.sort(key = lambda x: (-x[1], int(x[0][1:])))  
		val= data[:5]
		for v in val:
			rec.append(v[0])
		return sorted(rec, key=lambda x:-int(x[1:]))

	def movie_recommendations(data):
		users=data[1]
		result={}
		for i in users:
			movies=set(dict_inp[i])
			for m in movies:
				if m in result:
					result[m] += 1
				else:
					result[m] = 1
		result = sorted(result.items(), key=lambda x: (-x[1],int(x[0])))
		val = result[:3]
		final_list=[]
		for v in val: 
			rec = v[0]
			final_list.append(rec)
		return (data[0],final_list)

	#def sort(data):
	#	output=[]
	#	num=data[1]
	#	for i in num:
	#		output.append(int(i))
	#		output = sorted(output)
	#	return (data[0],output)


	movie_matrix = data.map(mapper).collect()
	dict_inp = dict(data.map(get_dictionary_input).collect())
	sign_matrix = sc.parallelize(movie_matrix).mapValues(get_signature)
	band = sign_matrix.flatMap(get_bands).collect()

	cand_pairs = sc.parallelize(band).groupByKey().mapValues(lambda x : list(x)).\
	filter(lambda x : len(x[1])>1).map(lambda x : x[1]).\
	flatMap(lambda x: itertools.combinations(x, 2)).\
	groupByKey().flatMap(lambda x: [(x[0], v) for v in set(x[1])])

	similarity = cand_pairs.map(jaccard_similarity).flatMap(get_similar_users).filter(lambda x: x[1][1]>0).groupByKey().mapValues(lambda x : list(x))

	final_list = similarity.mapValues(get_topFive_list).map(movie_recommendations).collect()

	finalOP = sorted(final_list, key = lambda x : int(x[0][1:]))

	fo = open(sys.argv[2], "w")
	for value in finalOP:
  		fo.write(value[0] +"," + ','.join(value[1]) + "\r\n")
	fo.close()


