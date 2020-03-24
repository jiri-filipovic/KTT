import pickle
import sys
import logging
import sklearn
import sklearn.ensemble
logging.disable(logging.WARNING)
logger = logging.getLogger('my-logger')
logger.propagate = False
filename = sys.argv[1]
loaded_model = pickle.load(open(filename, 'rb'))
configurations = eval(sys.argv[2]) #map(float,sys.argv[1].strip('[]').split(','))
#print(configurations)
predictionsTest = loaded_model.predict(configurations)
#predictionsTest = ','.join(predictionsTest)
print(predictionsTest)
