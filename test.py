from matplotlib import pylab
import scipy.io
import text_parse
# output_path='/tmp/wiki/trymoves-model=hdp_topic+mult-K=5/1/AllocPrior.mat'
# path = scipy.io.loadmat('/tmp/wiki/trymoves-model=hdp_topic+mult-K=5/1/AllocPrior.mat')
# pylab.plot([1,2,3,4])
# pylab.show()

# 1. Parse the text documents - specify the data location
text_parse.main(["input/*.txt","input/*.txt"])

# 2. Run the algorithm on the newly created documents
# python run.py

# 3. Parse the outputs 
# python lam_parse.py output/lam.csv output/vocab.txt 15