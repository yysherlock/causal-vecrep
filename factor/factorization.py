import json
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg

culinalg.init()


def normalize():
	"""
	normalize our cs vectors
	"""
	with open('copa_nomalized.json','wb') as outf:
		obj = json.loads(open('copa_matrix.json').read())
		
		for i in range(len(obj)):
			den = sum(obj[i])
			try:
				x = [item/den for item in obj[i]]
				obj[i] = x
			except:
				pass
		print len(obj)

		outf.write(json.dumps(obj))

normalize()
exit()
	
"""
Using svd to factorize the copa_matrix into 
cause/effect vector representations
"""

epsilon = 1e-3
data = json.loads(open('copa_matrix.json').read())
arr = np.asarray(np.array(data), np.float32)

arr[np.diag(np.diag(arr) < epsilon)] = 0.0
#print arr.shape[0]
#print np.identity(arr.shape[0])
#print gpuarray.to_gpu(arr)

a_gpu = gpuarray.to_gpu(arr + epsilon*np.identity(arr.shape[0]))
#u_gpu, s_gpu, vh_gpu = culinalg.svd(a_gpu)
u_gpu, s_gpu, vh_gpu = culinalg.svd(a_gpu, jobu='S', jobvt='S')

print u_gpu.shape
print vh_gpu.shape
print type(u_gpu.get())

with open('u.json','wb') as outf:
	outf.write(json.dumps(u_gpu.get().tolist()))

with open('v.json','wb') as outf:
        outf.write(json.dumps(vh_gpu.get().tolist()))

