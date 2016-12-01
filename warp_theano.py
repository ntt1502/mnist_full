import numpy as np
import theano
import theano.tensor as T

a = T.fvector() #prediction
b = T.ivector() #target_var

true_label = np.argmax(b) #theano.tensor.where
predicted_label = np.argmax(a)

f_y = T.cast(a[true_label], 'float32')
Y = 9
N = 0
err = 0
#c = a[np.where(b == 0)]
c = a[T.where(T.eq(b, 0), 1, 0)]
#x = theano.function([a, b], c)

while True:
    i = theano.shared(np.random.randint(8))
    temp = T.cast(c[i], 'float32')
#    c = np.delete(c, i)
    N += 1
    if T.gt(temp + 0.15, f_y):
        rank = np.int_(np.floor(Y/N))
        L_rank = 0
        for i in range(rank):
            L_rank += 1/(i+1)
        err += L_rank*(temp + 0.15 - f_y)
        break
    if N > Y - 1:
        break

warp_loss = theano.function([a, b], err)
