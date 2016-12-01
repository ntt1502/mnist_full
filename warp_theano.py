import numpy as np
import theano
import theano.tensor as T

prediction = T.fvector()
target_var = T.ivector()

true_label = np.argmax(target_var) #theano.tensor.where
predicted_label = np.argmax(prediction)

f_y = T.cast(prediction[true_label], 'float32')
Y = 9
N = 0
err = 0

false_prediction = prediction[T.where(T.eq(target_var, 0), 1, 0)]

while True:
    i = theano.shared(np.random.randint(8))
    temp = T.cast(false_prediction[i], 'float32')
    N += 1
    if T.gt(temp + 1, f_y):
        rank = np.int_(np.floor(Y/N))
        L_rank = 0
        for i in range(rank):
            L_rank += 1/(i+1)
        err += L_rank*(temp + 1 - f_y)
        break
    if N > Y - 1:
        break

warp_loss = theano.function([prediction, target_var], err)

