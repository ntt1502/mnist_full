import numpy as np
import theano
import theano.tensor as T

prediction = T.fvector()
target_var = T.ivector()

true_label = T.argmax(target_var) #theano.tensor.where
predicted_label = T.argmax(prediction)

f_y = T.cast(prediction[true_label], 'float32')
Y = theano.shared(9)
N = theano.shared(0)
err = theano.shared(0)

false_prediction = prediction[T.where(T.eq(target_var, 0), 1, 0)]

while True:
    i = theano.shared(np.random.randint(8))
    temp = T.cast(false_prediction[i], 'float32')
    N += 1
    if T.gt(temp + 1, f_y):
        rank = T.floor(Y/N)
        L_rank = theano.shared(0)
        for i in range(6):
            L_rank += 1/(i+1)
        err += L_rank*(temp + 1 - f_y)
        break
    if N > Y - 1:
        break


warp_loss = theano.function([prediction, target_var], err)

#diem nhan duoc sau score function
prediction_test = np.array(([0.1, 0.2, 0.12, 0.09, 0.02, 0.3, 0.3, 0.4, 0.16, 0.18]), dtype='float32')
#cac nhan duoc gan nhan hay khong
target_var_test = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype='int32')
a = warp_loss(prediction_test, target_var_test)
