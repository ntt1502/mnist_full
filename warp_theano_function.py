import numpy as np
import theano
import theano.tensor as T


def warp_loss(prediction, target_var):

    true_label = np.argmax(target_var) #theano.tensor.where
    predicted_label = np.argmax(prediction)

    # diem cua nhan dung
    f_y = T.cast(prediction[true_label], 'float32')
    # 10 label => Y = 9
    Y = 9
    N = 0
    err = 0
#    c = prediction[np.where(target_var == 0)]
    false_prediction = prediction[T.where(T.eq(target_var, 0), 1, 0)]
#    x = theano.function([prediction, target_var], false_prediction)

    while True:
        i = theano.shared(np.random.randint(8))
        # diem cua nhan da duoc pick random
        temp = T.cast(false_prediction[i], 'float32')
    #   false_prediction = np.delete(false_prediction, i)
        N += 1
        if T.gt(temp + 0.01, f_y):
            rank = np.int_(np.floor(Y/N))
            # add trong so L(rank)
            L_rank = 0
            for i in range(rank):
                L_rank += 1/(i+1)
            err += L_rank*(temp + 0.01 - f_y)
            break
        if N > Y - 1:
            break

#   loss = theano.function([prediction, target_var], err)
#   final_loss = loss(prediction, target_var)

    return err
